"""
PyTorch Lightning 是一个独立于 PyTorch 的库，提供了许多便利的功能来简化深度学习模型的训练，其中包括设置随机种子以确保实验可复现。
OmegaConf是一个基于YAML的层次化配置系统，支持从多个来源（如文件、命令行参数、环境变量）合并配置
"""
from data import ContrastiveSegDataset
from utils import ToTargetTensor, prep_args
import os
import torch
from PIL import Image
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms.functional import five_crop, get_image_size, crop
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf


def _random_crops(img, size, seed, n):
    """自定义函数：将传入的图像进行四角裁剪和中心裁剪
    如果img是torch张量， 维度应满足[..., H, W]，其中...表示任意数量的维度
    .. note::
        这个转换会返回一个图像元组，输入和目标数据集的数量可能不匹配
    Args:
        img (PIL Image or Tensor): 需要裁剪的图像
        size (sequence or int): 期望的输出尺寸
            如果size是int，输出尺寸将是正方形(size, size)。
            如果size是sequence，输出尺寸将是(size[0], size[1])。
        seed (int): 随机种子
        n (int): 裁剪数量
    Returns:
        tuple: 裁剪后的图像元组tuple (tl, tr, bl, br, center)
                对应左上角、右上角、左下角、右下角和中心裁剪
    """
    if isinstance(size, int):
        size = (int(size), int(size))
    elif isinstance(size, (tuple, list)) and len(size) == 1:
        size = (size[0], size[0])
    if len(size) != 2:
        raise ValueError("Please provide only two dimensions (h, w) for size.")

    image_width, image_height = get_image_size(img)
    crop_height, crop_width = size
    if crop_width > image_width or crop_height > image_height:
        msg = "Requested crop size {} is bigger than input size {}"
        raise ValueError(msg.format(size, (image_height, image_width)))

    images = []
    for _ in range(n):
        seed1 = hash((seed, _, 0))
        seed2 = hash((seed, _, 1))
        crop_height, crop_width = int(crop_height), int(crop_width)

        top = seed1 % (image_height - crop_height)
        left = seed2 % (image_width - crop_width)
        images.append(crop(img, top, left, crop_height, crop_width))

    return images


class RandomCropComputer(Dataset):
    def _get_size(self, img):
        if len(img.shape) == 3:
            return [int(img.shape[1] * self.crop_ratio), int(img.shape[2] * self.crop_ratio)]
        elif len(img.shape) == 2:
            return [int(img.shape[0] * self.crop_ratio), int(img.shape[1] * self.crop_ratio)]
        else:
            raise ValueError("Bad image shape {}".format(img.shape))

    def random_crops(self, i, img):
        return _random_crops(img, self._get_size(img), i, 5)

    def five_crops(self, i, img):
        return five_crop(img, self._get_size(img))

    def __init__(self, cfg, dataset_name, img_set, crop_type, crop_ratio):
        self.pytorch_data_dir = cfg.pytorch_data_dir
        self.crop_ratio = crop_ratio
        self.save_dir = os.path.join(
            cfg.pytorch_data_dir, "cropped", "{}_{}_crop_{}".format(dataset_name, crop_type, crop_ratio)
        )
        self.img_set = img_set
        self.dataset_name = dataset_name
        self.cfg = cfg

        self.img_dir = os.path.join(self.save_dir, "img", img_set)
        self.label_dir = os.path.join(self.save_dir, "label", img_set)
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.label_dir, exist_ok=True)

        if crop_type == "random":

            def cropper(i, x):
                return self.random_crops(i, x)

        elif crop_type == "five":

            def cropper(i, x):
                return five_crop(i, x)

        else:
            raise ValueError("Unknown crop type {}".format(crop_type))

        self.dataset = ContrastiveSegDataset(
            cfg.pytorch_data_dir,
            dataset_name,
            None,
            img_set,
            T.ToTensor(),
            ToTargetTensor(),
            cfg=cfg,
            num_neighbors=cfg.num_neighbors,
            pos_labels=False,
            pos_images=False,
            mask=False,
            aug_geometric_transform=None,
            aug_photometric_transform=None,
            extra_transform=cropper,
        )

    def __getitem__(self, item):
        """该方法从数据集中获取一个元素（一组图像和标签），将它们转换为JPEG和PNG格式并保存。"""
        batch = self.dataset[item]
        imgs = batch["img"]
        labels = batch["label"]
        for crop_num, (img, label) in enumerate(zip(imgs, labels)):
            """img_num = item * 5 + crop_num:
            假设每个 item 包含5个裁剪图像，所以使用 item * 5 来确定起始编号，crop_num 用于为这五个裁剪图中的每一个提供唯一的序号。
            """
            img_num = item * 5 + crop_num
            """
            将图像张量（预期为0到1的浮点数）转换为0到255的整数范围，适用于JPEG格式的保存。
            permute(1, 2, 0) 调整维度顺序以适配 PIL 和 NumPy 的图像格式（从通道优先变为高宽优先）。
            最后将数据移动到 CPU 并转换为 numpy 数组。
            """
            img_arr = img.mul(255).add(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            """对标签进行处理，通常标签张量是单通道的。这里首先将标签的值增加1（可能用于调整标签值的范围），
            然后增加一个新的维度，调整维度顺序，并转换为 numpy 数组。squeeze(-1) 用于移除单维度条目，最终形成二维数组。
            """
            label_arr = (label + 1).unsqueeze(0).permute(1, 2, 0).to("cpu", torch.uint8).numpy().squeeze(-1)
            Image.fromarray(img_arr).save(os.path.join(self.img_dir, "{}.jpg".format(img_num)), "JPEG")
            Image.fromarray(label_arr).save(os.path.join(self.label_dir, "{}.png".format(img_num)), "PNG")
        return True

    def __len__(self):
        return len(self.dataset)


@hydra.main(config_path="configs", config_name="train_config.yaml", version_base="1.1")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    """这里的 workers=True 表示也为数据加载器的工作进程设置随机种子。"""
    seed_everything(seed=0, workers=True)

    dataset_names = ["directory"]
    img_sets = ["train", "val"]
    crop_types = ["five"]
    crop_ratios = [0.5]

    for crop_ratio in crop_ratios:
        for crop_type in crop_types:
            for dataset_name in dataset_names:
                for img_set in img_sets:
                    dataset = RandomCropComputer(cfg, dataset_name, img_set, crop_type, crop_ratio)
                    loader = DataLoader(dataset, 1, shuffle=False, num_workers=cfg.num_workers, collate_fn=lambda x: x)
                    for _ in tqdm(loader):
                        pass


if __name__ == "__main__":
    prep_args()  # 统一参数格式
    my_app()
