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

# 设置环境变量
os.environ["HYDRA_FULL_ERROR"] = "1"


def _random_crops(img, size, seed, n):
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
        print("++++++++++++++++++++++++++img.shape:", img.shape)
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
                return self.five_crops(i, x)

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
        batch = self.dataset[item]
        imgs = batch["img"]
        labels = batch["label"]
        for crop_num, (img, label) in enumerate(zip(imgs, labels)):
            img_num = item * 5 + crop_num
            img_arr = img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            label_arr = (label + 1).unsqueeze(0).permute(1, 2, 0).to("cpu", torch.uint8).numpy().squeeze(-1)
            Image.fromarray(img_arr).save(os.path.join(self.img_dir, "{}.jpg".format(img_num)), "JPEG")
            Image.fromarray(label_arr).save(os.path.join(self.label_dir, "{}.png".format(img_num)), "PNG")
        return True

    def __len__(self):
        return len(self.dataset)


@hydra.main(config_path="configs", config_name="train_config.yaml", version_base="1.1")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
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
    print("Done.")
