import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from data import ContrastiveSegDataset
from utils import load_model, get_transform, prep_args
from modules import *

# 设置环境变量
os.environ["HYDRA_FULL_ERROR"] = "1"


# 获取特征
def get_feats(model, loader):
    all_feats = []
    for pack in tqdm(loader):
        img = pack["img"]
        feats = F.normalize(model.forward(img.cuda()).mean([2, 3]), dim=1)
        """
        non_blocking=True 参数可以在数据传输时不阻塞其他操作，提高效率。
        将这些特征添加到先前初始化的列表中
        """
        all_feats.append(feats.to("cpu", non_blocking=True))
    # .contiguous(): 确保返回的张量在内存中是连续的，这对于某些PyTorch操作来说可能是必要的，以避免潜在的性能问题。
    return torch.cat(all_feats, dim=0).contiguous()


@hydra.main(config_path="configs", config_name="train_config.yaml", version_base="1.1")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    # 程序创建必要的目录以存储处理后的数据和日志。
    pytorch_data_dir = cfg.pytorch_data_dir
    data_dir = os.path.join(cfg.output_root, "data")
    log_dir = os.path.join(cfg.output_root, "logs")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(pytorch_data_dir, "nns"), exist_ok=True)

    seed_everything(seed=0)

    print(data_dir)
    print(cfg.output_root)

    image_sets = ["val", "train"]
    dataset_names = ["directory"]
    crop_types = [None]

    # Uncomment these lines to run on custom datasets
    # dataset_names = ["directory"]
    # crop_types = [None]

    res = 224
    n_batches = 16
    # 数据集和模型准备
    if cfg.arch == "dino":
        from modules import DinoFeaturizer, LambdaLayer

        no_ap_model = nn.Sequential(
            DinoFeaturizer(20, cfg),
            LambdaLayer(lambda p: p[0]),
        ).cuda()
    else:
        cut_model = load_model(cfg.model_type, os.path.join(cfg.output_root, "data")).cuda()
        no_ap_model = nn.Sequential(*list(cut_model.children())[:-1]).cuda()

    # 用于支持多GPU环境，将模型封装以进行并行计算。
    par_model = nn.DataParallel(no_ap_model)
    # 数据加载和特征提取
    for crop_type in crop_types:
        for image_set in image_sets:
            for dataset_name in dataset_names:
                nice_dataset_name = cfg.dir_dataset_name if dataset_name == "directory" else dataset_name
                # 构建特征缓存的文件的路径和文件名
                feature_cache_file = os.path.join(
                    pytorch_data_dir,
                    "nns",
                    "nns_{}_{}_{}_{}_{}.npz".format(cfg.model_type, nice_dataset_name, image_set, crop_type, res),
                )

                if not os.path.exists(feature_cache_file):
                    print("{} not found, computing".format(feature_cache_file))
                    dataset = ContrastiveSegDataset(
                        pytorch_data_dir=pytorch_data_dir,
                        dataset_name=dataset_name,
                        crop_type=crop_type,
                        image_set=image_set,
                        transform=get_transform(res, False, None),
                        target_transform=get_transform(res, True, None),
                        cfg=cfg,
                    )

                    loader = DataLoader(dataset, 224, shuffle=False, num_workers=cfg.num_workers, pin_memory=False)

                    with torch.no_grad():
                        normed_feats = get_feats(par_model, loader)
                        all_nns = []
                        step = normed_feats.shape[0] // n_batches
                        print(normed_feats)
                        for i in tqdm(range(0, normed_feats.shape[0], step)):
                            torch.cuda.empty_cache()
                            batch_feats = normed_feats[i : i + step, :]
                            pairwise_sims = torch.einsum("nf,mf->nm", batch_feats, normed_feats)
                            print("+++++++++pairwise_sims.shape:", pairwise_sims.shape)
                            """all_nns.append(torch.topk(pairwise_sims, 30)[1])
                            从30修改为2可以顺利运行，推测可能是因为类别数量超过了实际，应该测试一下当这个参数为5的时候是否可行
                            """
                            all_nns.append(torch.topk(pairwise_sims, 2)[1])
                            del pairwise_sims
                        nearest_neighbors = torch.cat(all_nns, dim=0)

                        np.savez_compressed(feature_cache_file, nns=nearest_neighbors.numpy())
                        print("Saved NNs", cfg.model_type, nice_dataset_name, image_set)


if __name__ == "__main__":
    prep_args()
    my_app()
