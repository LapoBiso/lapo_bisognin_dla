import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader
import torch
from torchvision.datasets import GTSRB
from pathlib import Path

MODE_DICT = {
    "bilinear": T.InterpolationMode.BILINEAR,
    "bicubic": T.InterpolationMode.BICUBIC,
}

def is_dataset_present(ds_path) -> bool:
    """
    Check if dataset has been already downloaded
    """
    pt = Path(ds_path)
    return pt.exists()

def transform_data(cfg, path):
    """
    define data transformation pipeline
    """
    download = not is_dataset_present(path)
    size = cfg.model.get('full_input')
    crop_size = cfg.model.get('input_size')
    mode = MODE_DICT.get(cfg.model.get('mode'))

    transform = T.Compose([
        T.Resize(size, interpolation = mode),
        T.RandomCrop((crop_size, crop_size)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    center_transform = T.Compose([
        T.Resize(size, interpolation = mode),
        T.CenterCrop(crop_size),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ds_train = GTSRB(path, split="train", download=download, transform=transform)
    ds_val = GTSRB(path, split="train",  download=download, transform=center_transform)
    ds_test = GTSRB(path, split="test",  download=download, transform=center_transform)
    return ds_train, ds_val, ds_test

def build_std_dataloader(cfg, path):
    """
    build dataloader for each dataset
    """
    ds_train, ds_val, ds_test = transform_data(cfg, path)
    generator = torch.Generator().manual_seed(42)
    train_idx, val_idx = torch.utils.data.random_split(range(len(ds_train)), [0.8, 0.2], generator=generator)
    ds_train = torch.utils.data.Subset(ds_train, train_idx.indices)
    ds_val   = torch.utils.data.Subset(ds_val, val_idx.indices)

    batch_size = cfg.train.get('batch_size')
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False)
    dl_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False)
    
    return dl_train, dl_val, dl_test