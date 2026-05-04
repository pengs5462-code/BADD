import os
import shutil
from pathlib import Path
from typing import Tuple, Optional, List

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder


class CUB200Raw(Dataset):
    """CUB-200-2011 raw-format loader.

    Expected structure:
      root/CUB_200_2011/images.txt
      root/CUB_200_2011/image_class_labels.txt
      root/CUB_200_2011/train_test_split.txt
      root/CUB_200_2011/images/<class>/<image>.jpg
    """
    def __init__(self, root: str, train: bool, transform=None):
        self.root = Path(root)
        cub_root = self.root / "CUB_200_2011"
        if not cub_root.exists():
            cub_root = self.root
        self.cub_root = cub_root
        self.transform = transform
        split_value = 1 if train else 0

        images = {}
        with open(cub_root / "images.txt", "r", encoding="utf-8") as f:
            for line in f:
                idx, rel = line.strip().split(maxsplit=1)
                images[int(idx)] = rel
        labels = {}
        with open(cub_root / "image_class_labels.txt", "r", encoding="utf-8") as f:
            for line in f:
                idx, lab = line.strip().split()
                labels[int(idx)] = int(lab) - 1
        splits = {}
        with open(cub_root / "train_test_split.txt", "r", encoding="utf-8") as f:
            for line in f:
                idx, sp = line.strip().split()
                splits[int(idx)] = int(sp)

        self.samples = []
        for idx, rel in images.items():
            if splits[idx] == split_value:
                self.samples.append((str(cub_root / "images" / rel), labels[idx]))
        self.classes = sorted({s[1] for s in self.samples})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(dataset: str, image_size: int):
    dataset = dataset.lower()
    if dataset == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        train_tf = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        val_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        return train_tf, val_tf

    # ImageNet-style normalization for ImageNet-100, Tiny-ImageNet, CUB.
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    if image_size <= 64:
        train_tf = T.Compose([
            T.RandomCrop(image_size, padding=max(4, image_size // 8)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        val_tf = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    else:
        train_tf = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        val_tf = T.Compose([
            T.Resize(int(image_size * 256 / 224)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    return train_tf, val_tf


def build_datasets(name: str, root: str, image_size: int = 224, download: bool = False):
    name = name.lower()
    train_tf, val_tf = get_transforms(name, image_size)

    if name == "cifar100":
        train_set = torchvision.datasets.CIFAR100(root=root, train=True, download=download, transform=train_tf)
        val_set = torchvision.datasets.CIFAR100(root=root, train=False, download=download, transform=val_tf)
        num_classes = 100
        return train_set, val_set, num_classes

    if name in {"tiny-imagenet", "tiny_imagenet"}:
        # Expected after preparation: root/tiny-imagenet-200/train/<class> and root/tiny-imagenet-200/val/<class>
        base = Path(root)
        if (base / "tiny-imagenet-200").exists():
            base = base / "tiny-imagenet-200"
        train_set = ImageFolder(str(base / "train"), transform=train_tf)
        val_set = ImageFolder(str(base / "val"), transform=val_tf)
        return train_set, val_set, len(train_set.classes)

    if name in {"imagenet100", "imagenet-100"}:
        base = Path(root)
        train_set = ImageFolder(str(base / "train"), transform=train_tf)
        val_set = ImageFolder(str(base / "val"), transform=val_tf)
        return train_set, val_set, len(train_set.classes)

    if name in {"cub200", "cub-200", "cub200-2011", "cub-200-2011"}:
        base = Path(root)
        if (base / "train").exists() and (base / "val").exists():
            train_set = ImageFolder(str(base / "train"), transform=train_tf)
            val_set = ImageFolder(str(base / "val"), transform=val_tf)
            return train_set, val_set, len(train_set.classes)
        train_set = CUB200Raw(root, train=True, transform=train_tf)
        val_set = CUB200Raw(root, train=False, transform=val_tf)
        return train_set, val_set, 200

    raise ValueError(f"Unsupported dataset: {name}")


def build_loaders(name: str, root: str, image_size: int, batch_size: int, workers: int,
                  download: bool = False):
    train_set, val_set, num_classes = build_datasets(name, root, image_size=image_size, download=download)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers,
                              pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers,
                            pin_memory=True, drop_last=False)
    return train_loader, val_loader, num_classes


def prepare_tiny_imagenet_val(tiny_root: str) -> None:
    """Rearrange official Tiny-ImageNet validation images into ImageFolder format.

    Official format:
      val/images/*.JPEG
      val/val_annotations.txt
    Target:
      val/<wnid>/*.JPEG
    This operation is idempotent.
    """
    root = Path(tiny_root)
    val_dir = root / "val"
    annot = val_dir / "val_annotations.txt"
    img_dir = val_dir / "images"
    if not annot.exists() or not img_dir.exists():
        print("Tiny-ImageNet val already appears prepared or files are missing.")
        return
    mapping = {}
    with open(annot, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                mapping[parts[0]] = parts[1]
    for fname, cls in mapping.items():
        dst_dir = val_dir / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        src = img_dir / fname
        dst = dst_dir / fname
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))
    try:
        img_dir.rmdir()
    except OSError:
        pass
