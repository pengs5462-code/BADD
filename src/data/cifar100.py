from __future__ import annotations

from typing import Tuple

import torchvision
import torchvision.transforms as transforms
import torch

from ..config import TrainConfig


def build_cifar100_loaders(cfg: TrainConfig) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Build CIFAR-100 train/test dataloaders using standard augmentations.
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root=cfg.data_root, train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR100(
        root=cfg.data_root, train=False, download=True, transform=transform_test
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=100,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return trainloader, testloader
