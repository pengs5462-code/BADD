import os
import argparse

import torch
import torch.optim as optim

from src.config import TrainConfig, ensure_dir, set_seed
from src.data.cifar100 import build_cifar100_loaders
from src.models import resnet32, shufflenetv2
from src.engine.utils import get_device, make_run_name
from src.engine.trainer import train_one_epoch
from src.engine.evaluator import top1_accuracy
from src.engine.logger import CSVLogger


def build_models(cfg: TrainConfig, device: torch.device):
    """
    arch:
      hetero: ShuffleNetV2 + ResNet32
      homo:   ResNet32 + ResNet32
    """
    if cfg.arch == "hetero":
        net1 = shufflenetv2().to(device)
        net2 = resnet32().to(device)
        name1, name2 = "Shuffle", "ResNet"
    elif cfg.arch == "homo":
        net1 = resnet32().to(device)
        net2 = resnet32().to(device)
        name1, name2 = "ResNet_A", "ResNet_B"
    else:
        raise ValueError("cfg.arch must be hetero or homo")
    return net1, net2, name1, name2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="dynamic_v17_11")
    parser.add_argument("--arch", type=str, default="hetero", choices=["hetero", "homo"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./experiments")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = TrainConfig(
        data_root=args.data_root,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        temperature=args.temperature,
        seed=args.seed,
        arch=args.arch,
        mode=args.mode,
    )

    ensure_dir(cfg.save_dir)
    set_seed(cfg.seed)
    device = get_device(cfg)

    trainloader, testloader = build_cifar100_loaders(cfg)
    net1, net2, name1, name2 = build_models(cfg, device)

    opt1 = optim.SGD(net1.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    opt2 = optim.SGD(net2.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched1 = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=cfg.epochs)
    sched2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=cfg.epochs)

    run_name = make_run_name(cfg)
    out_csv = os.path.join(cfg.save_dir, f"{run_name}.csv")
    logger = CSVLogger(out_csv_path=out_csv)

    for epoch in range(1, cfg.epochs + 1):
        train_stats = train_one_epoch(net1, net2, trainloader, opt1, opt2, epoch, cfg, device)
        sched1.step()
        sched2.step()

        acc1 = top1_accuracy(net1, testloader, device)
        acc2 = top1_accuracy(net2, testloader, device)

        row = {
            "epoch": epoch,
            "mode": cfg.mode,
            "arch": cfg.arch,
            f"test_acc_{name1}": acc1,
            f"test_acc_{name2}": acc2,
            **train_stats,
        }
        logger.log(row)

        if epoch % 10 == 0 or epoch > cfg.epochs - 10:
            print(
                f"epoch={epoch:03d} "
                f"{name1}={acc1:.2f} {name2}={acc2:.2f} "
                f"kl1={train_stats['kl_1']:.3f}"
            )

    logger.close()
    print(out_csv)


if __name__ == "__main__":
    main()
