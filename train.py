import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from src.config import TrainConfig, ensure_dir
from src.models import resnet32, shufflenetv2
from src.distill.loss import distill_loss


def build_loaders(cfg: TrainConfig):
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

    trainset = torchvision.datasets.CIFAR100(root=cfg.data_root, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root=cfg.data_root, train=False, download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=cfg.num_workers, pin_memory=True
    )
    return trainloader, testloader


def build_models(arch: str, device: torch.device):
    """
    arch:
      hetero: ShuffleNetV2 + ResNet32
      homo:   ResNet32 + ResNet32
    """
    if arch == "hetero":
        net1 = shufflenetv2().to(device)  # student A
        net2 = resnet32().to(device)      # student B
        name1, name2 = "Shuffle", "ResNet"
    elif arch == "homo":
        net1 = resnet32().to(device)
        net2 = resnet32().to(device)
        name1, name2 = "ResNet_A", "ResNet_B"
    else:
        raise ValueError("arch must be hetero or homo")
    return net1, net2, name1, name2


@torch.no_grad()
def evaluate(net: torch.nn.Module, loader, device: torch.device) -> float:
    net.eval()
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        out = net(inputs)
        pred = out.argmax(dim=1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    return 100.0 * correct / total


def train_one_epoch(net1, net2, loader, opt1, opt2, epoch: int, mode: str, cfg: TrainConfig, device: torch.device):
    net1.train()
    net2.train()

    loss1_meter, loss2_meter = [], []
    w1_meter, w2_meter = [], []
    kl1_meter, kl2_meter = [], []
    extra_alpha = []

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        out1 = net1(inputs)
        out2 = net2(inputs)

        l1, s1 = distill_loss(out1, out2, labels, epoch, mode, T=3.0, epochs_total=cfg.epochs, device=device)
        l2, s2 = distill_loss(out2, out1, labels, epoch, mode, T=3.0, epochs_total=cfg.epochs, device=device)

        opt1.zero_grad()
        opt2.zero_grad()
        l1.backward(retain_graph=True)
        l2.backward()
        opt1.step()
        opt2.step()

        loss1_meter.append(l1.item())
        loss2_meter.append(l2.item())

        # weight stats are present for weight-based modes; for others we still output mean/min/max in loss.py
        w1_meter.append(s1.get("mean", 1.0))
        w2_meter.append(s2.get("mean", 1.0))
        kl1_meter.append(s1.get("kl", 0.0))
        kl2_meter.append(s2.get("kl", 0.0))

        if "alpha" in s1:
            extra_alpha.append(s1["alpha"])

    out = {
        "train_loss_1": float(np.mean(loss1_meter)),
        "train_loss_2": float(np.mean(loss2_meter)),
        "weight_1": float(np.mean(w1_meter)),
        "weight_2": float(np.mean(w2_meter)),
        "kl_1": float(np.mean(kl1_meter)),
        "kl_2": float(np.mean(kl2_meter)),
    }
    if len(extra_alpha) > 0:
        out["alpha_mean"] = float(np.mean(extra_alpha))
    return out


def run(mode: str, arch: str, cfg: TrainConfig):
    device = torch.device(cfg.device)
    ensure_dir(cfg.save_dir)

    trainloader, testloader = build_loaders(cfg)
    net1, net2, name1, name2 = build_models(arch, device)

    opt1 = optim.SGD(net1.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    opt2 = optim.SGD(net2.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched1 = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=cfg.epochs)
    sched2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=cfg.epochs)

    logs = []
    for epoch in range(1, cfg.epochs + 1):
        train_stats = train_one_epoch(net1, net2, trainloader, opt1, opt2, epoch, mode, cfg, device)
        sched1.step()
        sched2.step()

        acc1 = evaluate(net1, testloader, device)
        acc2 = evaluate(net2, testloader, device)

        row = {
            "epoch": epoch,
            f"test_acc_{name1}": acc1,
            f"test_acc_{name2}": acc2,
            "train_loss_1": train_stats["train_loss_1"],
            "train_loss_2": train_stats["train_loss_2"],
            f"weight_{name1}_learns": train_stats["weight_1"],
            f"weight_{name2}_learns": train_stats["weight_2"],
            f"kl_{name1}_div_{name2}": train_stats["kl_1"],
            f"kl_{name2}_div_{name1}": train_stats["kl_2"],
        }
        if "alpha_mean" in train_stats:
            row["alpha_mean"] = train_stats["alpha_mean"]

        logs.append(row)

        if epoch % 10 == 0 or epoch > cfg.epochs - 10:
            msg = f"Ep {epoch:03d} | {name1}: {acc1:.2f}% | {name2}: {acc2:.2f}% | KL: {row[f'kl_{name1}_div_{name2}']:.3f}"
            if "alpha_mean" in row:
                msg += f" | alpha: {row['alpha_mean']:.4f}"
            print(msg)

    df = pd.DataFrame(logs)
    out_csv = os.path.join(cfg.save_dir, f"{arch}_{mode}_logs.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved logs to: {out_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="dynamic_v17_11",
                        help="baseline/dynamic_v5/v8/v9/v10/v11/v13/v14/v15/v17/v17_11/v18/adm/kdcl/okddip/odkd")
    parser.add_argument("--arch", type=str, default="hetero", choices=["hetero", "homo"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./experiments")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    cfg = TrainConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        data_root=args.data_root,
        save_dir=args.save_dir,
        num_workers=args.num_workers,
    )
    run(args.mode, args.arch, cfg)


if __name__ == "__main__":
    main()