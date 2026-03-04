import os
import argparse
import logging
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from src.models import resnet32, shufflenetv2


def setup_logger() -> logging.Logger:
    """
    Configure a basic console logger.
    Using logging instead of print improves reproducibility and code professionalism.
    """
    logger = logging.getLogger("alpha_study")
    if logger.handlers:
        return logger  # avoid duplicated handlers in interactive runs
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def parse_alpha_list(s: str) -> List[float]:
    """
    Parse a comma-separated alpha list string into floats.
    Example: "0.465,0.475,0.485"
    """
    items = [x.strip() for x in s.split(",") if x.strip()]
    return [float(x) for x in items]


def build_loaders(data_root: str, batch_size: int, num_workers: int):
    """
    Build CIFAR-100 dataloaders with standard augmentation used in CIFAR training.
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761)
        ),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761)
        ),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root=data_root, train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR100(
        root=data_root, train=False, download=True, transform=transform_test
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=100,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return trainloader, testloader


@torch.no_grad()
def evaluate(net: torch.nn.Module, testloader, device: torch.device) -> float:
    """
    Compute top-1 accuracy on the test set.
    """
    net.eval()
    correct = 0
    total = 0
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        out = net(inputs)
        pred = out.argmax(dim=1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    return 100.0 * correct / total


def get_dml_loss_with_alpha(
    logits_student: torch.Tensor,
    logits_teacher: torch.Tensor,
    target: torch.Tensor,
    epoch: int,
    alpha: float,
    *,
    temperature: float = 3.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Alpha-parameterized V17-family weighting (for sensitivity study).

    Weight formula:
        conf_i = max softmax(logits_teacher)_i
        mu_B = mean(conf_i over batch)
        w_i = 1 + alpha * (conf_i - mu_B)
        w_i is clamped into [0.8, 1.2]
        warmup: w = 1 + (w - 1) * rampup, rampup = clip(epoch/10, 0, 1)

    Returns:
        total_loss: cross-entropy + weighted KL
        stats: weight statistics and an auxiliary KL magnitude for logging/analysis
    """
    ce_loss = F.cross_entropy(logits_student, target)

    # Sample-wise KL divergence (student || teacher) with temperature scaling.
    log_p_s = F.log_softmax(logits_student / temperature, dim=1)
    p_t = F.softmax(logits_teacher / temperature, dim=1)
    kl_per_sample = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=1) * (temperature ** 2)

    with torch.no_grad():
        probs_teacher = F.softmax(logits_teacher, dim=1)
        conf_teacher, _ = probs_teacher.max(dim=1)
        batch_mean = conf_teacher.mean()

        dynamic_weight = 1.0 + alpha * (conf_teacher - batch_mean)
        dynamic_weight = torch.clamp(dynamic_weight, min=0.8, max=1.2)

        rampup = float(np.clip(epoch / 10.0, 0.0, 1.0))
        final_weight = 1.0 + (dynamic_weight - 1.0) * rampup

        stats = {
            "weight_mean": float(final_weight.mean().item()),
            "weight_min": float(final_weight.min().item()),
            "weight_max": float(final_weight.max().item()),
            # This is the mean of per-sample KL, useful as a scale indicator.
            "kl_mean": float(kl_per_sample.mean().item()),
        }

    distillation_loss = (final_weight.detach() * kl_per_sample).mean()
    total_loss = ce_loss + distillation_loss
    return total_loss, stats


def run_experiment_for_alpha(
    alpha: float,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    data_root: str,
    save_dir: str,
    num_workers: int,
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[float, float]:
    """
    Run a single alpha experiment under heterogeneous peers:
        net1: ShuffleNetV2
        net2: ResNet32

    Outputs:
        One CSV file per alpha with epoch-wise logs.
    """
    logger.info("Starting alpha experiment: alpha=%.6f", alpha)

    trainloader, testloader = build_loaders(data_root, batch_size, num_workers)

    net1 = shufflenetv2().to(device)
    net2 = resnet32().to(device)

    opt1 = optim.SGD(net1.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    opt2 = optim.SGD(net2.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched1 = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=epochs)
    sched2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=epochs)

    logs = []
    best_acc1 = 0.0
    best_acc2 = 0.0

    for epoch in range(1, epochs + 1):
        net1.train()
        net2.train()

        loss_meter = []
        w_mean_meter = []
        w_min_meter = []
        w_max_meter = []

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            out1 = net1(inputs)
            out2 = net2(inputs)

            # Symmetric online distillation
            l1, s1 = get_dml_loss_with_alpha(out1, out2, labels, epoch, alpha)  # Shuffle learns Res
            l2, _s2 = get_dml_loss_with_alpha(out2, out1, labels, epoch, alpha)  # Res learns Shuffle

            opt1.zero_grad()
            opt2.zero_grad()
            l1.backward(retain_graph=True)
            l2.backward()
            opt1.step()
            opt2.step()

            loss_meter.append(l1.item())
            w_mean_meter.append(s1["weight_mean"])
            w_min_meter.append(s1["weight_min"])
            w_max_meter.append(s1["weight_max"])

        sched1.step()
        sched2.step()

        acc1 = evaluate(net1, testloader, device)
        acc2 = evaluate(net2, testloader, device)

        best_acc1 = max(best_acc1, acc1)
        best_acc2 = max(best_acc2, acc2)

        row = {
            "epoch": epoch,
            "alpha": alpha,
            "test_acc_shuffle": acc1,
            "test_acc_res": acc2,
            "best_acc_shuffle_so_far": best_acc1,
            "best_acc_res_so_far": best_acc2,
            "train_loss": float(np.mean(loss_meter)),
            "weight_mean": float(np.mean(w_mean_meter)),
            "weight_min": float(np.mean(w_min_meter)),
            "weight_max": float(np.mean(w_max_meter)),
        }
        logs.append(row)

        if epoch % 10 == 0 or epoch > epochs - 10:
            logger.info(
                "alpha=%.6f epoch=%03d acc_shuffle=%.2f acc_res=%.2f "
                "w_range=[%.3f, %.3f] best_shuffle=%.2f",
                alpha, epoch, acc1, acc2, row["weight_min"], row["weight_max"], best_acc1
            )

    df = pd.DataFrame(logs)
    out_path = os.path.join(save_dir, f"log_alpha_{alpha}.csv")
    df.to_csv(out_path, index=False)
    logger.info("Saved per-alpha logs: %s", out_path)

    return best_acc1, best_acc2


def main():
    logger = setup_logger()

    parser = argparse.ArgumentParser(description="Alpha sensitivity study for V17-family weighting.")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./experiments/alpha_study")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--alphas", type=str, default="0.465,0.475,0.485,0.495,0.505,0.5")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    alpha_values = parse_alpha_list(args.alphas)
    logger.info("Alpha study started")
    logger.info("Alphas: %s", alpha_values)
    logger.info("Save directory: %s", args.save_dir)

    summary_results = []

    for alpha_val in alpha_values:
        try:
            best_s, best_r = run_experiment_for_alpha(
                alpha_val,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                data_root=args.data_root,
                save_dir=args.save_dir,
                num_workers=args.num_workers,
                device=device,
                logger=logger,
            )

            summary_results.append({
                "alpha": alpha_val,
                "best_acc_shuffle": best_s,
                "best_acc_res": best_r,
            })

            summary_df = pd.DataFrame(summary_results).sort_values("alpha")
            summary_path = os.path.join(args.save_dir, "summary_accuracy_vs_alpha.csv")
            summary_df.to_csv(summary_path, index=False)
            logger.info("Updated summary: %s", summary_path)

        except Exception as e:
            logger.exception("Failed at alpha=%.6f due to error: %s", alpha_val, str(e))
            continue

    logger.info("Alpha study finished")
    logger.info("Final summary path: %s", os.path.join(args.save_dir, "summary_accuracy_vs_alpha.csv"))


if __name__ == "__main__":
    main()