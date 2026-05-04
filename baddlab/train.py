import argparse
import copy
import os
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm

from .datasets import build_loaders
from .models import build_peer_models
from .losses import LossConfig, mutual_kd_loss
from .utils import set_seed, load_config, save_json, ensure_dir, accuracy_top1, AverageMeter


def _cfg_get(cfg: Dict[str, Any], key: str, default=None):
    cur = cfg
    for part in key.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def build_optimizer(model, cfg: Dict[str, Any]):
    opt_cfg = cfg.get("optimizer", {})
    lr = float(opt_cfg.get("lr", 0.1))
    wd = float(opt_cfg.get("weight_decay", 5e-4))
    momentum = float(opt_cfg.get("momentum", 0.9))
    nesterov = bool(opt_cfg.get("nesterov", True))
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd, nesterov=nesterov)


def build_scheduler(optimizer, cfg: Dict[str, Any], epochs: int):
    sch_cfg = cfg.get("scheduler", {"name": "cosine"})
    name = str(sch_cfg.get("name", "cosine")).lower()
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "multistep":
        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(sch_cfg.get("milestones", [int(epochs * 0.5), int(epochs * 0.75)])),
            gamma=float(sch_cfg.get("gamma", 0.1)),
        )
    raise ValueError(f"Unknown scheduler: {name}")


def evaluate(model_a, model_b, loader, device):
    model_a.eval()
    model_b.eval()
    correct_a = 0
    correct_b = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out_a = model_a(x)
            out_b = model_b(x)
            correct_a += (out_a.argmax(dim=1) == y).sum().item()
            correct_b += (out_b.argmax(dim=1) == y).sum().item()
            total += y.numel()
    return 100.0 * correct_a / total, 100.0 * correct_b / total


@torch.no_grad()
def _update_ema(ema_model, online_model, decay: float):
    for p_ema, p_online in zip(ema_model.parameters(), online_model.parameters()):
        p_ema.mul_(decay).add_(p_online, alpha=(1.0 - decay))
    for b_ema, b_online in zip(ema_model.buffers(), online_model.buffers()):
        b_ema.copy_(b_online)


def train_one_run(cfg: Dict[str, Any], seed: int, mode: str, output_root: str, resume: str = None):
    set_seed(seed, deterministic=bool(_cfg_get(cfg, "runtime.deterministic", True)))
    device = torch.device(_cfg_get(cfg, "runtime.device", "cuda") if torch.cuda.is_available() else "cpu")

    dataset_name = _cfg_get(cfg, "dataset.name")
    data_root = _cfg_get(cfg, "dataset.root", "./data")
    image_size = int(_cfg_get(cfg, "dataset.image_size", 32))
    batch_size = int(_cfg_get(cfg, "training.batch_size", 128))
    workers = int(_cfg_get(cfg, "runtime.workers", 4))
    epochs = int(_cfg_get(cfg, "training.epochs", 300))
    download = bool(_cfg_get(cfg, "dataset.download", False))

    train_loader, val_loader, num_classes = build_loaders(dataset_name, data_root, image_size, batch_size, workers, download=download)

    model_a_name = _cfg_get(cfg, "models.peer_a", "shufflenetv2")
    model_b_name = _cfg_get(cfg, "models.peer_b", "resnet32")
    pretrained = bool(_cfg_get(cfg, "models.pretrained", False))
    model_a, model_b = build_peer_models(model_a_name, model_b_name, num_classes, pretrained=pretrained)
    model_a, model_b = model_a.to(device), model_b.to(device)
    ema_a = copy.deepcopy(model_a).to(device).eval()
    ema_b = copy.deepcopy(model_b).to(device).eval()
    for p in ema_a.parameters():
        p.requires_grad_(False)
    for p in ema_b.parameters():
        p.requires_grad_(False)
    ema_decay = float(_cfg_get(cfg, "ema.decay", 0.999))

    opt_a = build_optimizer(model_a, cfg)
    opt_b = build_optimizer(model_b, cfg)
    sch_a = build_scheduler(opt_a, cfg, epochs)
    sch_b = build_scheduler(opt_b, cfg, epochs)

    loss_cfg = LossConfig(
        mode=mode,
        tau=float(_cfg_get(cfg, "loss.tau", 3.0)),
        alpha=float(_cfg_get(cfg, "loss.alpha", 0.495)),
        clip_min=float(_cfg_get(cfg, "loss.clip_min", 0.8)),
        clip_max=float(_cfg_get(cfg, "loss.clip_max", 1.2)),
        warmup_epochs=int(_cfg_get(cfg, "loss.warmup_epochs", 10)),
        zscore_eps=float(_cfg_get(cfg, "loss.zscore_eps", 1e-6)),
        random_match_residual_std=bool(_cfg_get(cfg, "loss.random_match_residual_std", True)),
        top_fraction=float(_cfg_get(cfg, "analysis.top_fraction", 0.2)),
    )

    run_dir = Path(output_root) / dataset_name / f"{model_a_name}_vs_{model_b_name}" / mode / f"seed_{seed}"
    ensure_dir(str(run_dir))
    save_json({"seed": seed, "mode": mode, "config": cfg}, str(run_dir / "run_config.json"))

    epoch_rows = []
    trace_rows = []
    best_a = 0.0
    best_b = 0.0
    cumulative = {
        "a_abs_reallocated_kl": 0.0,
        "b_abs_reallocated_kl": 0.0,
        "a_abs_reallocated_grad": 0.0,
        "b_abs_reallocated_grad": 0.0,
    }
    global_step = 0
    log_interval = int(_cfg_get(cfg, "analysis.log_interval", 20))
    max_trace_rows = int(_cfg_get(cfg, "analysis.max_trace_rows", 0))
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model_a.train(); model_b.train()
        meters = {name: AverageMeter() for name in [
            "loss_a", "loss_b", "ce_a", "ce_b", "kd_a", "kd_b", "unweighted_kd_a", "unweighted_kd_b",
            "weight_mean_a", "weight_mean_b", "weight_std_a", "weight_std_b", "weight_min_a", "weight_min_b", "weight_max_a", "weight_max_b",
            "msp_mean_a", "msp_mean_b", "msp_std_a", "msp_std_b", "residual_std_a", "residual_std_b",
            "grad_share_top_unweighted_a", "grad_share_top_unweighted_b", "grad_share_top_weighted_a", "grad_share_top_weighted_b",
            "grad_share_top_shift_a", "grad_share_top_shift_b", "abs_reallocated_kl_a", "abs_reallocated_kl_b",
            "abs_reallocated_grad_a", "abs_reallocated_grad_b"
        ]}

        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs} seed={seed} mode={mode}", leave=False)
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out_a = model_a(x)
            out_b = model_b(x)

            loss_a, stats_a = mutual_kd_loss(out_a, out_b, y, epoch, loss_cfg)
            loss_b, stats_b = mutual_kd_loss(out_b, out_a, y, epoch, loss_cfg)
            loss = loss_a + loss_b

            opt_a.zero_grad(set_to_none=True)
            opt_b.zero_grad(set_to_none=True)
            loss.backward()
            opt_a.step(); opt_b.step()
            _update_ema(ema_a, model_a, ema_decay)
            _update_ema(ema_b, model_b, ema_decay)

            n = y.numel()
            meters["loss_a"].update(loss_a.item(), n); meters["loss_b"].update(loss_b.item(), n)
            meters["ce_a"].update(stats_a["ce_loss"], n); meters["ce_b"].update(stats_b["ce_loss"], n)
            meters["kd_a"].update(stats_a["kd_loss"], n); meters["kd_b"].update(stats_b["kd_loss"], n)
            meters["unweighted_kd_a"].update(stats_a["unweighted_kd_loss"], n); meters["unweighted_kd_b"].update(stats_b["unweighted_kd_loss"], n)
            for k in ["weight_mean", "weight_std", "weight_min", "weight_max", "msp_mean", "msp_std", "residual_std", "grad_share_top_unweighted", "grad_share_top_weighted", "grad_share_top_shift", "abs_reallocated_kl", "abs_reallocated_grad"]:
                meters[f"{k}_a"].update(stats_a[k], n)
                meters[f"{k}_b"].update(stats_b[k], n)
            cumulative["a_abs_reallocated_kl"] += float(stats_a["abs_reallocated_kl"]) * n
            cumulative["b_abs_reallocated_kl"] += float(stats_b["abs_reallocated_kl"]) * n
            cumulative["a_abs_reallocated_grad"] += float(stats_a["abs_reallocated_grad"]) * n
            cumulative["b_abs_reallocated_grad"] += float(stats_b["abs_reallocated_grad"]) * n

            if log_interval > 0 and global_step % log_interval == 0:
                if max_trace_rows <= 0 or len(trace_rows) < max_trace_rows:
                    row = {
                        "epoch": epoch,
                        "batch_idx": batch_idx,
                        "global_step": global_step,
                        "lr": opt_a.param_groups[0]["lr"],
                    }
                    for prefix, st in [("a_learns_from_b", stats_a), ("b_learns_from_a", stats_b)]:
                        for key in ["msp_mean", "msp_std", "residual_std", "weight_mean", "weight_std", "weight_min", "weight_max", "unweighted_kd_loss", "kd_loss", "grad_proxy_mean", "grad_proxy_weighted_mean", "grad_share_top_unweighted", "grad_share_top_weighted", "grad_share_top_shift", "grad_share_bottom_unweighted", "grad_share_bottom_weighted", "grad_share_bottom_shift", "abs_reallocated_kl", "signed_reallocated_kl", "abs_reallocated_grad", "signed_reallocated_grad"]:
                            row[f"{prefix}_{key}"] = st.get(key, np.nan)
                    trace_rows.append(row)
            global_step += 1

            if batch_idx % 20 == 0:
                pbar.set_postfix({"loss": f"{loss.item():.3f}", "wstd": f"{stats_a['weight_std']:.4f}"})

        sch_a.step(); sch_b.step()
        acc_a, acc_b = evaluate(model_a, model_b, val_loader, device)
        best_a = max(best_a, acc_a); best_b = max(best_b, acc_b)
        elapsed = time.time() - start_time

        epoch_row = {
            "epoch": epoch,
            "seed": seed,
            "mode": mode,
            "dataset": dataset_name,
            "peer_a": model_a_name,
            "peer_b": model_b_name,
            "lr": opt_a.param_groups[0]["lr"],
            "val_acc_a": acc_a,
            "val_acc_b": acc_b,
            "best_acc_a": best_a,
            "best_acc_b": best_b,
            "elapsed_sec": elapsed,
            "cum_abs_reallocated_kl_a": cumulative["a_abs_reallocated_kl"],
            "cum_abs_reallocated_kl_b": cumulative["b_abs_reallocated_kl"],
            "cum_abs_reallocated_grad_a": cumulative["a_abs_reallocated_grad"],
            "cum_abs_reallocated_grad_b": cumulative["b_abs_reallocated_grad"],
        }
        for k, m in meters.items():
            epoch_row[k] = m.avg
        epoch_rows.append(epoch_row)
        pd.DataFrame(epoch_rows).to_csv(run_dir / "epoch_metrics.csv", index=False)
        if trace_rows:
            pd.DataFrame(trace_rows).to_csv(run_dir / "filter_gradient_trace.csv", index=False)

        # Save latest + best checkpoints. For space, users may disable by config.
        if bool(_cfg_get(cfg, "runtime.save_checkpoints", True)):
            latest = {
                "epoch": epoch,
                "model_a": model_a.state_dict(),
                "model_b": model_b.state_dict(),
                "ema_a": ema_a.state_dict(),
                "ema_b": ema_b.state_dict(),
                "opt_a": opt_a.state_dict(),
                "opt_b": opt_b.state_dict(),
                "acc_a": acc_a,
                "acc_b": acc_b,
                "best_a": best_a,
                "best_b": best_b,
            }
            torch.save(latest, run_dir / "latest.pt")
            if acc_a >= best_a or acc_b >= best_b:
                torch.save(latest, run_dir / "best.pt")

        print(f"[{dataset_name}][{mode}][seed {seed}] epoch {epoch:03d}: A={acc_a:.2f} B={acc_b:.2f} bestA={best_a:.2f} bestB={best_b:.2f}")

    summary = {
        "seed": seed,
        "mode": mode,
        "dataset": dataset_name,
        "peer_a": model_a_name,
        "peer_b": model_b_name,
        "best_acc_a": best_a,
        "best_acc_b": best_b,
        "final_acc_a": epoch_rows[-1]["val_acc_a"],
        "final_acc_b": epoch_rows[-1]["val_acc_b"],
        "epochs": epochs,
        "time_sec": time.time() - start_time,
    }
    save_json(summary, str(run_dir / "summary.json"))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Reviewer-1 BADD mutual-learning experiment runner")
    parser.add_argument("--config", type=str, required=True, help="YAML config")
    parser.add_argument("--mode", type=str, default=None, help="baseline/badd/random_zero_mean/shuffled_residual/sign_flipped/badd_zscore")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.setdefault("training", {})["epochs"] = args.epochs
    if args.data_root is not None:
        cfg.setdefault("dataset", {})["root"] = args.data_root
    if args.device is not None:
        cfg.setdefault("runtime", {})["device"] = args.device
    if args.workers is not None:
        cfg.setdefault("runtime", {})["workers"] = args.workers

    output_root = args.output_root or _cfg_get(cfg, "runtime.output_root", "./paper_experiments_reviewer1")
    mode = args.mode or _cfg_get(cfg, "loss.mode", "baseline")
    seed = args.seed if args.seed is not None else int(_cfg_get(cfg, "runtime.seed", 0))
    train_one_run(cfg, seed=seed, mode=mode, output_root=output_root)


if __name__ == "__main__":
    main()
