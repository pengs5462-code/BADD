import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Ensure project root is importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baddlab.datasets import build_loaders
from baddlab.models import build_peer_models


def _cfg_get(cfg: Dict, key: str, default=None):
    cur = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _load_models_from_ckpt(ckpt_path: Path, cfg: Dict, num_classes: int, device: torch.device):
    model_a_name = _cfg_get(cfg, "models.peer_a", "shufflenetv2_cifar")
    model_b_name = _cfg_get(cfg, "models.peer_b", "resnet32_cifar")
    pretrained = bool(_cfg_get(cfg, "models.pretrained", False))
    model_a, model_b = build_peer_models(model_a_name, model_b_name, num_classes, pretrained=pretrained)
    try:
        obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except TypeError:
        # Backward compatibility for older torch versions without weights_only.
        obj = torch.load(str(ckpt_path), map_location="cpu")
    model_a.load_state_dict(obj["model_a"])
    model_b.load_state_dict(obj["model_b"])
    model_a = model_a.to(device).eval()
    model_b = model_b.to(device).eval()
    ema_a = None
    ema_b = None
    if "ema_a" in obj and "ema_b" in obj:
        ema_a, ema_b = build_peer_models(model_a_name, model_b_name, num_classes, pretrained=pretrained)
        ema_a.load_state_dict(obj["ema_a"])
        ema_b.load_state_dict(obj["ema_b"])
        ema_a = ema_a.to(device).eval()
        ema_b = ema_b.to(device).eval()
    return model_a, model_b, ema_a, ema_b


def _collect_direction(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    ref_teacher: torch.nn.Module,
    loader,
    tau: float,
    device: torch.device,
    true_label_prob_space: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    residuals, top1, teacher_ce, true_label_conf = [], [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            t_logits = teacher(x)
            _ = student(x)
            ref_logits = ref_teacher(x)

            p_t = F.softmax(t_logits / tau, dim=1)
            p_t_raw = F.softmax(t_logits, dim=1)
            p_r = F.softmax(ref_logits / tau, dim=1)
            conf = p_t.max(dim=1).values
            residual = conf - conf.mean()

            top1_correct = (t_logits.argmax(dim=1) == y).float()
            ce = F.cross_entropy(t_logits, y, reduction="none")
            if true_label_prob_space == "temp":
                p_true = p_t.gather(1, y.view(-1, 1)).squeeze(1)
            else:
                p_true = p_t_raw.gather(1, y.view(-1, 1)).squeeze(1)

            residuals.append(residual.cpu().numpy())
            top1.append(top1_correct.cpu().numpy())
            teacher_ce.append(ce.cpu().numpy())
            true_label_conf.append(p_true.cpu().numpy())
    return (
        np.concatenate(residuals),
        np.concatenate(top1),
        np.concatenate(teacher_ce),
        np.concatenate(true_label_conf),
    )


def _fmt_group(mask, top1, ce, ptrue):
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "top1": float("nan"), "ce": float("nan"), "ptrue": float("nan")}
    return {
        "n": n,
        "top1": float(top1[mask].mean() * 100.0),
        "ce": float(ce[mask].mean()),
        "ptrue": float(ptrue[mask].mean()),
    }


def main():
    ap = argparse.ArgumentParser(description="MSP residual reliability diagnostic from saved checkpoints.")
    ap.add_argument("--run-config", type=str, required=True, help="Path to run_config.json")
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt), usually latest.pt")
    ap.add_argument("--ema-checkpoint", type=str, default=None, help="Optional checkpoint for EMA-like diagnostic teacher (e.g., best.pt)")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--true-label-prob-space", type=str, default="raw", choices=["raw", "temp"])
    args = ap.parse_args()

    with open(args.run_config, "r", encoding="utf-8") as f:
        run_cfg = json.load(f)
    cfg = run_cfg["config"]

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    dataset_name = _cfg_get(cfg, "dataset.name")
    data_root = _cfg_get(cfg, "dataset.root", "./data")
    image_size = int(_cfg_get(cfg, "dataset.image_size", 32))
    batch_size = int(args.batch_size if args.batch_size is not None else _cfg_get(cfg, "training.batch_size", 128))
    workers = int(args.workers if args.workers is not None else _cfg_get(cfg, "runtime.workers", 4))
    download = False
    tau = float(_cfg_get(cfg, "loss.tau", 3.0))

    _, val_loader, num_classes = build_loaders(dataset_name, data_root, image_size, batch_size, workers, download=download)
    model_a, model_b, ema_a, ema_b = _load_models_from_ckpt(Path(args.checkpoint), cfg, num_classes, device)

    if ema_a is None or ema_b is None:
        if args.ema_checkpoint:
            _, _, ema_a, ema_b = _load_models_from_ckpt(Path(args.ema_checkpoint), cfg, num_classes, device)
            if ema_a is None or ema_b is None:
                fallback_a, fallback_b, _, _ = _load_models_from_ckpt(Path(args.ema_checkpoint), cfg, num_classes, device)
                ema_a, ema_b = fallback_a, fallback_b
        else:
            ema_a, ema_b = model_a, model_b

    # A learns from B, and B learns from A; merge both directions for a single diagnostic table.
    r1, t1, c1, p1 = _collect_direction(
        model_a, model_b, ema_b, val_loader, tau, device, args.true_label_prob_space
    )
    r2, t2, c2, p2 = _collect_direction(
        model_b, model_a, ema_a, val_loader, tau, device, args.true_label_prob_space
    )
    residual = np.concatenate([r1, r2])
    top1 = np.concatenate([t1, t2])
    ce = np.concatenate([c1, c2])
    ptrue = np.concatenate([p1, p2])

    q20 = np.quantile(residual, 0.2)
    q40 = np.quantile(residual, 0.4)
    q60 = np.quantile(residual, 0.6)
    q80 = np.quantile(residual, 0.8)

    low_mask = residual <= q20
    mid_mask = (residual >= q40) & (residual <= q60)
    high_mask = residual >= q80

    low = _fmt_group(low_mask, top1, ce, ptrue)
    mid = _fmt_group(mid_mask, top1, ce, ptrue)
    high = _fmt_group(high_mask, top1, ce, ptrue)

    print(f"dataset={dataset_name} samples={len(residual)} tau={tau}")
    print("group,n,teacher_top1_correct_pct,teacher_ce,teacher_true_label_conf")
    print(f"Lowest20,{low['n']},{low['top1']:.4f},{low['ce']:.6f},{low['ptrue']:.6f}")
    print(f"Middle20,{mid['n']},{mid['top1']:.4f},{mid['ce']:.6f},{mid['ptrue']:.6f}")
    print(f"Highest20,{high['n']},{high['top1']:.4f},{high['ce']:.6f},{high['ptrue']:.6f}")


if __name__ == "__main__":
    main()
