from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


@dataclass
class LossConfig:
    mode: str = "baseline"
    tau: float = 3.0
    alpha: float = 0.495
    clip_min: float = 0.8
    clip_max: float = 1.2
    warmup_epochs: int = 10
    zscore_eps: float = 1e-6
    random_match_residual_std: bool = True
    top_fraction: float = 0.2


def _warmup_factor(epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return 1.0
    return float(min(1.0, max(0.0, epoch / float(warmup_epochs))))


def _renorm_mean_one(w: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return w / (w.mean().clamp_min(eps))


def compute_weight_and_stats(
    logits_student: torch.Tensor,
    logits_teacher: torch.Tensor,
    target: torch.Tensor,
    epoch: int,
    cfg: LossConfig,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
    """Return detached sample weights and batch-level stats.

    Modes targeting Reviewer 1:
      baseline:              w = 1
      badd:                  w = 1 + alpha * (MSP - batch_mean)
      badd_zscore:           w = 1 + alpha * zscore(MSP)
      random_zero_mean:      random centered perturbation with residual-matched std
      shuffled_residual:     batch residual distribution preserved but sample mapping broken
      sign_flipped:          w = 1 - alpha * (MSP - batch_mean)
      absolute_msp:          w = 1 + alpha * (MSP - batch_mean detached? no: absolute control)
    """
    mode = cfg.mode.lower()
    device = logits_student.device
    batch_size = logits_student.shape[0]
    with torch.no_grad():
        probs_teacher = F.softmax(logits_teacher / cfg.tau, dim=1)
        conf, _ = probs_teacher.max(dim=1)
        batch_mean = conf.mean()
        residual = conf - batch_mean
        residual_std = residual.std(unbiased=False)

        if mode == "baseline":
            raw_w = torch.ones_like(conf)
        elif mode in {"badd", "badd_v17_11", "mean_centered"}:
            raw_w = 1.0 + cfg.alpha * residual
        elif mode in {"badd_zscore", "zscore"}:
            z = residual / (residual_std + cfg.zscore_eps)
            raw_w = 1.0 + cfg.alpha * z
        elif mode in {"random_zero_mean", "random"}:
            r = torch.randn_like(conf)
            r = r - r.mean()
            if cfg.random_match_residual_std:
                r = r / (r.std(unbiased=False) + cfg.zscore_eps) * residual_std
            raw_w = 1.0 + cfg.alpha * r
        elif mode in {"shuffled_residual", "shuffle"}:
            perm = torch.randperm(batch_size, device=device)
            raw_w = 1.0 + cfg.alpha * residual[perm]
        elif mode in {"sign_flipped", "anti_badd"}:
            raw_w = 1.0 - cfg.alpha * residual
        elif mode in {"absolute_msp", "abs_msp"}:
            # Deliberately not mean preserving; used only as an ablation/control.
            raw_w = 1.0 + cfg.alpha * conf
        else:
            raise ValueError(f"Unknown loss mode: {cfg.mode}")

        warm = _warmup_factor(epoch, cfg.warmup_epochs)
        raw_w = 1.0 + (raw_w - 1.0) * warm
        clipped_w = torch.clamp(raw_w, min=cfg.clip_min, max=cfg.clip_max)

        # Keep the main BADD-style variants mean-preserving even after clipping.
        if mode in {"badd", "badd_v17_11", "mean_centered", "badd_zscore", "zscore", "random_zero_mean", "random", "shuffled_residual", "shuffle", "sign_flipped", "anti_badd"}:
            final_w = _renorm_mean_one(clipped_w)
        else:
            final_w = clipped_w

        # Gradient proxy wrt student logits for tau^2 KL: tau * (p_s - p_t).
        probs_student = F.softmax(logits_student / cfg.tau, dim=1)
        grad_proxy = (cfg.tau * (probs_student - probs_teacher)).norm(dim=1)

        k = max(1, int(cfg.top_fraction * batch_size))
        top_idx = torch.topk(residual, k=k, largest=True).indices
        bottom_idx = torch.topk(residual, k=k, largest=False).indices
        g_sum = grad_proxy.sum().clamp_min(1e-12)
        wg = final_w * grad_proxy
        wg_sum = wg.sum().clamp_min(1e-12)
        share_top_unweighted = grad_proxy[top_idx].sum() / g_sum
        share_top_weighted = wg[top_idx].sum() / wg_sum
        share_bottom_unweighted = grad_proxy[bottom_idx].sum() / g_sum
        share_bottom_weighted = wg[bottom_idx].sum() / wg_sum

        stats = {
            "weight_mean": final_w.mean().item(),
            "weight_std": final_w.std(unbiased=False).item(),
            "weight_min": final_w.min().item(),
            "weight_max": final_w.max().item(),
            "msp_mean": conf.mean().item(),
            "msp_std": conf.std(unbiased=False).item(),
            "residual_mean": residual.mean().item(),
            "residual_std": residual_std.item(),
            "residual_abs_mean": residual.abs().mean().item(),
            "grad_proxy_mean": grad_proxy.mean().item(),
            "grad_proxy_weighted_mean": wg.mean().item(),
            "grad_share_top_unweighted": share_top_unweighted.item(),
            "grad_share_top_weighted": share_top_weighted.item(),
            "grad_share_top_shift": (share_top_weighted - share_top_unweighted).item(),
            "grad_share_bottom_unweighted": share_bottom_unweighted.item(),
            "grad_share_bottom_weighted": share_bottom_weighted.item(),
            "grad_share_bottom_shift": (share_bottom_weighted - share_bottom_unweighted).item(),
        }
        tensors = {
            "weight": final_w.detach(),
            "conf": conf.detach(),
            "residual": residual.detach(),
            "grad_proxy": grad_proxy.detach(),
            "top_idx": top_idx.detach(),
            "bottom_idx": bottom_idx.detach(),
        }
        return final_w.detach(), stats, tensors


def mutual_kd_loss(
    logits_student: torch.Tensor,
    logits_teacher: torch.Tensor,
    target: torch.Tensor,
    epoch: int,
    cfg: LossConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    ce = F.cross_entropy(logits_student, target)
    log_p_s = F.log_softmax(logits_student / cfg.tau, dim=1)
    p_t = F.softmax(logits_teacher.detach() / cfg.tau, dim=1)
    kl_per_sample = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=1) * (cfg.tau ** 2)
    weights, stats, tensors = compute_weight_and_stats(logits_student.detach(), logits_teacher.detach(), target, epoch, cfg)
    kd = (weights * kl_per_sample).mean()

    # Reviewer-1 cumulative-effect quantities.
    with torch.no_grad():
        unweighted_kd = kl_per_sample.mean()
        abs_reallocated_kl = ((weights - 1.0).abs() * kl_per_sample).mean()
        signed_reallocated_kl = ((weights - 1.0) * kl_per_sample).mean()
        grad_proxy = tensors["grad_proxy"]
        abs_reallocated_grad = ((weights - 1.0).abs() * grad_proxy).mean()
        signed_reallocated_grad = ((weights - 1.0) * grad_proxy).mean()
        stats.update({
            "ce_loss": ce.detach().item(),
            "kd_loss": kd.detach().item(),
            "unweighted_kd_loss": unweighted_kd.detach().item(),
            "abs_reallocated_kl": abs_reallocated_kl.detach().item(),
            "signed_reallocated_kl": signed_reallocated_kl.detach().item(),
            "abs_reallocated_grad": abs_reallocated_grad.detach().item(),
            "signed_reallocated_grad": signed_reallocated_grad.detach().item(),
        })
    return ce + kd, stats
