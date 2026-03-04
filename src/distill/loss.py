import torch
import torch.nn.functional as F

from .strategies import compute_weight
from .dkd import dkd_loss


_WEIGHT_MODES = {
    "baseline",
    "dynamic_v5", "dynamic_v8", "dynamic_v9", "dynamic_v10", "dynamic_v11",
    "dynamic_v13", "dynamic_v14", "dynamic_v15",
    "dynamic_v17", "dynamic_v17_11", "dynamic_v18",
    "adm",
}

_OTHER_MODES = {"kdcl", "okddip", "odkd"}  # odkd uses DKD


def distill_loss(
    logits_student: torch.Tensor,
    logits_teacher: torch.Tensor,
    target: torch.Tensor,
    epoch: int,
    mode: str,
    T: float = 3.0,
    epochs_total: int = 300,
    device: torch.device | None = None,
):
    """
    Unified loss entry.
    Returns:
      total_loss, stats
    """
    if device is None:
        device = logits_student.device

    # CE always present for all modes (consistent with your script style)
    ce = F.cross_entropy(logits_student, target)

    # logging KL (for comparison)
    with torch.no_grad():
        obs_kl = F.kl_div(
            F.log_softmax(logits_student, dim=1),
            F.softmax(logits_teacher, dim=1),
            reduction="batchmean",
        ).item()

    stats = {"kl": float(obs_kl)}

    # -------------------------
    # Weight-based DML (CE + weighted sample-wise KL)
    # -------------------------
    if mode in _WEIGHT_MODES:
        log_p_s = F.log_softmax(logits_student / T, dim=1)
        p_t = F.softmax(logits_teacher / T, dim=1)

        kl_per_sample = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=1) * (T ** 2)

        w, wstats = compute_weight(mode, logits_student, logits_teacher, target, epoch, epochs_total, device)
        stats.update(wstats)

        dml = (w.detach() * kl_per_sample).mean()
        return ce + dml, stats

    # -------------------------
    # KDCL: ensemble logits = (S + T)/2; KL(S || ensemble)
    # -------------------------
    if mode == "kdcl":
        ensemble_logits = (logits_student + logits_teacher) / 2.0
        p_ensemble = F.softmax(ensemble_logits / T, dim=1)
        log_p_s = F.log_softmax(logits_student / T, dim=1)
        kd = F.kl_div(log_p_s, p_ensemble.detach(), reduction="batchmean") * (T ** 2)
        stats.update({"mean": 1.0, "min": 1.0, "max": 1.0})
        return ce + kd, stats

    # -------------------------
    # OKDDip: entropy-weighted ensemble; KL(S || ensemble)
    # -------------------------
    if mode == "okddip":
        with torch.no_grad():
            p_s_raw = F.softmax(logits_student, dim=1)
            p_t_raw = F.softmax(logits_teacher, dim=1)

            entropy_s = -torch.sum(p_s_raw * torch.log(p_s_raw + 1e-6), dim=1)
            entropy_t = -torch.sum(p_t_raw * torch.log(p_t_raw + 1e-6), dim=1)

            w_s = torch.exp(-entropy_s)
            w_t = torch.exp(-entropy_t)
            sum_w = w_s + w_t + 1e-6
            norm_w_s = w_s / sum_w
            norm_w_t = w_t / sum_w

        ensemble_logits = norm_w_s.unsqueeze(1) * logits_student + norm_w_t.unsqueeze(1) * logits_teacher
        p_ensemble = F.softmax(ensemble_logits / T, dim=1)
        log_p_s = F.log_softmax(logits_student / T, dim=1)

        kd = F.kl_div(log_p_s, p_ensemble.detach(), reduction="batchmean") * (T ** 2)
        stats.update({
            "mean": float((norm_w_t.mean().item())),  # 一个可解释的统计：teacher平均占比
            "min": float((norm_w_t.min().item())),
            "max": float((norm_w_t.max().item())),
        })
        return ce + kd, stats

    # -------------------------
    # ODKD: use DKD loss as distillation term
    # (your script uses beta=2.0; temperature=T)
    # -------------------------
    if mode == "odkd":
        kd = dkd_loss(
            logits_student=logits_student,
            logits_teacher=logits_teacher,
            target=target,
            alpha=1.0,
            beta=2.0,
            temperature=T,
        )
        stats.update({"mean": 1.0, "min": 1.0, "max": 1.0})
        return ce + kd, stats

    raise ValueError(f"Unknown mode: {mode}. Known weight modes: {sorted(_WEIGHT_MODES)}; other modes: {sorted(_OTHER_MODES)}")