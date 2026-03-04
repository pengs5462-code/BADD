import numpy as np
import torch
import torch.nn.functional as F


def compute_weight(
    mode: str,
    logits_student: torch.Tensor,
    logits_teacher: torch.Tensor,
    target: torch.Tensor,
    epoch: int,
    epochs_total: int,
    device: torch.device,
):
    """
    Weight-based strategies only.
    Returns:
      w: Tensor [B]
      stats: dict (mean/min/max + optional extra fields)
    """
    B = logits_student.size(0)
    stats = {"mean": 1.0, "min": 1.0, "max": 1.0}

    # -------------------------
    # baseline (no reweight)
    # -------------------------
    if mode == "baseline":
        w = torch.ones(B, device=device)

    # -------------------------
    # V5: Booster
    # w = (1 + 2*P_t(y)) * rampup
    # -------------------------
    elif mode == "dynamic_v5":
        with torch.no_grad():
            probs_teacher = F.softmax(logits_teacher, dim=1)
            pt_target = probs_teacher.gather(1, target.unsqueeze(1)).squeeze()
            dynamic_weight = 1.0 + 2.0 * pt_target
            rampup = float(np.clip(epoch / 50.0, 0.1, 1.0))
            w = dynamic_weight * rampup

    # -------------------------
    # V8: Switching
    # correct -> 2.0 ; wrong -> 0.5 ; rampup
    # -------------------------
    elif mode == "dynamic_v8":
        with torch.no_grad():
            pred_teacher = logits_teacher.argmax(dim=1)
            is_correct = (pred_teacher == target)
            dynamic_weight = torch.full((B,), 0.5, device=device)
            dynamic_weight[is_correct] = 2.0
            rampup = float(np.clip(epoch / 20.0, 0.0, 1.0))
            w = dynamic_weight * rampup

    # -------------------------
    # V9: Soft residual by P_t(y) in [0.8,1.2], fast rampup
    # w = (0.8 + 0.4*P_t(y)) * rampup
    # -------------------------
    elif mode == "dynamic_v9":
        with torch.no_grad():
            probs_teacher = F.softmax(logits_teacher, dim=1)
            pt_target = probs_teacher.gather(1, target.unsqueeze(1)).squeeze()
            dynamic_weight = 0.8 + 0.4 * pt_target
            rampup = float(np.clip(epoch / 5.0, 0.0, 1.0))
            w = dynamic_weight * rampup

    # -------------------------
    # V10: gap-based + time decay
    # w = clamp(1 + (P_t(y)-P_s(y)), 0.1, 2.0) * decay_factor
    # -------------------------
    elif mode == "dynamic_v10":
        with torch.no_grad():
            probs_teacher = F.softmax(logits_teacher, dim=1)
            probs_student = F.softmax(logits_student, dim=1)
            pt_target = probs_teacher.gather(1, target.unsqueeze(1)).squeeze()
            ps_target = probs_student.gather(1, target.unsqueeze(1)).squeeze()
            gap = pt_target - ps_target
            dynamic_weight = torch.clamp(1.0 + gap, min=0.1, max=2.0)

            # cosine decay from 1.0 to 0.2
            decay_factor = 0.2 + 0.8 * 0.5 * (1.0 + np.cos(np.pi * epoch / float(epochs_total)))
            w = dynamic_weight * float(decay_factor)

    # -------------------------
    # V11: Adaptive thresholding (±0.15 around 1.0)
    # threshold increases 0.4 -> 0.85
    # -------------------------
    elif mode == "dynamic_v11":
        with torch.no_grad():
            probs_teacher = F.softmax(logits_teacher, dim=1)
            conf_teacher, pred_teacher = probs_teacher.max(dim=1)
            is_correct = (pred_teacher == target)

            current_threshold = 0.4 + 0.45 * (epoch / float(epochs_total))
            mask_active = (conf_teacher > current_threshold)

            dynamic_weight = torch.ones_like(conf_teacher)
            dynamic_weight[mask_active & is_correct] = 1.15
            dynamic_weight[mask_active & (~is_correct)] = 0.85

            rampup = float(np.clip(epoch / 5.0, 0.0, 1.0))
            w = 1.0 + (dynamic_weight - 1.0) * rampup

    # -------------------------
    # V13: only reward correct samples, curriculum factor
    # w = 1 + 0.5*conf* (epoch/E)^2  on correct
    # -------------------------
    elif mode == "dynamic_v13":
        with torch.no_grad():
            probs_teacher = F.softmax(logits_teacher, dim=1)
            conf_teacher, pred_teacher = probs_teacher.max(dim=1)
            is_correct = (pred_teacher == target)

            dynamic_weight = torch.ones_like(conf_teacher)
            time_factor = (epoch / float(epochs_total)) ** 2

            boost_mask = is_correct
            dynamic_weight[boost_mask] = dynamic_weight[boost_mask] + 0.5 * conf_teacher[boost_mask] * float(time_factor)
            w = dynamic_weight

    # -------------------------
    # V14: aggressive reward/punish by confidence, warmup 20
    # correct: 1+0.6*conf, wrong: 1-0.6*conf, then *rampup
    # -------------------------
    elif mode == "dynamic_v14":
        with torch.no_grad():
            probs_teacher = F.softmax(logits_teacher, dim=1)
            conf_teacher, pred_teacher = probs_teacher.max(dim=1)
            is_correct = (pred_teacher == target)

            amplitude = 0.6
            dynamic_weight = torch.ones_like(conf_teacher)
            dynamic_weight[is_correct] += amplitude * conf_teacher[is_correct]
            dynamic_weight[~is_correct] -= amplitude * conf_teacher[~is_correct]

            rampup = float(np.clip(epoch / 20.0, 0.2, 1.0))
            w = dynamic_weight * rampup

    # -------------------------
    # V15: early self-learning then elite gating reward
    # epoch<=30: scalar warmup; else reward correct&conf>0.75
    # -------------------------
    elif mode == "dynamic_v15":
        with torch.no_grad():
            if epoch <= 30:
                initial_warmup = 0.1 + 0.9 * (epoch / 30.0)
                w = torch.full((B,), float(initial_warmup), device=device)
            else:
                probs_teacher = F.softmax(logits_teacher, dim=1)
                conf_teacher, pred_teacher = probs_teacher.max(dim=1)
                is_correct = (pred_teacher == target)

                dynamic_weight = torch.ones_like(conf_teacher)
                high_conf_threshold = 0.75
                mask_elite = is_correct & (conf_teacher > high_conf_threshold)
                dynamic_weight[mask_elite] += 0.3 * conf_teacher[mask_elite]
                w = dynamic_weight

    # -------------------------
    # V17: batch-mean centered confidence, clamp [0.6,1.4], rampup 10
    # -------------------------
    elif mode == "dynamic_v17":
        with torch.no_grad():
            probs_teacher = F.softmax(logits_teacher, dim=1)
            conf_teacher_max, _ = probs_teacher.max(dim=1)
            batch_mean = conf_teacher_max.mean()
            alpha = 0.5
            dynamic_weight = 1.0 + alpha * (conf_teacher_max - batch_mean)
            dynamic_weight = torch.clamp(dynamic_weight, min=0.6, max=1.4)
            rampup = float(np.clip(epoch / 10.0, 0.0, 1.0))
            w = 1.0 + (dynamic_weight - 1.0) * rampup

    # -------------------------
    # V17_11: batch-mean centered confidence, clamp [0.8,1.2], alpha=0.495, rampup 10
    # -------------------------
    elif mode == "dynamic_v17_11":
        with torch.no_grad():
            probs_teacher = F.softmax(logits_teacher, dim=1)
            conf_teacher, _ = probs_teacher.max(dim=1)
            batch_mean = conf_teacher.mean()
            alpha = 0.50
            dynamic_weight = 1.0 + alpha * (conf_teacher - batch_mean)
            dynamic_weight = torch.clamp(dynamic_weight, min=0.8, max=1.2)
            rampup = float(np.clip(epoch / 10.0, 0.0, 1.0))
            w = 1.0 + (dynamic_weight - 1.0) * rampup

    # -------------------------
    # V18: adaptive alpha by batch std + mean, clamp [0.8,1.2], rampup 10
    # -------------------------
    elif mode == "dynamic_v18":
        with torch.no_grad():
            probs_teacher = F.softmax(logits_teacher, dim=1)
            conf_teacher, _ = probs_teacher.max(dim=1)

            batch_mean = conf_teacher.mean()
            batch_std = conf_teacher.std(unbiased=False)

            sigma0 = 0.1
            alpha_var = batch_std / sigma0

            mu_clamped = torch.clamp(batch_mean, 0.3, 0.9)
            alpha_mu = 0.6 + (1.0 - 0.6) * (mu_clamped - 0.3) / (0.9 - 0.3)

            alpha_base = 0.48
            raw_alpha = alpha_base * alpha_var * alpha_mu
            alpha = torch.clamp(raw_alpha, min=0.3, max=0.9)

            dynamic_weight = 1.0 + alpha * (conf_teacher - batch_mean)
            dynamic_weight = torch.clamp(dynamic_weight, min=0.8, max=1.2)

            rampup = float(np.clip(epoch / 10.0, 0.0, 1.0))
            w = 1.0 + (dynamic_weight - 1.0) * rampup
            stats["alpha"] = float(alpha.item())

    # -------------------------
    # ADM: asymmetric gating by (P_t(y)-P_s(y))
    # w = clamp(1 + tanh(lambda*(diff)), 0.1, 2.0)
    # -------------------------
    elif mode == "adm":
        with torch.no_grad():
            probs_student = F.softmax(logits_student, dim=1)
            probs_teacher = F.softmax(logits_teacher, dim=1)

            score_s = probs_student.gather(1, target.unsqueeze(1)).squeeze()
            score_t = probs_teacher.gather(1, target.unsqueeze(1)).squeeze()

            diff = score_t - score_s
            scale_factor = 3.0
            adm_weight = 1.0 + torch.tanh(scale_factor * diff)
            w = torch.clamp(adm_weight, min=0.1, max=2.0)

    else:
        raise ValueError(f"Unknown weight strategy mode: {mode}")

    stats["mean"] = float(w.mean().item())
    stats["min"] = float(w.min().item())
    stats["max"] = float(w.max().item())
    return w, stats