import torch
import torch.nn.functional as F


def _get_gt_mask(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t: torch.Tensor, mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(dim=1, keepdims=True)
    return torch.cat([t1, t2], dim=1)


def dkd_loss(
    logits_student: torch.Tensor,
    logits_teacher: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 8.0,
    temperature: float = 4.0,
) -> torch.Tensor:
    # Online DKD (Decoupled Knowledge Distillation)
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)

    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)

    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)

    log_pred_student = torch.log(pred_student + 1e-12)

    # TCKD
    tckd = (
        F.kl_div(log_pred_student, pred_teacher, reduction="sum")
        * (temperature ** 2)
        / target.shape[0]
    )

    pred_teacher_part2 = F.softmax(logits_teacher / temperature - 1000.0 * gt_mask, dim=1)
    log_pred_student_part2 = F.log_softmax(logits_student / temperature - 1000.0 * gt_mask, dim=1)

    # NCKD
    nckd = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction="sum")
        * (temperature ** 2)
        / target.shape[0]
    )

    return alpha * tckd + beta * nckd