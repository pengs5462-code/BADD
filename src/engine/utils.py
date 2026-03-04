from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, Any

import torch

from ..config import TrainConfig


def get_device(cfg: TrainConfig) -> torch.device:
    """
    Return torch.device based on cfg.device.
    """
    return torch.device(cfg.device)


def make_run_name(cfg: TrainConfig) -> str:
    """
    Create a readable run identifier for logs/outputs.
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{cfg.arch}_{cfg.mode}_ep{cfg.epochs}_bs{cfg.batch_size}_lr{cfg.lr}_{ts}"


def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """
    Flatten nested dict for CSV logging convenience.
    """
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix == "" else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        else:
            out[key] = v
    return out
