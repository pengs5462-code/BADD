from dataclasses import dataclass
import os
import torch

@dataclass
class TrainConfig:
    batch_size: int = 128
    lr: float = 0.1
    epochs: int = 300
    data_root: str = "./data"
    save_dir: str = "./experiments"
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)