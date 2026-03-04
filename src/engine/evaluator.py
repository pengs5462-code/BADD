from __future__ import annotations

import torch


@torch.no_grad()
def top1_accuracy(model: torch.nn.Module, loader, device: torch.device) -> float:
    """
    Compute Top-1 accuracy over a dataloader.
    """
    model.eval()
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        out = model(inputs)
        pred = out.argmax(dim=1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    return 100.0 * correct / total
