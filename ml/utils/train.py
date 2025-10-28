from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn

from ml.utils.eval import compute_metrics, detach_tensor


@dataclass
class TorchTrainConfig:
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 5
    grad_clip: float = 1.0


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    criterion,
    task: str,
    device: torch.device,
    config: TorchTrainConfig,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_state = None
    best_loss = float("inf")
    best_metrics: Dict[str, float] = {}
    patience = 0
    history: Dict[str, Dict[str, float]] = {}
    for epoch in range(config.epochs):
        model.train()
        for batch in train_loader:
            batch_x, batch_y = batch[0], batch[1]
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
        val_loss, metrics = evaluate(model, val_loader, criterion, task, device)
        history[f"epoch_{epoch}"] = {"loss": val_loss, **metrics}
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_metrics = metrics
            patience = 0
        else:
            patience += 1
            if patience >= config.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history, {"val_loss": best_loss, **best_metrics}


def evaluate(model: nn.Module, loader, criterion, task: str, device: torch.device) -> Tuple[float, Dict[str, float]]:
    model.eval()
    losses = []
    preds = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            batch_x, batch_y = batch[0], batch[1]
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            losses.append(loss.item())
            preds.append(detach_tensor(outputs))
            targets.append(detach_tensor(batch_y))
    if preds:
        y_pred = np.concatenate(preds)
        y_true = np.concatenate(targets)
        metrics = compute_metrics(y_true, y_pred, task)
    else:
        y_pred = np.array([])
        y_true = np.array([])
        metrics = {}
    return float(sum(losses) / max(len(losses), 1)), metrics
