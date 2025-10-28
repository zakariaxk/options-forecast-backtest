from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score


@dataclass
class EvalResult:
    loss: float
    metrics: Dict[str, float]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task: str) -> Dict[str, float]:
    if task == "regression":
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        return {"mse": float(mse), "rmse": float(np.sqrt(mse)), "mae": float(mae)}
    probs = y_pred.clip(1e-6, 1 - 1e-6)
    auc = roc_auc_score(y_true, probs)
    preds = (probs >= 0.5).astype(int)
    acc = (preds == y_true).mean()
    return {"auc": float(auc), "accuracy": float(acc)}


def detach_tensor(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()
