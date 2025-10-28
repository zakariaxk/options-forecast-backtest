from __future__ import annotations

import os
import random

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional
    torch = None


def set_seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    os.environ["PYTHONHASHSEED"] = str(value)
    if torch is not None:
        torch.manual_seed(value)
        torch.cuda.manual_seed_all(value)
        torch.use_deterministic_algorithms(True, warn_only=True)
