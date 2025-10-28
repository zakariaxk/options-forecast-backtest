from __future__ import annotations

import torch
from torch import nn

from ml.models.heads import ClassificationHead, RegressionHead


class OptionLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        task: str = "regression",
    ) -> None:
        super().__init__()
        self.task = task
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        if task == "regression":
            self.head = RegressionHead(hidden_dim)
        else:
            self.head = ClassificationHead(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        activated = self.dropout(last_hidden)
        return self.head(activated)
