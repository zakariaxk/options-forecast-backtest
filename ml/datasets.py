from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class SequenceConfig:
    feature_columns: List[str]
    target_column: str
    seq_len: int = 32
    batch_size: int = 64


class OptionSequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, config: SequenceConfig):
        self.config = config
        self.samples: List[tuple[np.ndarray, float, int]] = []
        self._build_samples(df)

    def _build_samples(self, df: pd.DataFrame) -> None:
        for _, group in df.groupby("option_key"):
            group = group.sort_values("trade_date")
            features = group[self.config.feature_columns].to_numpy(dtype=np.float32)
            targets = group[self.config.target_column].to_numpy(dtype=np.float32)
            if features.shape[0] < self.config.seq_len:
                continue
            for idx in range(self.config.seq_len - 1, features.shape[0]):
                window = features[idx - self.config.seq_len + 1 : idx + 1]
                target = targets[idx]
                if np.isfinite(target):
                    row_index = int(group.index[idx])
                    self.samples.append((window, target, row_index))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        window, target, row_index = self.samples[index]
        return (
            torch.from_numpy(window),
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(row_index, dtype=torch.long),
        )


def create_dataloader(df: pd.DataFrame, config: SequenceConfig, shuffle: bool = True) -> DataLoader:
    dataset = OptionSequenceDataset(df, config)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, drop_last=False)
