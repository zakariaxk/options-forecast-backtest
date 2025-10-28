from __future__ import annotations

import argparse
from typing import Iterable, Optional

import torch
from torch import nn

from common.io import StorageURI, read_json, read_parquet, write_json
from common.schema import TrainConfig
from ml.datasets import SequenceConfig, create_dataloader
from ml.models.lstm import OptionLSTM
from ml.utils.seed import set_seed
from ml.utils.train import TorchTrainConfig, train_model as torch_train_model


def _load_features(config: TrainConfig):
    base = f"{config.processed_uri}/{config.symbol}/{config.feature_version}"
    features = read_parquet(f"{base}/features.parquet")
    schema = read_json(f"{base}/schema.json")
    return features, schema


def _train_valid_split(df, valid_size: float = 0.2):
    df = df.sort_values("trade_date")
    split_idx = int(len(df) * (1 - valid_size))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def train_lstm(config: TrainConfig):
    set_seed(config.seed)
    data, schema = _load_features(config)
    feature_cols = schema["scaled_columns"]
    target_col = "target_reg" if config.target == "regression" else "target_bin"
    train_df, valid_df = _train_valid_split(data)
    seq_cfg = SequenceConfig(
        feature_columns=feature_cols,
        target_column=target_col,
        seq_len=config.seq_len,
        batch_size=config.batch_size,
    )
    train_loader = create_dataloader(train_df, seq_cfg, shuffle=True)
    valid_loader = create_dataloader(valid_df, seq_cfg, shuffle=False)
    if len(train_loader.dataset) == 0 or len(valid_loader.dataset) == 0:
        raise ValueError("Insufficient data to train LSTM - check sequence length and dataset size")
    model = OptionLSTM(len(feature_cols), task=config.target)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss() if config.target == "regression" else nn.BCELoss()
    train_cfg = TorchTrainConfig(
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history, summary = torch_train_model(
        model=model,
        train_loader=train_loader,
        val_loader=valid_loader,
        criterion=criterion,
        task=config.target,
        device=device,
        config=train_cfg,
    )
    model_base = f"{config.output_uri}/{config.model_name}/{config.run_name}"
    state_uri = StorageURI(f"{model_base}/state_dict.pt")
    state_path = state_uri.local_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), state_path)
    metrics_uri = f"{model_base}/metrics.json"
    history_uri = f"{model_base}/history.json"
    info_uri = f"{model_base}/info.json"
    write_json(summary, metrics_uri)
    write_json(history, history_uri)
    meta = {
        "model_name": config.model_name,
        "run_name": config.run_name,
        "feature_version": config.feature_version,
        "target": config.target,
        "seq_len": config.seq_len,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
    }
    write_json(meta, info_uri)
    return {
        "state_dict_uri": f"{model_base}/state_dict.pt",
        "metrics_uri": metrics_uri,
        "history_uri": history_uri,
        **summary,
    }


def _parse_args(args: Optional[Iterable[str]] = None) -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train LSTM model for options forecasting.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--feature-version", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--target", choices=["regression", "classification"], default="regression")
    parser.add_argument("--processed-uri", default="data/processed")
    parser.add_argument("--output-uri", default="data/models")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=1337)
    parsed = parser.parse_args(args=args)
    return TrainConfig(
        symbol=parsed.symbol,
        feature_version=parsed.feature_version,
        run_name=parsed.run_name,
        model_name="torch_lstm",
        target=parsed.target,
        processed_uri=parsed.processed_uri,
        output_uri=parsed.output_uri,
        seq_len=parsed.seq_len,
        batch_size=parsed.batch_size,
        epochs=parsed.epochs,
        learning_rate=parsed.learning_rate,
        weight_decay=parsed.weight_decay,
        seed=parsed.seed,
    )


def main(argv: Optional[Iterable[str]] = None):
    config = _parse_args(argv)
    return train_lstm(config)


if __name__ == "__main__":
    main()
