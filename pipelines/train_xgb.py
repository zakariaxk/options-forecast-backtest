from __future__ import annotations

import argparse
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score

from common.io import StorageURI, read_json, read_parquet, write_json
from common.schema import TrainConfig
from ml.utils.seed import set_seed


def _load_features(config: TrainConfig) -> tuple[pd.DataFrame, dict]:
    base = f"{config.processed_uri}/{config.symbol}/{config.feature_version}"
    features_uri = f"{base}/features.parquet"
    schema_uri = f"{base}/schema.json"
    data = read_parquet(features_uri)
    schema = read_json(schema_uri)
    return data, schema


def _train_valid_split(df: pd.DataFrame, valid_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("trade_date")
    split_idx = int((1 - valid_size) * len(df))
    train_df = df.iloc[:split_idx]
    valid_df = df.iloc[split_idx:]
    return train_df, valid_df


def train_model(config: TrainConfig) -> dict[str, str | float]:
    set_seed(config.seed)
    data, schema = _load_features(config)
    feature_cols = schema["scaled_columns"]
    target_col = "target_reg" if config.target == "regression" else "target_bin"
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=feature_cols + [target_col])
    train_df, valid_df = _train_valid_split(data)
    if train_df.empty or valid_df.empty:
        raise ValueError("Insufficient data after filtering; check feature_version and horizon.")
    dtrain = xgb.DMatrix(train_df[feature_cols], label=train_df[target_col])
    dvalid = xgb.DMatrix(valid_df[feature_cols], label=valid_df[target_col])
    params = {
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "lambda": 1.0,
        "seed": config.seed,
    }
    if config.target == "regression":
        params["objective"] = "reg:squarederror"
        eval_metric = "rmse"
    else:
        params["objective"] = "binary:logistic"
        eval_metric = "auc"
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=25,
        verbose_eval=False,
    )
    preds = booster.predict(dvalid)
    metrics = _compute_metrics(valid_df[target_col], preds, config.target)
    model_base = f"{config.output_uri}/{config.model_name}/{config.run_name}"
    model_storage = StorageURI(f"{model_base}/model.json")
    model_path = model_storage.local_path()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(model_path.as_posix())
    params_uri = f"{model_base}/params.json"
    metrics_uri = f"{model_base}/metrics.json"
    info = {
        "model_name": config.model_name,
        "run_name": config.run_name,
        "feature_version": config.feature_version,
        "target": config.target,
        "best_iteration": int(booster.best_iteration),
    }
    write_json({**params, **info}, params_uri)
    write_json(metrics, metrics_uri)
    return {
        "model_uri": f"{model_base}/model.json",
        "params_uri": params_uri,
        "metrics_uri": metrics_uri,
        **metrics,
    }


def _compute_metrics(y_true: pd.Series, y_pred: np.ndarray, target: str) -> dict[str, float]:
    if target == "regression":
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        return {"mse": float(mse), "rmse": float(np.sqrt(mse)), "mae": float(mae)}
    auc = roc_auc_score(y_true, y_pred)
    preds = (y_pred >= 0.5).astype(int)
    acc = accuracy_score(y_true, preds)
    return {"auc": float(auc), "accuracy": float(acc)}


def _parse_args(args: Optional[Iterable[str]] = None) -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train XGBoost model.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--feature-version", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model-name", default="xgb_reg")
    parser.add_argument("--target", choices=["regression", "classification"], default="regression")
    parser.add_argument("--processed-uri", default="data/processed")
    parser.add_argument("--output-uri", default="data/models")
    parser.add_argument("--seed", type=int, default=1337)
    parsed = parser.parse_args(args=args)
    return TrainConfig(
        symbol=parsed.symbol,
        feature_version=parsed.feature_version,
        run_name=parsed.run_name,
        model_name=parsed.model_name,
        target=parsed.target,
        processed_uri=parsed.processed_uri,
        output_uri=parsed.output_uri,
        seed=parsed.seed,
    )


def main(argv: Optional[Iterable[str]] = None) -> dict[str, str | float]:
    config = _parse_args(argv)
    return train_model(config)


if __name__ == "__main__":
    main()
