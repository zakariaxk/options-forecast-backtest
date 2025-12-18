from __future__ import annotations

import argparse
from datetime import date
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from common.io import StorageURI, read_json, read_parquet, write_json, write_parquet
from common.schema import PredictConfig
from ml.models.lstm import OptionLSTM


def _load_features(config: PredictConfig) -> tuple[pd.DataFrame, dict]:
    base = f"{config.processed_uri}/{config.symbol}/{config.feature_version}"
    features = read_parquet(f"{base}/features.parquet")
    schema = read_json(f"{base}/schema.json")
    if config.as_of_date is not None:
        as_of = pd.to_datetime(config.as_of_date)
        features["trade_date"] = pd.to_datetime(features["trade_date"])
        features = features[features["trade_date"] == as_of].copy()
        if features.empty:
            raise ValueError(f"No features found for as_of_date={config.as_of_date}")
    return features, schema


def _predict_xgb(config: PredictConfig, features: pd.DataFrame, schema: dict) -> pd.DataFrame:
    feature_cols = schema["scaled_columns"]
    model_base = f"{config.models_uri}/{config.model_name}/{config.model_run_id}"
    model_path = StorageURI(f"{model_base}/model.json").local_path()
    params_path = f"{model_base}/params.json"
    params = read_json(params_path)
    booster = xgb.Booster()
    booster.load_model(model_path.as_posix())
    dmatrix = xgb.DMatrix(features[feature_cols])
    preds = booster.predict(dmatrix)
    target = params.get("target", "regression")
    base_cols = [
        "trade_date",
        "symbol",
        "option_key",
        "expiry",
        "type",
        "strike",
        "mid",
        "close",
        "oi",
        "volume",
        "expiry_dte",
        "moneyness",
        "spread_pct",
    ]
    output = features[base_cols].copy()
    if target == "classification":
        prob = preds
        output["score"] = prob
        output["yhat"] = (prob >= 0.5).astype(int)
        output["p_up"] = prob
    else:
        output["score"] = preds
        output["yhat"] = preds
        output["p_up"] = np.nan
    return output


def _prepare_lstm_inputs(df: pd.DataFrame, feature_cols: list[str], seq_len: int) -> tuple[torch.Tensor, list[int]]:
    samples = []
    indices = []
    for _, group in df.groupby("option_key"):
        group = group.sort_values("trade_date")
        if len(group) < seq_len:
            continue
        window = group.iloc[-seq_len:]
        samples.append(window[feature_cols].to_numpy(dtype=np.float32))
        indices.append(int(window.index[-1]))
    if not samples:
        return torch.empty(0), []
    tensor = torch.tensor(np.stack(samples), dtype=torch.float32)
    return tensor, indices


def _predict_lstm(config: PredictConfig, features: pd.DataFrame, schema: dict) -> pd.DataFrame:
    model_base = f"{config.models_uri}/{config.model_name}/{config.model_run_id}"
    info = read_json(f"{model_base}/info.json")
    seq_len = int(info.get("seq_len", 32))
    feature_cols = schema["scaled_columns"]
    inputs, indices = _prepare_lstm_inputs(features, feature_cols, seq_len)
    if inputs.nelement() == 0:
        raise ValueError("No sequences available for LSTM prediction")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OptionLSTM(len(feature_cols), task=info.get("target", "regression"))
    state_dict = torch.load(
        StorageURI(f"{model_base}/state_dict.pt").local_path(), map_location=device
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(inputs.to(device)).cpu().numpy()
    base_cols = [
        "trade_date",
        "symbol",
        "option_key",
        "expiry",
        "type",
        "strike",
        "mid",
        "close",
        "oi",
        "volume",
        "expiry_dte",
        "moneyness",
        "spread_pct",
    ]
    output_df = features.loc[indices, base_cols].copy()
    if info.get("target", "regression") == "classification":
        output_df["score"] = outputs
        output_df["yhat"] = (outputs >= 0.5).astype(int)
        output_df["p_up"] = outputs
    else:
        output_df["score"] = outputs
        output_df["yhat"] = outputs
        output_df["p_up"] = np.nan
    return output_df


def run_predictions(config: PredictConfig):
    features, schema = _load_features(config)
    if config.model_name.startswith("xgb"):
        preds = _predict_xgb(config, features, schema)
    else:
        preds = _predict_lstm(config, features, schema)
    prediction_base = f"{config.output_uri}/{config.symbol}/{config.model_name}/{config.prediction_run_id}"
    predictions_uri = f"{prediction_base}/predictions.parquet"
    metadata_uri = f"{prediction_base}/metadata.json"
    write_parquet(preds, predictions_uri)
    meta = {
        "symbol": config.symbol,
        "model_name": config.model_name,
        "model_run_id": config.model_run_id,
        "prediction_run_id": config.prediction_run_id,
        "rows": int(preds.shape[0]),
    }
    write_json(meta, metadata_uri)
    return {"predictions_uri": predictions_uri, "metadata_uri": metadata_uri}


def _parse_args(args: Optional[Iterable[str]] = None) -> PredictConfig:
    parser = argparse.ArgumentParser(description="Generate predictions from trained models.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--run-id",
        required=True,
        help="Model run id (directory under models_uri/model_name/).",
    )
    parser.add_argument(
        "--prediction-run-id",
        default=None,
        help="Where to write predictions (directory under output_uri/symbol/model_name/). Defaults to --run-id.",
    )
    parser.add_argument("--feature-version", required=True)
    parser.add_argument("--processed-uri", default="data/processed")
    parser.add_argument("--models-uri", default="data/models")
    parser.add_argument("--output-uri", default="data/predictions")
    parser.add_argument("--as-of-date", default=None, help="Optional YYYY-MM-DD filter on trade_date.")
    parsed = parser.parse_args(args=args)
    as_of_date: Optional[date] = None
    if parsed.as_of_date:
        as_of_date = date.fromisoformat(parsed.as_of_date)
    return PredictConfig(
        symbol=parsed.symbol,
        model_name=parsed.model_name,
        model_run_id=parsed.run_id,
        prediction_run_id=parsed.prediction_run_id or parsed.run_id,
        feature_version=parsed.feature_version,
        processed_uri=parsed.processed_uri,
        models_uri=parsed.models_uri,
        output_uri=parsed.output_uri,
        as_of_date=as_of_date,
    )


def main(argv: Optional[Iterable[str]] = None):
    config = _parse_args(argv)
    return run_predictions(config)


if __name__ == "__main__":
    main()
