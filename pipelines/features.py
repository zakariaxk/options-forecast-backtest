from __future__ import annotations

import argparse
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from common.io import StorageURI, read_parquet, write_json, write_parquet
from common.schema import FeatureConfig


def _load_equity(raw_partition_uri: str) -> pd.DataFrame:
    path = f"{raw_partition_uri}/equity.parquet"
    df = read_parquet(path)
    df["trade_date"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(None).dt.normalize()
    return df


def _load_options(raw_partition_uri: str) -> pd.DataFrame:
    path = f"{raw_partition_uri}/options.parquet"
    df = read_parquet(path)
    df["trade_date"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(None).dt.normalize()
    df["option_key"] = (
        df["symbol"]
        + "_"
        + df["expiry"].astype(str)
        + "_"
        + df["type"]
        + "_"
        + df["strike"].astype(str)
    )
    return df


def _compute_equity_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("trade_date").copy()
    df["r1"] = df["close"].pct_change(1)
    df["r5"] = df["close"].pct_change(5)
    df["r21"] = df["close"].pct_change(21)
    df["rsi_14"] = _rsi(df["close"], window=14)
    df["macd"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["bb_upper"] = df["close"].rolling(20).mean() + 2 * df["close"].rolling(20).std()
    df["bb_lower"] = df["close"].rolling(20).mean() - 2 * df["close"].rolling(20).std()
    df["bb_pct_20"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    eq_cols = [
        "trade_date",
        "symbol",
        "close",
        "r1",
        "r5",
        "r21",
        "rsi_14",
        "macd",
        "bb_pct_20",
    ]
    return df[eq_cols]


def _rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _compute_option_features(
    options: pd.DataFrame,
    equity_features: pd.DataFrame,
    config: FeatureConfig,
) -> pd.DataFrame:
    df = options.merge(equity_features, on=["trade_date", "symbol"], how="left", suffixes=("", "_eq"))
    if "close" in df.columns:
        df["close"] = df["close"].fillna(df["mid"])
    df["expiry_dte"] = (
        pd.to_datetime(df["expiry"]).dt.normalize() - df["trade_date"]
    ).dt.days
    df["moneyness"] = (df["close"] - df["strike"]) / df["close"]
    df["spread_pct"] = (df["ask"] - df["bid"]) / df["mid"].replace(0, np.nan)
    df["dow"] = df["trade_date"].dt.dayofweek
    df["dom"] = df["trade_date"].dt.day
    target = _compute_targets(df, horizon=config.horizon_days, threshold=config.classification_threshold)
    df = df.join(target)
    df = df.dropna(subset=["mid"])
    return df


def _compute_targets(df: pd.DataFrame, horizon: int, threshold: float) -> pd.DataFrame:
    df_option = df.sort_values(["option_key", "trade_date"]).copy()
    future_mid = df_option.groupby("option_key")["mid"].shift(-horizon)
    regression_option = (future_mid - df_option["mid"]) / df_option["mid"]
    regression = pd.Series(regression_option, index=df_option.index).reindex(df.index)

    df_date = df.sort_values("trade_date")
    future_close = df_date.groupby("symbol")["close"].shift(-horizon)
    fallback_series = (future_close - df_date["close"]) / df_date["close"]
    fallback = pd.Series(fallback_series, index=df_date.index).reindex(df.index)

    regression = regression.fillna(fallback)
    regression = regression.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    classification = (regression > threshold).astype(int)
    return pd.DataFrame({"target_reg": regression, "target_bin": classification}, index=df.index)


def _scale_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler, list[str]]:
    feature_cols = [
        "mid",
        "iv",
        "delta",
        "gamma",
        "theta",
        "vega",
        "oi",
        "volume",
        "r1",
        "r5",
        "r21",
        "rsi_14",
        "macd",
        "bb_pct_20",
        "expiry_dte",
        "moneyness",
        "spread_pct",
        "dow",
        "dom",
    ]
    features = df[feature_cols].replace([np.inf, -np.inf], 0.0).astype(np.float32).fillna(0.0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features.values)
    scaled_df = pd.DataFrame(scaled, columns=[f"{col}_z" for col in feature_cols], index=df.index)
    out = pd.concat([df, scaled_df], axis=1)
    return out, scaler, feature_cols


def build_features(config: FeatureConfig) -> dict[str, str]:
    raw_partition_uri = f"{config.raw_uri}/{config.raw_partition}"
    equity = _load_equity(raw_partition_uri)
    print(f"Equity rows: {len(equity)}")
    options = _load_options(raw_partition_uri)
    print(f"Options rows: {len(options)}")
    equity_features = _compute_equity_features(equity)
    joined = _compute_option_features(options, equity_features, config)
    print(f"Joined rows: {len(joined)}")
    processed, scaler, feature_cols = _scale_features(joined)
    output_base = f"{config.processed_uri}/{config.symbol}/{config.version}"
    features_uri = f"{output_base}/features.parquet"
    schema_uri = f"{output_base}/schema.json"
    scaler_uri = f"{output_base}/scaler.pkl"
    data_hash = write_parquet(processed, features_uri)
    schema = {
        "symbol": config.symbol,
        "version": config.version,
        "horizon_days": config.horizon_days,
        "feature_columns": feature_cols,
        "scaled_columns": [f"{col}_z" for col in feature_cols],
        "hash": data_hash,
        "rows": int(processed.shape[0]),
    }
    write_json(schema, schema_uri)
    scaler_path = StorageURI(scaler_uri).local_path()
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    with scaler_path.open("wb") as fh:
        joblib.dump({"scaler": scaler, "feature_columns": feature_cols}, fh)
    return {
        "features_uri": features_uri,
        "schema_uri": schema_uri,
        "scaler_uri": scaler_uri,
        "hash": data_hash,
    }


def _parse_args(args: Optional[Iterable[str]] = None) -> FeatureConfig:
    parser = argparse.ArgumentParser(description="Build features from ingested data.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--raw-uri", default="data/raw")
    parser.add_argument("--raw-partition", required=True)
    parser.add_argument("--processed-uri", default="data/processed")
    parser.add_argument("--horizon-days", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.0)
    parsed = parser.parse_args(args=args)
    return FeatureConfig(
        symbol=parsed.symbol,
        version=parsed.version,
        raw_uri=parsed.raw_uri,
        raw_partition=parsed.raw_partition,
        processed_uri=parsed.processed_uri,
        horizon_days=parsed.horizon_days,
        classification_threshold=parsed.threshold,
    )


def main(argv: Optional[Iterable[str]] = None) -> dict[str, str]:
    config = _parse_args(argv)
    return build_features(config)


if __name__ == "__main__":
    main()
