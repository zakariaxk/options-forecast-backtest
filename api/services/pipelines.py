from __future__ import annotations

from pathlib import Path

from api.core.settings import get_settings
from common.schema import (
    FeatureConfig,
    FeatureRequest,
    FeatureResponse,
    IngestConfig,
    IngestRequest,
    IngestResponse,
    TrainConfig,
    TrainRequest,
    TrainResponse,
)
from pipelines.features import build_features
from pipelines.ingest_yf import ingest
from pipelines.train_torch import train_lstm
from pipelines.train_xgb import train_model


def run_ingestion(req: IngestRequest) -> IngestResponse:
    settings = get_settings()
    cfg = IngestConfig(
        symbol=req.symbol,
        start_date=req.start_date,
        end_date=req.end_date,
        dest_uri=settings.raw_uri,
        min_open_interest=req.min_open_interest,
        min_volume=req.min_volume,
        max_expiries=req.max_expiries,
        max_contracts_per_expiry=req.max_contracts_per_expiry,
    )
    result = ingest(cfg)
    return IngestResponse(
        symbol=result.symbol,
        partition=result.partition,
        equity_uri=result.equity_uri,
        options_uri=result.options_uri,
        metadata_uri=result.metadata_uri,
    )


def run_features(req: FeatureRequest) -> FeatureResponse:
    settings = get_settings()
    cfg = FeatureConfig(
        symbol=req.symbol,
        version=req.version,
        raw_uri=settings.raw_uri,
        raw_partition=req.raw_partition,
        processed_uri=settings.processed_uri,
        horizon_days=req.horizon_days,
        classification_threshold=req.classification_threshold,
    )
    out = build_features(cfg)
    return FeatureResponse(
        symbol=req.symbol,
        version=req.version,
        features_uri=out["features_uri"],
        schema_uri=out["schema_uri"],
        scaler_uri=out["scaler_uri"],
        hash=out["hash"],
    )


def run_train(req: TrainRequest) -> TrainResponse:
    settings = get_settings()
    cfg = TrainConfig(
        symbol=req.symbol,
        feature_version=req.feature_version,
        run_name=req.run_name,
        model_name=req.model_name,
        target=req.target,
        processed_uri=settings.processed_uri,
        output_uri=settings.models_uri,
        seed=req.seed,
        seq_len=req.seq_len,
        batch_size=req.batch_size,
        epochs=req.epochs,
        learning_rate=req.learning_rate,
        weight_decay=req.weight_decay,
    )
    if cfg.model_name == "torch_lstm":
        artifacts = train_lstm(cfg)
    else:
        artifacts = train_model(cfg)
    return TrainResponse(
        symbol=req.symbol,
        model_name=cfg.model_name,
        run_name=req.run_name,
        artifacts=artifacts,
    )


def latest_raw_partition(symbol: str) -> str | None:
    settings = get_settings()
    base = Path(settings.raw_uri) / symbol
    if not base.exists():
        return None
    partitions = sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not partitions:
        return None
    return f"{symbol}/{partitions[-1].name}"
