from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from api.core.settings import get_settings
from api.schemas.io import ModelInfo
from common.schema import PredictConfig, PredictRequest, PredictResponse
from pipelines.predict import run_predictions


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d%H%M%S")


def list_models() -> list[ModelInfo]:
    settings = get_settings()
    models_dir = Path(settings.models_uri)
    items: list[ModelInfo] = []
    if models_dir.exists():
        for model_path in sorted(models_dir.iterdir()):
            if model_path.is_dir():
                runs = sorted(child.name for child in model_path.iterdir() if child.is_dir())
                items.append(ModelInfo(name=model_path.name, runs=runs))
    return items


def generate_predictions(req: PredictRequest) -> PredictResponse:
    settings = get_settings()
    model_name = req.model_name or _default_model_name()
    model_run_id = req.model_run_id or _latest_model_run_id(model_name)
    prediction_run_id = req.prediction_run_id or f"pred_{_timestamp()}"
    config = PredictConfig(
        symbol=req.symbol,
        model_name=model_name,
        model_run_id=model_run_id,
        prediction_run_id=prediction_run_id,
        feature_version=req.feature_version,
        processed_uri=settings.processed_uri,
        models_uri=settings.models_uri,
        output_uri=settings.predictions_uri,
        as_of_date=req.as_of_date,
        horizon=req.horizon,
    )
    result = run_predictions(config)
    predictions_uri = Path(result["predictions_uri"]).resolve()
    df = pd.read_parquet(predictions_uri)
    return PredictResponse(
        prediction_run_id=prediction_run_id,
        symbol=req.symbol,
        model_name=model_name,
        model_run_id=model_run_id,
        n=int(len(df)),
        uri=predictions_uri.as_uri(),
    )


def fetch_predictions(
    symbol: str,
    prediction_run_id: str,
    model_name: Optional[str] = None,
) -> pd.DataFrame:
    settings = get_settings()
    model = model_name or _default_model_name()
    base = Path(settings.predictions_uri) / symbol / model / prediction_run_id
    file_path = base / "predictions.parquet"
    if not file_path.exists():
        raise FileNotFoundError(f"Predictions file not found at {file_path}")
    return pd.read_parquet(file_path)


def _default_model_name() -> str:
    models = list_models()
    if not models:
        return "xgb_reg"
    return models[0].name


def _latest_model_run_id(model_name: str) -> str:
    settings = get_settings()
    model_dir = Path(settings.models_uri) / model_name
    if not model_dir.exists():
        raise FileNotFoundError(f"Model {model_name} not found at {model_dir}")
    runs = sorted([p for p in model_dir.iterdir() if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No runs found for model {model_name} at {model_dir}")
    return runs[-1].name


def latest_model_metrics(model_name: str) -> dict:
    settings = get_settings()
    model_dir = Path(settings.models_uri) / model_name
    if not model_dir.exists():
        raise FileNotFoundError(f"Model {model_name} not found")
    runs = sorted([p for p in model_dir.iterdir() if p.is_dir()])
    for run_dir in reversed(runs):
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            return {
                "model_name": model_name,
                "run_id": run_dir.name,
                "metrics": json.loads(metrics_path.read_text()),
            }
    raise FileNotFoundError(f"No metrics found for model {model_name}")


def list_prediction_runs() -> list[dict]:
    settings = get_settings()
    base = Path(settings.predictions_uri)
    items: list[dict] = []
    if not base.exists():
        return items
    for symbol_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
        for model_dir in sorted([p for p in symbol_dir.iterdir() if p.is_dir()]):
            for run_dir in sorted([p for p in model_dir.iterdir() if p.is_dir()]):
                items.append(
                    {
                        "symbol": symbol_dir.name,
                        "model_name": model_dir.name,
                        "prediction_run_id": run_dir.name,
                    }
                )
    return items


def prediction_artifact_paths(symbol: str, model_name: str, prediction_run_id: str) -> dict[str, str]:
    settings = get_settings()
    base = Path(settings.predictions_uri) / symbol / model_name / prediction_run_id
    return {
        "predictions_uri": (base / "predictions.parquet").as_posix(),
        "metadata_uri": (base / "metadata.json").as_posix(),
    }
