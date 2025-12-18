from __future__ import annotations

from fastapi import APIRouter, Query

from api.core.logging import get_logger
from api.schemas.io import ModelInfo
from api.core.errors import not_found
from api.services.inference import (
    fetch_predictions,
    generate_predictions,
    list_models,
    list_prediction_runs,
    prediction_artifact_paths,
)
from common.schema import PredictRequest, PredictResponse

router = APIRouter(prefix="/predictions", tags=["predictions"])
logger = get_logger("predict")


@router.post("/", response_model=PredictResponse)
def create_prediction(request: PredictRequest) -> PredictResponse:
    response = generate_predictions(request)
    logger.info(
        "prediction_generated",
        prediction_run_id=response.prediction_run_id,
        model_name=response.model_name,
        model_run_id=response.model_run_id,
        symbol=request.symbol,
    )
    return response


@router.get("/{symbol}/{run_id}")
def get_prediction(symbol: str, run_id: str, model_name: str | None = Query(default=None)) -> dict:
    resolved_model = model_name
    if resolved_model is None:
        models = list_models()
        resolved_model = models[0].name if models else "xgb_reg"
    try:
        df = fetch_predictions(symbol=symbol, prediction_run_id=run_id, model_name=resolved_model)
    except FileNotFoundError as exc:
        raise not_found(str(exc))
    return {
        "prediction_run_id": run_id,
        "symbol": symbol,
        "model_name": resolved_model,
        "n": int(len(df)),
        **prediction_artifact_paths(symbol=symbol, model_name=resolved_model, prediction_run_id=run_id),
    }


@router.get("/{symbol}/{run_id}/data")
def get_prediction_data(
    symbol: str,
    run_id: str,
    model_name: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=5000),
    sort: str = Query(default="score"),
    descending: bool = Query(default=True),
) -> dict:
    resolved_model = model_name
    if resolved_model is None:
        models = list_models()
        resolved_model = models[0].name if models else "xgb_reg"
    try:
        df = fetch_predictions(symbol=symbol, prediction_run_id=run_id, model_name=resolved_model)
    except FileNotFoundError as exc:
        raise not_found(str(exc))
    if sort in df.columns:
        df = df.sort_values(sort, ascending=not descending)
    data = df.head(limit).to_dict(orient="records")
    return {
        "prediction_run_id": run_id,
        "symbol": symbol,
        "model_name": resolved_model,
        "n": int(len(df)),
        "returned": len(data),
        "columns": list(df.columns),
        "rows": data,
    }


@router.get("/models", response_model=list[ModelInfo])
def list_available_models() -> list[ModelInfo]:
    return list_models()


@router.get("/runs")
def list_runs() -> list[dict]:
    return list_prediction_runs()
