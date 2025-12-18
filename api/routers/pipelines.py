from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Query

from api.core.errors import bad_request, not_found
from api.core.logging import get_logger
from api.services.pipelines import latest_raw_partition, run_features, run_ingestion, run_train
from common.schema import FeatureRequest, FeatureResponse, IngestRequest, IngestResponse, TrainRequest, TrainResponse

router = APIRouter(prefix="/pipelines", tags=["pipelines"])
logger = get_logger("pipelines")


@router.post("/ingest", response_model=IngestResponse)
def ingest_yahoo(request: IngestRequest) -> IngestResponse:
    result = run_ingestion(request)
    logger.info("ingest_done", symbol=result.symbol, partition=result.partition)
    return result


@router.post("/features", response_model=FeatureResponse)
def features_build(request: FeatureRequest) -> FeatureResponse:
    result = run_features(request)
    logger.info("features_done", symbol=result.symbol, version=result.version)
    return result


@router.post("/train", response_model=TrainResponse)
def train(request: TrainRequest) -> TrainResponse:
    result = run_train(request)
    logger.info("train_done", symbol=result.symbol, model_name=result.model_name, run_name=result.run_name)
    return result


@router.get("/latest-raw-partition/{symbol}")
def get_latest_partition(symbol: str) -> dict:
    partition = latest_raw_partition(symbol)
    if not partition:
        raise not_found(f"No raw partitions found for {symbol}")
    return {"symbol": symbol, "raw_partition": partition}


@router.post("/run-all")
def run_all(
    symbol: str = Query(...),
    start_date: date = Query(...),
    end_date: date = Query(...),
    feature_version: str = Query(default="v1"),
    model_name: str = Query(default="xgb_reg"),
    model_run_name: str = Query(default="demo"),
) -> dict:
    try:
        ingest_res = run_ingestion(
            IngestRequest(symbol=symbol, start_date=start_date, end_date=end_date)
        )
    except Exception as exc:
        raise bad_request(f"Ingest failed: {exc}")
    try:
        feat_res = run_features(
            FeatureRequest(symbol=symbol, raw_partition=ingest_res.partition, version=feature_version)
        )
    except Exception as exc:
        raise bad_request(f"Features failed: {exc}")
    try:
        train_res = run_train(
            TrainRequest(
                symbol=symbol,
                feature_version=feature_version,
                run_name=model_run_name,
                model_name=model_name,  # type: ignore[arg-type]
            )
        )
    except Exception as exc:
        raise bad_request(f"Train failed: {exc}")
    return {"ingest": ingest_res.dict(), "features": feat_res.dict(), "train": train_res.dict()}
