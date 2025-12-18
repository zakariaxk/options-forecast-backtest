from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.services.inference import latest_model_metrics, list_models
from api.schemas.io import ModelInfo

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/", response_model=list[ModelInfo])
def get_models() -> list[ModelInfo]:
    return list_models()


@router.get("/{model_name}/latest")
def get_latest(model_name: str) -> dict:
    try:
        return latest_model_metrics(model_name)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
