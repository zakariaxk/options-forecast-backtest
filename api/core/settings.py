from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    env: str = "dev"
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "options_forecast"
    redis_url: str = "redis://localhost:6379/0"
    s3_bucket: str = "ofb"
    s3_endpoint_url: Optional[str] = None
    raw_uri: str = "data/raw"
    models_uri: str = "data/models"
    predictions_uri: str = "data/predictions"
    processed_uri: str = "data/processed"
    backtests_uri: str = "data/backtests"
    secret_key: str = "change-me"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
