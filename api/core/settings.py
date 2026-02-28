from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """App configuration — loaded from environment / .env file."""
    env: str = "dev"
    cache_dir: str = "data/cache"

    model_config = {"env_file": ".env", "case_sensitive": False}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
