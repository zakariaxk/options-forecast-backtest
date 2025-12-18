from __future__ import annotations

import os
from functools import lru_cache

import redis


@lru_cache(maxsize=1)
def _client() -> redis.Redis:
    url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    return redis.Redis.from_url(url, decode_responses=True)


def ping() -> bool:
    try:
        return bool(_client().ping())
    except redis.RedisError:
        return False


def get_client() -> redis.Redis:
    return _client()
