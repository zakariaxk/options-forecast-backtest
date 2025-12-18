from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from pymongo import MongoClient


@lru_cache(maxsize=1)
def _client() -> MongoClient:
    uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    return MongoClient(uri, tz_aware=True)


def get_database(name: str | None = None):
    client = _client()
    db_name = name or os.getenv("MONGO_DB", "options_forecast")
    return client[db_name]


def get_collection(collection: str, *, db_name: str | None = None):
    db = get_database(db_name)
    return db[collection]


def close_client() -> None:
    if _client.cache_info().currsize:
        client = _client()
        client.close()
        _client.cache_clear()
