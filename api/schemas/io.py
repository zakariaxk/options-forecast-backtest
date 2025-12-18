from __future__ import annotations

from datetime import datetime
from typing import List

from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: str
    time: datetime


class ModelInfo(BaseModel):
    name: str
    runs: List[str]
