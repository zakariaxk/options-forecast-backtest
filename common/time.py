from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, List

import pandas as pd

UTC = timezone.utc


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def parse_date(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(UTC)
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(UTC)
    else:
        ts = ts.tz_convert(UTC)
    return ts.to_pydatetime()


def trading_days(start: str | datetime, end: str | datetime) -> List[pd.Timestamp]:
    start_dt = parse_date(start)
    end_dt = parse_date(end)
    index = pd.bdate_range(start=start_dt.date(), end=end_dt.date(), freq="B")
    return list(index.tz_localize(UTC))


def to_utc_series(series: Iterable[str | datetime]) -> pd.Series:
    values = [parse_date(val) for val in series]
    return pd.Series(values, dtype="datetime64[ns, UTC]")
