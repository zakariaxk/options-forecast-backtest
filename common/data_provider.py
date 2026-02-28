"""DataProvider: fetch and cache underlying price data."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd


def _default_cache_dir() -> Path:
    """Use data/cache locally, /tmp/stockpulse_cache on Render / cloud."""
    env = os.getenv("ENV", "dev")
    if env == "production":
        p = Path("/tmp/stockpulse_cache")
    else:
        p = Path("data/cache")
    p.mkdir(parents=True, exist_ok=True)
    return p


_CACHE_DIR = _default_cache_dir()


def _cache_key(symbol: str, start: date, end: date) -> str:
    """Deterministic cache key for a price data request."""
    raw = f"{symbol}_{start.isoformat()}_{end.isoformat()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _cache_path(symbol: str, start: date, end: date) -> Path:
    return _CACHE_DIR / symbol / f"{_cache_key(symbol, start, end)}.parquet"


def fetch_underlying_prices(
    symbol: str,
    start_date: date,
    end_date: date,
    *,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Fetch daily OHLCV for *symbol* between *start_date* and *end_date*.

    Data is cached locally after the first fetch to guarantee deterministic
    replay on subsequent calls with the same arguments.

    Returns a DataFrame with columns:
        date (datetime64), open, high, low, close, volume
    sorted by date ascending.  Raises ValueError when yfinance returns no data
    (bad symbol or date range).
    """
    cache = (cache_dir or _CACHE_DIR) / symbol
    cache.mkdir(parents=True, exist_ok=True)
    parquet_path = cache / f"{_cache_key(symbol, start_date, end_date)}.parquet"

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        return df

    # Fetch from yfinance  ------------------------------------------------
    import yfinance as yf  # deferred import: heavy + has side effects

    ticker = yf.Ticker(symbol)
    # yfinance end is exclusive — add one day
    raw = ticker.history(
        start=start_date.isoformat(),
        end=(end_date + pd.tseries.offsets.BDay(1)).date().isoformat(),
        auto_adjust=True,
    )
    if raw.empty:
        raise ValueError(
            f"yfinance returned no data for {symbol} "
            f"({start_date} – {end_date}). Check symbol and date range."
        )

    df = (
        raw.reset_index()
        .rename(columns=lambda c: c.lower().replace(" ", "_"))
        .rename(columns={"date": "date"})
    )
    # Keep only the columns we need
    df = df[["date", "open", "high", "low", "close", "volume"]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values("date").reset_index(drop=True)

    # Filter to exact range (yfinance sometimes returns extra rows)
    mask = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
    df = df.loc[mask].reset_index(drop=True)

    if df.empty:
        raise ValueError(
            f"No trading days found for {symbol} between {start_date} and {end_date}."
        )

    df.to_parquet(parquet_path, index=False)
    return df
