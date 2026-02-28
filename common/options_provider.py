"""Options data provider — fetch chains from yfinance."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class OptionsChainResult:
    symbol: str
    expiry: str
    underlying_price: float
    calls: List[Dict[str, Any]]
    puts: List[Dict[str, Any]]
    available_expiries: List[str]

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "expiry": self.expiry,
            "underlying_price": self.underlying_price,
            "calls": self.calls,
            "puts": self.puts,
            "available_expiries": self.available_expiries,
        }


def _clean_chain(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert yfinance option chain DataFrame to clean list of dicts."""
    if df.empty:
        return []

    keep_cols = [
        "strike", "lastPrice", "bid", "ask", "volume",
        "openInterest", "impliedVolatility", "inTheMoney",
    ]
    # Only keep columns that exist
    cols = [c for c in keep_cols if c in df.columns]
    clean = df[cols].copy()

    # Rename for consistency
    rename = {
        "lastPrice": "last_price",
        "openInterest": "open_interest",
        "impliedVolatility": "implied_vol",
        "inTheMoney": "itm",
    }
    clean = clean.rename(columns={k: v for k, v in rename.items() if k in clean.columns})

    # Round floats
    for col in ["strike", "last_price", "bid", "ask", "implied_vol"]:
        if col in clean.columns:
            clean[col] = clean[col].round(4)

    # Fill NaN with 0 for numeric cols
    for col in ["volume", "open_interest"]:
        if col in clean.columns:
            clean[col] = clean[col].fillna(0).astype(int)

    return clean.to_dict(orient="records")


def fetch_options_chain(
    symbol: str,
    expiry: Optional[str] = None,
) -> OptionsChainResult:
    """Fetch options chain for *symbol*.

    Parameters
    ----------
    symbol : ticker symbol
    expiry : expiration date string (YYYY-MM-DD). If None, uses nearest expiry.

    Returns
    -------
    OptionsChainResult with calls, puts, and available expiries.
    """
    import yfinance as yf

    ticker = yf.Ticker(symbol)

    # Get available expiries
    try:
        available = list(ticker.options)
    except Exception:
        available = []

    if not available:
        raise ValueError(f"No options data available for {symbol}")

    # Pick expiry
    if expiry is None:
        target_expiry = available[0]
    else:
        if expiry in available:
            target_expiry = expiry
        else:
            # Find nearest
            target_date = date.fromisoformat(expiry)
            nearest = min(available, key=lambda e: abs((date.fromisoformat(e) - target_date).days))
            target_expiry = nearest

    # Fetch chain
    chain = ticker.option_chain(target_expiry)

    # Get underlying price
    info = ticker.fast_info
    underlying_price = float(getattr(info, "last_price", 0) or 0)
    if underlying_price == 0:
        # Fallback
        hist = ticker.history(period="1d")
        if not hist.empty:
            underlying_price = float(hist["Close"].iloc[-1])

    return OptionsChainResult(
        symbol=symbol.upper(),
        expiry=target_expiry,
        underlying_price=round(underlying_price, 2),
        calls=_clean_chain(chain.calls),
        puts=_clean_chain(chain.puts),
        available_expiries=available,
    )
