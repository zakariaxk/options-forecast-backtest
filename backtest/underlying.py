"""Underlying backtest engine.

Runs deterministic backtests over daily OHLCV price data for a single
equity.  Strategies: buy & hold, SMA crossover, RSI mean-reversion.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from common.data_provider import fetch_underlying_prices


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class UnderlyingBacktestResult:
    """Immutable result of an underlying-only backtest."""
    bt_id: str
    symbol: str
    strategy: str
    start_date: str
    end_date: str
    initial_cash: float
    summary: Dict[str, float]
    equity_curve: List[Dict[str, Any]]
    trades: List[Dict[str, Any]]
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "bt_id": self.bt_id,
            "symbol": self.symbol,
            "strategy": self.strategy,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_cash": self.initial_cash,
            "summary": self.summary,
            "equity_curve": self.equity_curve,
            "trades": self.trades,
            "params": self.params,
        }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(equity_curve: pd.DataFrame, initial_cash: float) -> Dict[str, float]:
    """Compute summary metrics from a date/nav equity curve DataFrame.

    Returns a dict that is safe to JSON-serialise (no NaN / Inf).
    """
    if equity_curve.empty or len(equity_curve) < 2:
        return _safe_dict({
            "total_return": 0.0,
            "cagr": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "volatility": 0.0,
            "calmar": 0.0,
        })

    navs = equity_curve["nav"].values.astype(float)
    returns = np.diff(navs) / navs[:-1]

    total_return = (navs[-1] - initial_cash) / initial_cash

    n_days = len(returns)
    ann_factor = 252  # trading days per year

    mean_ret = float(np.mean(returns))
    vol = float(np.std(returns, ddof=0))
    ann_vol = vol * math.sqrt(ann_factor)

    sharpe = (mean_ret / vol * math.sqrt(ann_factor)) if vol > 0 else 0.0

    down = returns[returns < 0]
    down_std = float(np.std(down, ddof=0)) if len(down) > 0 else 0.0
    sortino = (mean_ret / down_std * math.sqrt(ann_factor)) if down_std > 0 else 0.0

    # CAGR
    years = n_days / ann_factor
    if initial_cash > 0 and navs[-1] > 0 and years > 0:
        cagr = (navs[-1] / initial_cash) ** (1 / years) - 1
    else:
        cagr = 0.0

    # Max drawdown
    peak = np.maximum.accumulate(navs)
    dd = (navs - peak) / np.where(peak > 0, peak, 1.0)
    max_dd = float(np.min(dd))

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    return _safe_dict({
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "volatility": ann_vol,
        "calmar": calmar,
    })


def _safe_dict(d: dict) -> dict:
    """Replace NaN/Inf with 0.0 so JSON serialisation never fails."""
    out = {}
    for k, v in d.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            out[k] = 0.0
        else:
            out[k] = round(v, 8)
    return out


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

UNDERLYING_STRATEGIES = {"buy_and_hold", "sma_crossover", "rsi_mean_reversion"}

# Default strategy parameters
DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    "buy_and_hold": {},
    "sma_crossover": {"fast_period": 20, "slow_period": 50},
    "rsi_mean_reversion": {"rsi_period": 14, "oversold": 30, "overbought": 70},
}


def _date_str(val: Any) -> str:
    """Convert a date-like value to ISO string."""
    if hasattr(val, "isoformat"):
        return val.isoformat()
    return str(val)


def run_buy_and_hold(
    prices: pd.DataFrame,
    initial_cash: float,
    **_kwargs: Any,
) -> tuple[pd.DataFrame, list[dict]]:
    """Execute a buy-and-hold strategy on daily close prices.

    Parameters
    ----------
    prices : DataFrame with columns [date, close] (at minimum)
    initial_cash : starting cash

    Returns
    -------
    equity_curve : DataFrame with columns [date, nav]
    trades : list of trade dicts
    """
    if prices.empty:
        return pd.DataFrame(columns=["date", "nav"]), []

    first_close = float(prices.iloc[0]["close"])
    shares = math.floor(initial_cash / first_close)
    if shares <= 0:
        raise ValueError(
            f"initial_cash {initial_cash} is not enough to buy a single share at {first_close}"
        )
    cash_remainder = initial_cash - shares * first_close

    trades = [
        {
            "date": _date_str(prices.iloc[0]["date"]),
            "action": "BUY",
            "symbol": "",  # filled by caller
            "qty": shares,
            "price": round(first_close, 4),
            "value": round(shares * first_close, 4),
        }
    ]

    equity_rows = []
    for _, row in prices.iterrows():
        nav = cash_remainder + shares * float(row["close"])
        equity_rows.append({
            "date": row["date"],
            "nav": round(nav, 4),
        })

    equity_df = pd.DataFrame(equity_rows)
    return equity_df, trades


def run_sma_crossover(
    prices: pd.DataFrame,
    initial_cash: float,
    *,
    fast_period: int = 20,
    slow_period: int = 50,
    **_kwargs: Any,
) -> tuple[pd.DataFrame, list[dict]]:
    """SMA crossover strategy.

    BUY when the fast SMA crosses above the slow SMA.
    SELL when the fast SMA crosses below the slow SMA.
    Position is always fully in or fully out.
    """
    if prices.empty:
        return pd.DataFrame(columns=["date", "nav"]), []

    if fast_period >= slow_period:
        raise ValueError(f"fast_period ({fast_period}) must be < slow_period ({slow_period})")
    if len(prices) < slow_period:
        raise ValueError(
            f"Need at least {slow_period} price bars for SMA crossover, got {len(prices)}"
        )

    closes = prices["close"].astype(float).values
    dates = prices["date"].values

    fast_sma = _rolling_mean(closes, fast_period)
    slow_sma = _rolling_mean(closes, slow_period)

    cash = initial_cash
    shares = 0
    trades: list[dict] = []
    equity_rows: list[dict] = []

    for i in range(len(closes)):
        price = closes[i]
        dt = _date_str(dates[i])

        # Wait until both SMAs are valid (i >= slow_period - 1)
        if i >= slow_period and fast_sma[i] is not None and slow_sma[i] is not None:
            prev_fast = fast_sma[i - 1]
            prev_slow = slow_sma[i - 1]

            # Cross above → BUY
            if prev_fast is not None and prev_slow is not None:
                if prev_fast <= prev_slow and fast_sma[i] > slow_sma[i] and shares == 0:
                    shares = math.floor(cash / price)
                    if shares > 0:
                        cost = shares * price
                        cash -= cost
                        trades.append({
                            "date": dt,
                            "action": "BUY",
                            "symbol": "",
                            "qty": shares,
                            "price": round(price, 4),
                            "value": round(cost, 4),
                        })

                # Cross below → SELL
                elif prev_fast >= prev_slow and fast_sma[i] < slow_sma[i] and shares > 0:
                    proceeds = shares * price
                    trades.append({
                        "date": dt,
                        "action": "SELL",
                        "symbol": "",
                        "qty": shares,
                        "price": round(price, 4),
                        "value": round(proceeds, 4),
                    })
                    cash += proceeds
                    shares = 0

        nav = cash + shares * price
        equity_rows.append({"date": dates[i], "nav": round(nav, 4)})

    equity_df = pd.DataFrame(equity_rows)
    return equity_df, trades


def run_rsi_mean_reversion(
    prices: pd.DataFrame,
    initial_cash: float,
    *,
    rsi_period: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
    **_kwargs: Any,
) -> tuple[pd.DataFrame, list[dict]]:
    """RSI mean-reversion strategy.

    BUY when RSI crosses below oversold threshold.
    SELL when RSI crosses above overbought threshold.
    Position is always fully in or fully out.
    """
    if prices.empty:
        return pd.DataFrame(columns=["date", "nav"]), []

    if len(prices) < rsi_period + 1:
        raise ValueError(
            f"Need at least {rsi_period + 1} price bars for RSI strategy, got {len(prices)}"
        )
    if not (0 < oversold < overbought < 100):
        raise ValueError(
            f"Invalid RSI thresholds: oversold={oversold}, overbought={overbought}"
        )

    closes = prices["close"].astype(float).values
    dates = prices["date"].values
    rsi_values = _compute_rsi(closes, rsi_period)

    cash = initial_cash
    shares = 0
    trades: list[dict] = []
    equity_rows: list[dict] = []

    for i in range(len(closes)):
        price = closes[i]
        dt = _date_str(dates[i])

        if rsi_values[i] is not None:
            rsi = rsi_values[i]

            # RSI below oversold → BUY
            if rsi < oversold and shares == 0:
                shares = math.floor(cash / price)
                if shares > 0:
                    cost = shares * price
                    cash -= cost
                    trades.append({
                        "date": dt,
                        "action": "BUY",
                        "symbol": "",
                        "qty": shares,
                        "price": round(price, 4),
                        "value": round(cost, 4),
                    })

            # RSI above overbought → SELL
            elif rsi > overbought and shares > 0:
                proceeds = shares * price
                trades.append({
                    "date": dt,
                    "action": "SELL",
                    "symbol": "",
                    "qty": shares,
                    "price": round(price, 4),
                    "value": round(proceeds, 4),
                })
                cash += proceeds
                shares = 0

        nav = cash + shares * price
        equity_rows.append({"date": dates[i], "nav": round(nav, 4)})

    equity_df = pd.DataFrame(equity_rows)
    return equity_df, trades


# ---------------------------------------------------------------------------
# Technical indicator helpers
# ---------------------------------------------------------------------------

def _rolling_mean(data: np.ndarray, period: int) -> list[float | None]:
    """Simple moving average. Returns None for indices < period-1."""
    result: list[float | None] = [None] * len(data)
    for i in range(period - 1, len(data)):
        result[i] = float(np.mean(data[i - period + 1: i + 1]))
    return result


def _compute_rsi(closes: np.ndarray, period: int) -> list[float | None]:
    """Wilder's RSI. Returns None for indices < period."""
    result: list[float | None] = [None] * len(closes)
    if len(closes) < period + 1:
        return result

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Seed with simple average over first `period` bars
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))

    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - 100.0 / (1.0 + rs)

    # Wilder smoothing for subsequent bars
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            result[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i + 1] = 100.0 - 100.0 / (1.0 + rs)

    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_underlying_backtest(
    *,
    symbol: str,
    strategy: str,
    start_date: date,
    end_date: date,
    initial_cash: float = 100_000.0,
    bt_id: str = "",
    prices: pd.DataFrame | None = None,
    params: Optional[Dict[str, Any]] = None,
) -> UnderlyingBacktestResult:
    """Top-level function: fetch data → run strategy → compute metrics.

    If *prices* is provided it is used directly (useful for testing without
    network calls).  Otherwise yfinance data is fetched and cached.

    *params* are strategy-specific overrides (e.g. fast_period, rsi_period).
    Missing keys fall back to DEFAULT_PARAMS for the strategy.
    """
    if strategy not in UNDERLYING_STRATEGIES:
        raise ValueError(f"Unknown underlying strategy: {strategy}")

    if prices is None:
        prices = fetch_underlying_prices(symbol, start_date, end_date)

    # Merge user params with defaults
    effective_params = {**DEFAULT_PARAMS.get(strategy, {}), **(params or {})}

    _strategy_dispatch = {
        "buy_and_hold": run_buy_and_hold,
        "sma_crossover": run_sma_crossover,
        "rsi_mean_reversion": run_rsi_mean_reversion,
    }

    runner = _strategy_dispatch.get(strategy)
    if runner is None:
        raise ValueError(f"Strategy {strategy} not implemented")

    equity_df, trades = runner(prices, initial_cash, **effective_params)

    # Stamp symbol into trades
    for t in trades:
        t["symbol"] = symbol

    summary = compute_metrics(equity_df, initial_cash)

    # Serialise equity curve for JSON
    equity_list = [
        {
            "date": _date_str(row["date"]),
            "nav": round(float(row["nav"]), 4),
        }
        for _, row in equity_df.iterrows()
    ]

    return UnderlyingBacktestResult(
        bt_id=bt_id,
        symbol=symbol,
        strategy=strategy,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        initial_cash=initial_cash,
        summary=summary,
        equity_curve=equity_list,
        trades=trades,
        params=effective_params,
    )
