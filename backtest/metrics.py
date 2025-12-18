from __future__ import annotations

import numpy as np
import pandas as pd


def compute_equity_metrics(equity: pd.DataFrame) -> dict[str, float]:
    if equity.empty:
        return {}
    equity = equity.sort_values("date")
    returns = equity["nav"].pct_change().dropna()
    if returns.empty:
        return {}
    mean_return = returns.mean()
    vol = returns.std(ddof=0)
    sharpe = (mean_return / vol) * np.sqrt(252) if vol > 0 else 0.0
    downside = returns[returns < 0]
    downside_std = downside.std(ddof=0)
    sortino = (mean_return / downside_std) * np.sqrt(252) if downside_std > 0 else 0.0
    start_nav = equity["nav"].iloc[0]
    end_nav = equity["nav"].iloc[-1]
    periods = len(returns)
    cagr = (end_nav / start_nav) ** (252 / periods) - 1 if start_nav > 0 and periods > 0 else 0.0
    drawdown = _max_drawdown(equity["nav"].values)
    calmar = cagr / abs(drawdown) if drawdown != 0 else 0.0
    return {
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "volatility": float(vol * np.sqrt(252)),
        "cagr": float(cagr),
        "max_drawdown": float(drawdown),
        "calmar": float(calmar),
    }


def compute_trade_metrics(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {}
    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]
    win_rate = len(wins) / len(trades) if len(trades) else 0.0
    avg_win = wins["pnl"].mean() if not wins.empty else 0.0
    avg_loss = losses["pnl"].mean() if not losses.empty else 0.0
    payoff = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan
    avg_hold = trades["holding_days"].mean() if "holding_days" in trades else np.nan
    return {
        "win_rate": float(win_rate),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "payoff_ratio": float(payoff) if not np.isnan(payoff) else 0.0,
        "avg_holding_days": float(avg_hold) if not np.isnan(avg_hold) else 0.0,
    }


def summarize(trades: pd.DataFrame, equity: pd.DataFrame) -> dict[str, float]:
    metrics = {}
    metrics.update(compute_equity_metrics(equity))
    metrics.update(compute_trade_metrics(trades))
    return metrics


def _max_drawdown(values: np.ndarray) -> float:
    peak = -np.inf
    max_dd = 0.0
    for value in values:
        peak = max(peak, value)
        drawdown = (value - peak) / peak if peak > 0 else 0.0
        max_dd = min(max_dd, drawdown)
    return float(max_dd)
