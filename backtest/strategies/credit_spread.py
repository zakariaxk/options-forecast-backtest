from __future__ import annotations

from typing import TYPE_CHECKING, List

import pandas as pd

from backtest.broker import Order
from common.schema import RiskConfig, SignalConfig

if TYPE_CHECKING:  # pragma: no cover
    from backtest.engine import Portfolio


def generate_orders(
    date: pd.Timestamp,
    data: pd.DataFrame,
    signal_cfg: SignalConfig,
    risk_cfg: RiskConfig,
    portfolio: "Portfolio",
) -> List[Order]:
    if data.empty:
        return []
    # Focus on call spreads when selling, put spreads when buying
    side = signal_cfg.side
    option_type = "C" if side == "sell" else "P"
    subset = (
        data[(data["type"] == option_type) & (data["score"] > 0)].copy()
        if side == "sell"
        else data[(data["type"] == option_type) & (data["score"] < 0)].copy()
    )
    subset = subset.sort_values("score", ascending=False if side == "sell" else True)
    orders: List[Order] = []
    for _, row in subset.head(signal_cfg.select_top_k * 2).iterrows():
        order = Order(
            option_key=row["option_key"],
            symbol=row["symbol"],
            option_type=row["type"],
            strike=float(row["strike"]),
            expiry=pd.to_datetime(row["expiry"]),
            trade_date=pd.to_datetime(date),
            side=side,
            qty=1,
            mid_price=float(row["mid"]),
        )
        if portfolio.can_open(order, risk_cfg):
            orders.append(order)
    return orders
