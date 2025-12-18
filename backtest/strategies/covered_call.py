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
    calls = data[data["type"] == "C"].copy()
    # Prefer near-the-money calls with elevated moneyness but manageable spread
    calls = calls[(calls["moneyness"] > -0.05) & (calls["moneyness"] < 0.1)]
    calls = calls.sort_values("spread_pct")
    orders: List[Order] = []
    for _, row in calls.head(signal_cfg.select_top_k).iterrows():
        order = Order(
            option_key=row["option_key"],
            symbol=row["symbol"],
            option_type=row["type"],
            strike=float(row["strike"]),
            expiry=pd.to_datetime(row["expiry"]),
            trade_date=pd.to_datetime(date),
            side="sell",
            qty=1,
            mid_price=float(row["mid"]),
        )
        if portfolio.can_open(order, risk_cfg):
            orders.append(order)
    return orders
