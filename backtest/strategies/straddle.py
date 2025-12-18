from __future__ import annotations

from typing import TYPE_CHECKING, List

import pandas as pd

from backtest.broker import Order
from common.schema import RiskConfig, SignalConfig

if TYPE_CHECKING:  # pragma: no cover - avoid runtime import cycle
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
    calls = data[data["type"] == "C"].set_index("strike")
    puts = data[data["type"] == "P"].set_index("strike")
    common_strikes = calls.index.intersection(puts.index)
    pairs = []
    for strike in common_strikes:
        call_row = calls.loc[strike]
        put_row = puts.loc[strike]
        if isinstance(call_row, pd.DataFrame) or isinstance(put_row, pd.DataFrame):
            continue
        score = (abs(call_row["score"]) + abs(put_row["score"])) / 2
        pairs.append((score, call_row, put_row))
    pairs.sort(key=lambda item: item[0], reverse=True)
    orders: List[Order] = []
    side = signal_cfg.side
    for score, call_row, put_row in pairs[: signal_cfg.select_top_k]:
        for row in (call_row, put_row):
            order = Order(
                option_key=row["option_key"],
                symbol=row["symbol"],
                option_type=row["type"],
                strike=float(row.name),
                expiry=pd.to_datetime(row["expiry"]),
                trade_date=pd.to_datetime(date),
                side=side,
                qty=1,
                mid_price=float(row["mid"]),
            )
            if portfolio.can_open(order, risk_cfg):
                orders.append(order)
    return orders
