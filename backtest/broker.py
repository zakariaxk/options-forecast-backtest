from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from common.schema import ExecutionConfig

CONTRACT_MULTIPLIER = 100


@dataclass(frozen=True)
class Order:
    option_key: str
    symbol: str
    option_type: Literal["C", "P"]
    strike: float
    expiry: pd.Timestamp
    trade_date: pd.Timestamp
    side: Literal["buy", "sell"]
    qty: int
    mid_price: float

    def notional(self) -> float:
        return self.mid_price * self.qty * CONTRACT_MULTIPLIER


@dataclass(frozen=True)
class Fill:
    order: Order
    price: float
    fees: float

    def cash_flow(self) -> float:
        sign = -1 if self.order.side == "buy" else 1
        return sign * self.price * self.order.qty * CONTRACT_MULTIPLIER - self.fees


def apply_slippage(price: float, slippage_bps: int, side: str) -> float:
    adjust = price * (slippage_bps / 10_000)
    return price + adjust if side == "buy" else price - adjust


def simulate_fills(orders: list[Order], exec_cfg: ExecutionConfig) -> list[Fill]:
    fills: list[Fill] = []
    for order in orders:
        if order.qty <= 0 or order.mid_price <= 0:
            continue
        price = order.mid_price
        if exec_cfg.price == "mid":
            price = apply_slippage(order.mid_price, exec_cfg.slippage_bps, order.side)
        elif exec_cfg.price == "bid" and order.side == "sell":
            price = order.mid_price - abs(exec_cfg.slippage_bps) * order.mid_price / 10_000
        elif exec_cfg.price == "ask" and order.side == "buy":
            price = order.mid_price + abs(exec_cfg.slippage_bps) * order.mid_price / 10_000
        fees = exec_cfg.commission_per_contract * order.qty
        fills.append(Fill(order=order, price=price, fees=fees))
    return fills
