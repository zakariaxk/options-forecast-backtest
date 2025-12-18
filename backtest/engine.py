from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import pandas as pd

from backtest import metrics
from backtest.broker import CONTRACT_MULTIPLIER, Fill, Order, simulate_fills
from backtest.strategies import credit_spread, covered_call, straddle
from common.io import read_parquet, write_json, write_parquet
from common.schema import BacktestConfig, ExecutionConfig, RiskConfig, SignalConfig


@dataclass
class Position:
    option_key: str
    symbol: str
    option_type: str
    strike: float
    expiry: pd.Timestamp
    qty: int
    direction: int  # 1 for long, -1 for short
    entry_price: float
    entry_date: pd.Timestamp
    fees: float = 0.0

    def market_value(self, price: float) -> float:
        return self.direction * self.qty * price * CONTRACT_MULTIPLIER

    def pnl_value(self, price: float) -> float:
        return (price - self.entry_price) * self.direction * self.qty * CONTRACT_MULTIPLIER

    def pnl_pct(self, price: float) -> float:
        if self.entry_price == 0:
            return 0.0
        return self.direction * (price - self.entry_price) / self.entry_price


@dataclass
class TradeRecord:
    option_key: str
    symbol: str
    direction: str
    qty: int
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    pnl: float
    fees: float
    holding_days: int


class Portfolio:
    def __init__(self, initial_cash: float) -> None:
        self.cash: float = initial_cash
        self.positions: Dict[str, Position] = {}
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[dict] = []

    def can_open(self, order: Order, risk_cfg: RiskConfig) -> bool:
        existing = self.positions.get(order.option_key)
        direction = 1 if order.side == "buy" else -1
        if existing:
            if existing.direction == direction:
                # already have exposure in same direction, keep it simple
                return False
            # closing allowed
            return True
        if order.qty > risk_cfg.max_position_per_option:
            return False
        notional = abs(order.mid_price) * order.qty * CONTRACT_MULTIPLIER
        if self.gross_notional() + notional > risk_cfg.max_gross_notional:
            return False
        return True

    def gross_notional(self) -> float:
        total = 0.0
        for position in self.positions.values():
            total += abs(position.entry_price) * position.qty * CONTRACT_MULTIPLIER
        return total

    def apply_fill(self, fill: Fill) -> None:
        self.cash += fill.cash_flow()
        direction = 1 if fill.order.side == "buy" else -1
        existing = self.positions.get(fill.order.option_key)
        if existing and existing.direction != direction:
            self._close_position(existing, fill)
            self.positions.pop(existing.option_key, None)
        else:
            self.positions[fill.order.option_key] = Position(
                option_key=fill.order.option_key,
                symbol=fill.order.symbol,
                option_type=fill.order.option_type,
                strike=fill.order.strike,
                expiry=fill.order.expiry,
                qty=fill.order.qty,
                direction=direction,
                entry_price=fill.price,
                entry_date=fill.order.trade_date,
                fees=fill.fees,
            )

    def _close_position(self, position: Position, fill: Fill) -> None:
        exit_price = fill.price
        total_fees = position.fees + fill.fees
        pnl = position.pnl_value(exit_price) - total_fees
        holding = max((fill.order.trade_date - position.entry_date).days, 0)
        trade = TradeRecord(
            option_key=position.option_key,
            symbol=position.symbol,
            direction="long" if position.direction == 1 else "short",
            qty=position.qty,
            entry_date=position.entry_date,
            exit_date=fill.order.trade_date,
            entry_price=position.entry_price,
            exit_price=exit_price,
            pnl=pnl,
            fees=total_fees,
            holding_days=holding,
        )
        self.trades.append(trade)

    def record_nav(self, date: pd.Timestamp, price_map: pd.Series) -> None:
        nav = self.cash
        for position in self.positions.values():
            price = float(price_map.get(position.option_key, position.entry_price))
            nav += position.market_value(price)
        record_date = pd.to_datetime(date)
        if self.equity_curve and self.equity_curve[-1]["date"] == record_date:
            self.equity_curve[-1]["nav"] = nav
        else:
            self.equity_curve.append({"date": record_date, "nav": nav})

    def liquidation_orders(self, date: pd.Timestamp, price_map: Dict[str, float]) -> List[Order]:
        orders: List[Order] = []
        for position in list(self.positions.values()):
            price = price_map.get(position.option_key, position.entry_price)
            side = "sell" if position.direction == 1 else "buy"
            orders.append(
                Order(
                    option_key=position.option_key,
                    symbol=position.symbol,
                    option_type=position.option_type,
                    strike=position.strike,
                    expiry=position.expiry,
                    trade_date=date,
                    side=side,
                    qty=position.qty,
                    mid_price=float(price),
                )
            )
        return orders


StrategyFn = Callable[[pd.Timestamp, pd.DataFrame, SignalConfig, RiskConfig, Portfolio], List[Order]]

STRATEGIES: Dict[str, StrategyFn] = {
    "straddle": straddle.generate_orders,
    "straddle_buy": straddle.generate_orders,
    "credit_spread": credit_spread.generate_orders,
    "credit_spread_sell": credit_spread.generate_orders,
    "covered_call": covered_call.generate_orders,
}


@dataclass
class BacktestResult:
    metrics: Dict[str, float]
    trades: pd.DataFrame
    equity: pd.DataFrame
    config: BacktestConfig

    def save(self, base_uri: str) -> dict[str, str]:
        trades_uri = f"{base_uri}/trades.parquet"
        equity_uri = f"{base_uri}/equity.parquet"
        metrics_uri = f"{base_uri}/metrics.json"
        write_parquet(self.trades, trades_uri)
        write_parquet(self.equity, equity_uri)
        write_json(self.metrics, metrics_uri)
        return {
            "trades_uri": trades_uri,
            "equity_uri": equity_uri,
            "metrics_uri": metrics_uri,
        }


def load_predictions(config: BacktestConfig) -> pd.DataFrame:
    data_cfg = config.data
    if "predictions_uri" in data_cfg:
        uri = data_cfg["predictions_uri"]
    else:
        model_name = data_cfg.get("model_name") or config.signal.side
        run_id = data_cfg.get("prediction_run")
        base = data_cfg.get("predictions_base", "data/predictions")
        if not run_id:
            raise ValueError("prediction_run is required in config.data")
        uri = f"{base}/{config.symbol}/{model_name}/{run_id}/predictions.parquet"
    df = read_parquet(uri)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["expiry"] = pd.to_datetime(df["expiry"])
    return df


def run_backtest(config: BacktestConfig, predictions: Optional[pd.DataFrame] = None) -> BacktestResult:
    predictions = predictions or load_predictions(config)
    mask = (predictions["trade_date"] >= pd.to_datetime(config.start_date)) & (
        predictions["trade_date"] <= pd.to_datetime(config.end_date)
    )
    df = predictions.loc[mask].copy()
    df = df.sort_values("trade_date")
    strategy_key = config.strategy or config.name
    strategy_fn = STRATEGIES.get(strategy_key)
    if strategy_fn is None:
        for key, fn in STRATEGIES.items():
            if strategy_key and key in strategy_key:
                strategy_fn = fn
                break
    if strategy_fn is None:
        raise ValueError(f"Unknown strategy {strategy_key}")
    portfolio = Portfolio(initial_cash=config.risk.max_gross_notional)
    for date, day_data in df.groupby("trade_date"):
        day_data = _apply_universe(day_data, config)
        if day_data.empty:
            portfolio.record_nav(date, pd.Series(dtype=float))
            continue
        price_map = day_data.set_index("option_key")["mid"]
        orders = strategy_fn(date, day_data, config.signal, config.risk, portfolio)
        fills = simulate_fills(orders, config.execution)
        for fill in fills:
            portfolio.apply_fill(fill)
        exit_orders = _risk_exits(date, day_data, portfolio, config)
        exit_fills = simulate_fills(exit_orders, config.execution)
        for fill in exit_fills:
            portfolio.apply_fill(fill)
        portfolio.record_nav(date, price_map)
    if portfolio.positions:
        last_date = df["trade_date"].max()
        last_prices = df[df["trade_date"] == last_date].set_index("option_key")["mid"].to_dict()
        liquidation_orders = portfolio.liquidation_orders(last_date, last_prices)
        fills = simulate_fills(liquidation_orders, config.execution)
        for fill in fills:
            portfolio.apply_fill(fill)
        portfolio.record_nav(last_date, pd.Series(last_prices))
    trade_records = [trade.__dict__ for trade in portfolio.trades]
    if trade_records:
        trades_df = pd.DataFrame(trade_records)
    else:
        trades_df = pd.DataFrame(
            columns=[
                "option_key",
                "symbol",
                "direction",
                "qty",
                "entry_date",
                "exit_date",
                "entry_price",
                "exit_price",
                "pnl",
                "fees",
                "holding_days",
            ]
        )
    if portfolio.equity_curve:
        equity_df = pd.DataFrame(portfolio.equity_curve)
    else:
        equity_df = pd.DataFrame(columns=["date", "nav"])
    metric_summary = metrics.summarize(trades_df, equity_df)
    return BacktestResult(metrics=metric_summary, trades=trades_df, equity=equity_df, config=config)


def _apply_universe(day_data: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    uni = config.universe
    data = day_data.copy()
    data = data[(data["expiry_dte"] >= uni.dte_min) & (data["expiry_dte"] <= uni.dte_max)]
    data = data[data["type"].isin(uni.type)]
    data = data[(data["moneyness"] >= uni.moneyness_min) & (data["moneyness"] <= uni.moneyness_max)]
    data = data[(data["oi"] >= uni.min_oi) & (data["spread_pct"] <= uni.max_spread_pct)]
    return data


def _risk_exits(
    date: pd.Timestamp,
    day_data: pd.DataFrame,
    portfolio: Portfolio,
    config: BacktestConfig,
) -> List[Order]:
    exits: List[Order] = []
    price_lookup = day_data.set_index("option_key")
    for position in list(portfolio.positions.values()):
        if position.option_key not in price_lookup.index:
            continue
        row = price_lookup.loc[position.option_key]
        current_price = float(row["mid"])
        pnl_pct = position.pnl_pct(current_price)
        days_held = (pd.to_datetime(date) - position.entry_date).days
        should_exit = False
        exit_price = current_price
        if pnl_pct <= -config.risk.stop_loss_pct or pnl_pct >= config.risk.take_profit_pct:
            should_exit = True
        elif days_held >= config.rebalance.exit_after_days:
            should_exit = True
        elif position.expiry <= pd.to_datetime(date):
            underlying = float(row.get("close", current_price))
            if position.option_type == "C":
                exit_price = max(underlying - position.strike, 0.0)
            else:
                exit_price = max(position.strike - underlying, 0.0)
            should_exit = True
        if should_exit:
            side = "sell" if position.direction == 1 else "buy"
            exits.append(
                Order(
                    option_key=position.option_key,
                    symbol=position.symbol,
                    option_type=position.option_type,
                    strike=position.strike,
                    expiry=position.expiry,
                    trade_date=pd.to_datetime(date),
                    side=side,
                    qty=position.qty,
                    mid_price=max(exit_price, 0.01),
                )
            )
    return exits
