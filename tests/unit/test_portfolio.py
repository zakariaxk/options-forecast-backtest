from datetime import datetime

from backtest.broker import Order, simulate_fills
from backtest.engine import Portfolio
from common.schema import ExecutionConfig, RiskConfig


def make_order(side: str = "buy", trade_date: datetime | None = None) -> Order:
    return Order(
        option_key="AAPL_2023-01-20_150_C",
        symbol="AAPL",
        option_type="C",
        strike=150.0,
        expiry=datetime(2023, 1, 20),
        trade_date=trade_date or datetime(2023, 1, 1),
        side=side,
        qty=1,
        mid_price=2.5,
    )


def test_portfolio_can_open_and_close():
    portfolio = Portfolio(initial_cash=100000)
    risk = RiskConfig(
        max_gross_notional=200000,
        max_position_per_option=5,
        stop_loss_pct=0.2,
        take_profit_pct=0.3,
    )
    exec_cfg = ExecutionConfig()
    order = make_order("buy")
    assert portfolio.can_open(order, risk)
    fill = simulate_fills([order], exec_cfg)[0]
    portfolio.apply_fill(fill)
    assert order.option_key in portfolio.positions
    close_order = make_order("sell", trade_date=datetime(2023, 1, 2))
    close_fill = simulate_fills([close_order], exec_cfg)[0]
    portfolio.apply_fill(close_fill)
    assert order.option_key not in portfolio.positions
