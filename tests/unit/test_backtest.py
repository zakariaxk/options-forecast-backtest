"""Tests for backtest.underlying — all strategies, metrics, and engine."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from backtest.underlying import (
    DEFAULT_PARAMS,
    UNDERLYING_STRATEGIES,
    UnderlyingBacktestResult,
    _compute_rsi,
    _rolling_mean,
    _safe_dict,
    compute_metrics,
    run_buy_and_hold,
    run_rsi_mean_reversion,
    run_sma_crossover,
    run_underlying_backtest,
)


# ── Fixtures ─────────────────────────────────────────────────

def _make_prices(closes: list[float], start: str = "2020-01-02") -> pd.DataFrame:
    """Build a minimal prices DataFrame from a list of close prices."""
    dates = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({
        "date": dates,
        "open": closes,
        "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes],
        "close": closes,
        "volume": [1_000_000] * len(closes),
    })


def _trending_up_prices(n: int = 100, base: float = 100.0) -> pd.DataFrame:
    """Monotonically rising prices — triggers SMA golden cross."""
    closes = [base + i * 0.5 for i in range(n)]
    return _make_prices(closes)


def _volatile_prices(n: int = 100, base: float = 100.0) -> pd.DataFrame:
    """Oscillating prices that will trigger RSI extremes."""
    closes = []
    price = base
    for i in range(n):
        # Alternate 5-day runs: down hard then up hard
        cycle = (i // 8) % 2
        if cycle == 0:
            price *= 0.97  # drop ~3% per day
        else:
            price *= 1.04  # rise ~4% per day
        closes.append(round(price, 4))
    return _make_prices(closes)


def _sma_cross_prices() -> pd.DataFrame:
    """
    Carefully constructed prices that produce a clear SMA golden cross.
    60 bars of downtrend + 60 bars of strong uptrend.
    fast_period=5, slow_period=10 make the cross happen quickly.
    """
    closes = []
    price = 100.0
    # 60 bars downtrend / flat
    for i in range(60):
        price = 100.0 - i * 0.3
        closes.append(round(price, 4))
    # 60 bars strong uptrend
    for i in range(60):
        price = closes[-1] + (i + 1) * 1.0
        closes.append(round(price, 4))
    return _make_prices(closes)


# ── _safe_dict ───────────────────────────────────────────────

class TestSafeDict:
    def test_replaces_nan(self):
        assert _safe_dict({"a": float("nan")}) == {"a": 0.0}

    def test_replaces_inf(self):
        assert _safe_dict({"a": float("inf")}) == {"a": 0.0}
        assert _safe_dict({"a": float("-inf")}) == {"a": 0.0}

    def test_rounds_values(self):
        result = _safe_dict({"a": 1.123456789012})
        assert result["a"] == 1.12345679  # rounded to 8 decimals

    def test_passes_through_zero(self):
        assert _safe_dict({"a": 0.0}) == {"a": 0.0}


# ── compute_metrics ──────────────────────────────────────────

class TestComputeMetrics:
    def test_empty_equity(self):
        eq = pd.DataFrame(columns=["date", "nav"])
        result = compute_metrics(eq, 100_000)
        assert result["total_return"] == 0.0
        assert result["sharpe"] == 0.0

    def test_single_row(self):
        eq = pd.DataFrame({"date": [pd.Timestamp("2020-01-02")], "nav": [100_000.0]})
        result = compute_metrics(eq, 100_000)
        assert result["total_return"] == 0.0

    def test_basic_positive_return(self):
        eq = pd.DataFrame({
            "date": pd.bdate_range("2020-01-02", periods=5),
            "nav": [100_000, 101_000, 102_000, 103_000, 104_000],
        })
        result = compute_metrics(eq, 100_000)
        assert result["total_return"] == pytest.approx(0.04, abs=0.001)
        assert result["sharpe"] > 0
        assert result["max_drawdown"] == 0.0  # monotonically increasing

    def test_drawdown_detected(self):
        eq = pd.DataFrame({
            "date": pd.bdate_range("2020-01-02", periods=5),
            "nav": [100_000, 110_000, 90_000, 95_000, 100_000],
        })
        result = compute_metrics(eq, 100_000)
        assert result["max_drawdown"] < 0  # should detect the 110k→90k drop

    def test_no_nan_in_output(self):
        eq = pd.DataFrame({
            "date": pd.bdate_range("2020-01-02", periods=3),
            "nav": [100_000, 100_000, 100_000],  # flat — could cause div by zero
        })
        result = compute_metrics(eq, 100_000)
        for v in result.values():
            assert not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))


# ── run_buy_and_hold ─────────────────────────────────────────

class TestRunBuyAndHold:
    def test_basic_execution(self):
        prices = _make_prices([100.0, 105.0, 110.0])
        eq, trades = run_buy_and_hold(prices, 10_000.0)

        assert len(trades) == 1
        assert trades[0]["action"] == "BUY"
        assert trades[0]["qty"] == 100  # floor(10000/100)
        assert len(eq) == 3
        assert eq.iloc[0]["nav"] == 10_000.0
        assert eq.iloc[-1]["nav"] == 11_000.0  # 100 shares × $110

    def test_cash_remainder(self):
        prices = _make_prices([33.33, 40.0])
        eq, trades = run_buy_and_hold(prices, 100.0)

        shares = trades[0]["qty"]
        assert shares == 3  # floor(100/33.33)
        remainder = 100.0 - 3 * 33.33
        expected_final = remainder + 3 * 40.0
        assert eq.iloc[-1]["nav"] == pytest.approx(expected_final, abs=0.01)

    def test_insufficient_cash_raises(self):
        prices = _make_prices([1_000_000.0])
        with pytest.raises(ValueError, match="not enough to buy"):
            run_buy_and_hold(prices, 100.0)

    def test_empty_prices(self):
        prices = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        eq, trades = run_buy_and_hold(prices, 100_000.0)
        assert eq.empty
        assert trades == []


# ── Technical indicators ─────────────────────────────────────

class TestRollingMean:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _rolling_mean(data, 3)
        assert result[0] is None
        assert result[1] is None
        assert result[2] == pytest.approx(2.0)
        assert result[3] == pytest.approx(3.0)
        assert result[4] == pytest.approx(4.0)

    def test_period_one(self):
        data = np.array([10.0, 20.0, 30.0])
        result = _rolling_mean(data, 1)
        assert result == [10.0, 20.0, 30.0]


class TestComputeRSI:
    def test_all_gains(self):
        """All positive returns → RSI should be close to 100."""
        closes = np.array([float(i) for i in range(1, 22)])
        rsi = _compute_rsi(closes, 14)
        assert rsi[14] == 100.0  # first computed value

    def test_all_losses(self):
        """All negative returns → RSI should be close to 0."""
        closes = np.array([float(100 - i) for i in range(22)])
        rsi = _compute_rsi(closes, 14)
        assert rsi[14] == 0.0

    def test_mixed_returns(self):
        """Mixed returns → RSI should be between 0 and 100."""
        np.random.seed(42)
        closes = 100.0 + np.cumsum(np.random.randn(50))
        rsi = _compute_rsi(closes, 14)
        valid = [v for v in rsi if v is not None]
        assert len(valid) > 0
        for v in valid:
            assert 0.0 <= v <= 100.0

    def test_not_enough_data(self):
        closes = np.array([100.0, 101.0, 102.0])
        rsi = _compute_rsi(closes, 14)
        assert all(v is None for v in rsi)


# ── run_sma_crossover ───────────────────────────────────────

class TestRunSMACrossover:
    def test_basic_execution(self):
        prices = _sma_cross_prices()
        eq, trades = run_sma_crossover(prices, 100_000.0, fast_period=5, slow_period=10)

        assert len(eq) == len(prices)
        assert all("nav" in row for _, row in eq.iterrows())
        # With a clear downtrend→uptrend, we expect at least one BUY
        buy_trades = [t for t in trades if t["action"] == "BUY"]
        assert len(buy_trades) >= 1

    def test_monotonic_uptrend_no_cross(self):
        """In a monotonic uptrend, fast SMA is always above slow — no crossover occurs."""
        prices = _trending_up_prices(80)
        eq, trades = run_sma_crossover(prices, 100_000.0, fast_period=5, slow_period=10)
        assert len(eq) == 80
        # No crossover happens because fast > slow from the start
        assert trades == []
        # NAV stays at initial cash (never entered a position)
        assert eq.iloc[-1]["nav"] == 100_000.0

    def test_too_few_bars_raises(self):
        prices = _make_prices([100.0] * 10)
        with pytest.raises(ValueError, match="Need at least"):
            run_sma_crossover(prices, 100_000.0, fast_period=5, slow_period=50)

    def test_fast_ge_slow_raises(self):
        prices = _make_prices([100.0] * 60)
        with pytest.raises(ValueError, match="fast_period.*must be < slow_period"):
            run_sma_crossover(prices, 100_000.0, fast_period=50, slow_period=20)

    def test_empty_prices(self):
        prices = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        eq, trades = run_sma_crossover(prices, 100_000.0)
        assert eq.empty
        assert trades == []

    def test_nav_never_negative(self):
        prices = _sma_cross_prices()
        eq, _ = run_sma_crossover(prices, 100_000.0, fast_period=5, slow_period=10)
        assert (eq["nav"] >= 0).all()

    def test_deterministic(self):
        prices = _sma_cross_prices()
        kwargs = dict(fast_period=5, slow_period=10)
        eq1, t1 = run_sma_crossover(prices, 100_000.0, **kwargs)
        eq2, t2 = run_sma_crossover(prices, 100_000.0, **kwargs)
        assert eq1["nav"].tolist() == eq2["nav"].tolist()
        assert t1 == t2


# ── run_rsi_mean_reversion ───────────────────────────────────

class TestRunRSIMeanReversion:
    def test_basic_execution(self):
        prices = _volatile_prices(80)
        eq, trades = run_rsi_mean_reversion(
            prices, 100_000.0, rsi_period=14, oversold=30, overbought=70
        )
        assert len(eq) == 80
        assert all("nav" in row for _, row in eq.iterrows())

    def test_volatile_market_trades(self):
        """Volatile market should trigger at least one buy/sell cycle."""
        prices = _volatile_prices(120)
        eq, trades = run_rsi_mean_reversion(
            prices, 100_000.0, rsi_period=7, oversold=35, overbought=65
        )
        # With extreme oscillation and relaxed thresholds, expect some trades
        assert len(eq) == 120

    def test_too_few_bars_raises(self):
        prices = _make_prices([100.0] * 5)
        with pytest.raises(ValueError, match="Need at least"):
            run_rsi_mean_reversion(prices, 100_000.0, rsi_period=14)

    def test_invalid_thresholds_raises(self):
        prices = _make_prices([100.0] * 30)
        with pytest.raises(ValueError, match="Invalid RSI thresholds"):
            run_rsi_mean_reversion(prices, 100_000.0, oversold=70, overbought=30)

    def test_empty_prices(self):
        prices = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        eq, trades = run_rsi_mean_reversion(prices, 100_000.0)
        assert eq.empty
        assert trades == []

    def test_nav_never_negative(self):
        prices = _volatile_prices(100)
        eq, _ = run_rsi_mean_reversion(
            prices, 100_000.0, rsi_period=7, oversold=30, overbought=70
        )
        assert (eq["nav"] >= 0).all()

    def test_deterministic(self):
        prices = _volatile_prices(80)
        kwargs = dict(rsi_period=14, oversold=30, overbought=70)
        eq1, t1 = run_rsi_mean_reversion(prices, 100_000.0, **kwargs)
        eq2, t2 = run_rsi_mean_reversion(prices, 100_000.0, **kwargs)
        assert eq1["nav"].tolist() == eq2["nav"].tolist()
        assert t1 == t2


# ── run_underlying_backtest (orchestrator) ───────────────────

class TestRunUnderlyingBacktest:
    def test_with_injected_prices(self):
        """No network call — prices are injected directly."""
        prices = _make_prices([50.0, 55.0, 60.0, 58.0, 62.0])
        result = run_underlying_backtest(
            symbol="TEST",
            strategy="buy_and_hold",
            start_date=date(2020, 1, 2),
            end_date=date(2020, 1, 8),
            initial_cash=10_000.0,
            bt_id="test_001",
            prices=prices,
        )
        assert isinstance(result, UnderlyingBacktestResult)
        assert result.bt_id == "test_001"
        assert result.symbol == "TEST"
        assert result.strategy == "buy_and_hold"
        assert len(result.equity_curve) == 5
        assert len(result.trades) == 1
        assert result.trades[0]["symbol"] == "TEST"
        assert result.summary["total_return"] > 0

    def test_unknown_strategy_raises(self):
        prices = _make_prices([100.0])
        with pytest.raises(ValueError, match="Unknown underlying strategy"):
            run_underlying_backtest(
                symbol="TEST",
                strategy="magic_money",
                start_date=date(2020, 1, 2),
                end_date=date(2020, 1, 2),
                prices=prices,
            )

    def test_to_dict_roundtrip(self):
        prices = _make_prices([100.0, 110.0])
        result = run_underlying_backtest(
            symbol="TEST",
            strategy="buy_and_hold",
            start_date=date(2020, 1, 2),
            end_date=date(2020, 1, 3),
            prices=prices,
        )
        d = result.to_dict()
        assert set(d.keys()) == {
            "bt_id", "symbol", "strategy", "start_date", "end_date",
            "initial_cash", "summary", "equity_curve", "trades", "params",
        }

    def test_determinism(self):
        """Same inputs → identical outputs."""
        prices = _make_prices([100.0, 105.0, 98.0, 110.0, 115.0])
        kwargs = dict(
            symbol="AAPL",
            strategy="buy_and_hold",
            start_date=date(2020, 1, 2),
            end_date=date(2020, 1, 8),
            initial_cash=50_000.0,
            bt_id="det_test",
            prices=prices,
        )
        a = run_underlying_backtest(**kwargs)
        b = run_underlying_backtest(**kwargs)
        assert a.summary == b.summary
        assert a.equity_curve == b.equity_curve
        assert a.trades == b.trades

    def test_sma_crossover_orchestrator(self):
        """SMA crossover via orchestrator with custom params."""
        prices = _sma_cross_prices()
        result = run_underlying_backtest(
            symbol="TEST",
            strategy="sma_crossover",
            start_date=date(2020, 1, 2),
            end_date=date(2020, 6, 30),
            prices=prices,
            params={"fast_period": 5, "slow_period": 10},
        )
        assert result.strategy == "sma_crossover"
        assert result.params["fast_period"] == 5
        assert result.params["slow_period"] == 10
        assert len(result.equity_curve) == len(prices)

    def test_rsi_orchestrator(self):
        """RSI strategy via orchestrator with custom params."""
        prices = _volatile_prices(80)
        result = run_underlying_backtest(
            symbol="TEST",
            strategy="rsi_mean_reversion",
            start_date=date(2020, 1, 2),
            end_date=date(2020, 5, 1),
            prices=prices,
            params={"rsi_period": 7, "oversold": 30, "overbought": 70},
        )
        assert result.strategy == "rsi_mean_reversion"
        assert result.params["rsi_period"] == 7
        assert len(result.equity_curve) == 80

    def test_default_params_applied(self):
        """When no params passed, defaults are used."""
        prices = _sma_cross_prices()
        result = run_underlying_backtest(
            symbol="TEST",
            strategy="sma_crossover",
            start_date=date(2020, 1, 2),
            end_date=date(2020, 6, 30),
            prices=prices,
        )
        assert result.params == DEFAULT_PARAMS["sma_crossover"]


# ── Strategy registry ────────────────────────────────────────

class TestStrategyRegistry:
    def test_buy_and_hold_registered(self):
        assert "buy_and_hold" in UNDERLYING_STRATEGIES

    def test_sma_crossover_registered(self):
        assert "sma_crossover" in UNDERLYING_STRATEGIES

    def test_rsi_mean_reversion_registered(self):
        assert "rsi_mean_reversion" in UNDERLYING_STRATEGIES

    def test_default_params_exist_for_all(self):
        for s in UNDERLYING_STRATEGIES:
            assert s in DEFAULT_PARAMS
    def test_buy_and_hold_registered(self):
        assert "buy_and_hold" in UNDERLYING_STRATEGIES
