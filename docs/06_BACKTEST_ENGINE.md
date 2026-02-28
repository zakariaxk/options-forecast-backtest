# Backtest Engine

## Overview

The backtest engine (`backtest/underlying.py`) runs deterministic simulations over daily OHLCV price data for a single equity. All strategies are fully in or fully out — no partial positions or margin.

## Supported Strategies

### 1. Buy & Hold (`buy_and_hold`)

The simplest strategy: buy as many shares as `initial_cash` allows on day 1, hold until the end.

**Logic**: Buy `floor(initial_cash / close_price)` shares at close on day 1. Hold forever.

**Parameters**: None.

### 2. SMA Crossover (`sma_crossover`)

Buy when the fast SMA crosses above the slow SMA (golden cross). Sell when the fast SMA crosses below the slow SMA (death cross).

**Logic**:
1. Compute two simple moving averages over close prices
2. When fast SMA crosses above slow SMA → buy all-in
3. When fast SMA crosses below slow SMA → sell all
4. Stays flat until both SMAs have valid values (after `slow_period` bars)

**Parameters**:
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `fast_period` | int | 20 | Fast SMA lookback period |
| `slow_period` | int | 50 | Slow SMA lookback period |

**Constraints**: `fast_period` < `slow_period`. Needs at least `slow_period` bars of data.

### 3. RSI Mean Reversion (`rsi_mean_reversion`)

Buy when RSI drops below the oversold threshold (stock is cheap). Sell when RSI rises above the overbought threshold (stock is expensive).

**Logic**:
1. Compute Wilder's RSI over close prices
2. When RSI < oversold → buy all-in
3. When RSI > overbought → sell all
4. Stays flat until RSI has a valid value (after `rsi_period` bars)

**Parameters**:
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `rsi_period` | int | 14 | RSI lookback period |
| `oversold` | float | 30 | Buy trigger threshold |
| `overbought` | float | 70 | Sell trigger threshold |

**Constraints**: `0 < oversold < overbought < 100`. Needs at least `rsi_period + 1` bars.

## Technical Indicators

- **SMA**: Simple rolling mean over `period` bars. Returns `None` for indices < `period - 1`.
- **RSI**: Wilder's RSI using EMA smoothing. Seeded with simple average over first `period` bars, then smoothed with `(prev * (period-1) + current) / period`.

## Determinism

- Same symbol + same date range + same cached data + same params = same output
- No random number generators
- yfinance data is cached after first fetch

## Options Strategies (UNSUPPORTED)

The following strategies return a `409 Conflict` error because they require historical options quote data that is not available:
- `straddle`, `credit_spread`, `covered_call`

## Metrics

Computed from the equity curve:
- `total_return` — (final_nav - initial_nav) / initial_nav
- `cagr` — annualized compound return
- `sharpe` — annualized Sharpe ratio (risk-free rate = 0)
- `sortino` — annualized Sortino ratio
- `max_drawdown` — maximum peak-to-trough decline
- `volatility` — annualized standard deviation of daily returns
- `calmar` — CAGR / |max_drawdown|
