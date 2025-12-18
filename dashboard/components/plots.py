from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def equity_curve(equity: pd.DataFrame):
    if equity.empty:
        return px.line(title="Equity Curve")
    data = equity.sort_values("date").copy()
    data["date"] = pd.to_datetime(data["date"])
    fig = px.line(data, x="date", y="nav", title="Equity Curve")
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig


def drawdown_curve(equity: pd.DataFrame):
    if equity.empty:
        return px.area(title="Drawdown")
    data = equity.sort_values("date").copy()
    data["date"] = pd.to_datetime(data["date"])
    data["peak"] = data["nav"].cummax()
    data["drawdown"] = data["nav"] / data["peak"] - 1
    fig = px.area(data, x="date", y="drawdown", title="Drawdown")
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig


def daily_returns(equity: pd.DataFrame):
    if equity.empty:
        return px.bar(title="Daily Returns")
    data = equity.sort_values("date").copy()
    data["date"] = pd.to_datetime(data["date"])
    data["ret"] = data["nav"].pct_change().fillna(0.0)
    fig = px.bar(data, x="date", y="ret", title="Daily Returns")
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig


def rolling_sharpe(equity: pd.DataFrame, window: int = 20):
    if equity.empty:
        return px.line(title="Rolling Sharpe")
    data = equity.sort_values("date").copy()
    data["date"] = pd.to_datetime(data["date"])
    rets = data["nav"].pct_change()
    roll_mean = rets.rolling(window).mean()
    roll_std = rets.rolling(window).std()
    sharpe = (roll_mean / roll_std) * (252**0.5)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["date"], y=sharpe, mode="lines", name=f"Sharpe({window})"))
    fig.update_layout(title="Rolling Sharpe", margin=dict(l=20, r=20, t=40, b=20))
    return fig
