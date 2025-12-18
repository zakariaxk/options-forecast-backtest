from __future__ import annotations

import pandas as pd


def trades_table(trades: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "option_key",
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
    if trades.empty:
        return pd.DataFrame(columns=cols)
    df = trades.copy()
    for col in ["entry_date", "exit_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ["entry_price", "exit_price", "pnl", "fees"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in cols:
        if col not in df.columns:
            df[col] = pd.NA
    return df[cols].sort_values("exit_date", ascending=False, na_position="last")


def predictions_table(preds: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    if preds.empty:
        return pd.DataFrame()
    df = preds.copy()
    cols = ["trade_date", "option_key", "type", "strike", "score", "p_up", "mid"]
    for col in cols:
        if col not in df.columns:
            df[col] = pd.NA
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["p_up"] = pd.to_numeric(df["p_up"], errors="coerce")
    df["mid"] = pd.to_numeric(df["mid"], errors="coerce")
    df = df.sort_values("score", ascending=False, na_position="last")
    return df[cols].head(top_k)


def metrics_table(metrics: dict) -> pd.DataFrame:
    if not metrics:
        return pd.DataFrame(columns=["metric", "value"])
    rows = [{"metric": k, "value": v} for k, v in sorted(metrics.items())]
    return pd.DataFrame(rows)
