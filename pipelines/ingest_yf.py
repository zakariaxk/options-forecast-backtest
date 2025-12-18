from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, timedelta, timezone
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from math import erf

from common.io import write_json, write_parquet
from common.schema import IngestConfig

UTC = timezone.utc


@dataclass
class IngestionResult:
    symbol: str
    partition: str
    equity_uri: str
    options_uri: str
    metadata_uri: str


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def _black_scholes_greeks(
    side: str,
    spot: float,
    strike: float,
    iv: float,
    dte: float,
    rate: float = 0.01,
) -> tuple[float, float, float, float]:
    if spot <= 0 or strike <= 0 or iv <= 0 or dte <= 0:
        return (np.nan, np.nan, np.nan, np.nan)
    vol = iv
    sqrt_t = np.sqrt(dte)
    d1 = (np.log(spot / strike) + (rate + 0.5 * vol * vol) * dte) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    if side == "C":
        delta = _norm_cdf(d1)
        theta = (
            -(spot * vol * np.exp(-0.5 * d1 * d1) / (np.sqrt(2 * np.pi) * 2 * sqrt_t))
            - rate * strike * np.exp(-rate * dte) * _norm_cdf(d2)
        )
    else:
        delta = _norm_cdf(d1) - 1
        theta = (
            -(spot * vol * np.exp(-0.5 * d1 * d1) / (np.sqrt(2 * np.pi) * 2 * sqrt_t))
            + rate * strike * np.exp(-rate * dte) * _norm_cdf(-d2)
        )
    gamma = np.exp(-0.5 * d1 * d1) / (spot * vol * sqrt_t * np.sqrt(2 * np.pi))
    vega = spot * np.exp(-0.5 * d1 * d1) * sqrt_t / np.sqrt(2 * np.pi)
    return (delta, gamma, theta / 365.0, vega / 100.0)


def _format_equity(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    ).reset_index()
    df["ts"] = pd.to_datetime(df["Date"], utc=True)
    df["symbol"] = symbol
    df = df[["ts", "symbol", "open", "high", "low", "close", "adj_close", "volume"]]
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def _format_options(
    frames: Iterable[pd.DataFrame],
    *,
    symbol: str,
    close_map: dict[date, float],
    min_oi: int,
    min_volume: int,
    max_contracts_per_expiry: Optional[int],
) -> pd.DataFrame:
    records = []
    for frame in frames:
        if frame.empty:
            continue
        frame = frame.copy()
        frame["ts"] = pd.to_datetime(frame["lastTradeDate"], utc=True)
        frame["iv"] = frame["impliedVolatility"].astype(float).clip(lower=1e-6)
        frame["mid"] = (frame["bid"] + frame["ask"]) / 2.0
        frame["oi"] = frame["openInterest"].fillna(0)
        frame["volume"] = frame["volume"].fillna(0)
        frame = frame[(frame["oi"] >= min_oi) & (frame["volume"] >= min_volume)]
        if frame.empty:
            continue
        frame["underlying"] = frame["ts"].dt.date.map(close_map)
        if max_contracts_per_expiry is not None and max_contracts_per_expiry > 0:
            frame["atm_dist"] = (frame["strike"] - frame["underlying"]).abs()
            frame = frame.sort_values("atm_dist").head(max_contracts_per_expiry).copy()
        frame["dte"] = (
            pd.to_datetime(frame["expiry"], utc=True) - frame["ts"]
        ).dt.days.clip(lower=1)
        greeks = frame.apply(
            lambda row: _black_scholes_greeks(
                row["type"],
                row["underlying"] if pd.notnull(row["underlying"]) else np.nan,
                row["strike"],
                row["iv"],
                row["dte"] / 365.0,
            ),
            axis=1,
            result_type="expand",
        )
        greeks.columns = ["delta", "gamma", "theta", "vega"]
        frame = pd.concat([frame, greeks], axis=1)
        frame["symbol"] = symbol
        frame = frame[
            [
                "ts",
                "symbol",
                "expiry",
                "type",
                "strike",
                "bid",
                "ask",
                "mid",
                "iv",
                "delta",
                "gamma",
                "theta",
                "vega",
                "oi",
                "volume",
            ]
        ]
        if "atm_dist" in frame.columns:
            frame = frame.drop(columns=["atm_dist"])
        records.append(frame)
    if not records:
        return pd.DataFrame(
            columns=[
                "ts",
                "symbol",
                "expiry",
                "type",
                "strike",
                "bid",
                "ask",
                "mid",
                "iv",
                "delta",
                "gamma",
                "theta",
                "vega",
                "oi",
                "volume",
            ]
        )
    df = pd.concat(records, ignore_index=True)
    df = df.sort_values(["ts", "expiry", "strike", "type"]).reset_index(drop=True)
    return df


def _build_options_history(
    *,
    symbol: str,
    start: date,
    end: date,
    ticker: yf.Ticker,
    close_map: dict[date, float],
    min_oi: int,
    min_volume: int,
    max_expiries: Optional[int],
    max_contracts_per_expiry: Optional[int],
) -> pd.DataFrame:
    expiry_list = ticker.options or []
    if max_expiries is not None and max_expiries > 0:
        expiry_list = expiry_list[: max_expiries]

    if not close_map:
        return pd.DataFrame()
    spot = float(list(close_map.values())[-1])
    contracts: list[dict] = []
    for expiry in expiry_list:
        option_chain = ticker.option_chain(expiry)
        for frame, opt_type in ((option_chain.calls, "C"), (option_chain.puts, "P")):
            if frame.empty:
                continue
            tmp = frame.copy()
            tmp["type"] = opt_type
            tmp["expiry"] = expiry
            tmp["iv"] = tmp["impliedVolatility"].astype(float).clip(lower=1e-6)
            tmp["oi"] = tmp["openInterest"].fillna(0)
            tmp["volume"] = tmp["volume"].fillna(0)
            tmp = tmp[(tmp["oi"] >= min_oi) & (tmp["volume"] >= min_volume)]
            if tmp.empty:
                continue
            tmp["atm_dist"] = (tmp["strike"] - spot).abs()
            if max_contracts_per_expiry is not None and max_contracts_per_expiry > 0:
                tmp = tmp.sort_values("atm_dist").head(max_contracts_per_expiry)
            for row in tmp.itertuples():
                contracts.append(
                    {
                        "contract": getattr(row, "contractSymbol"),
                        "expiry": expiry,
                        "type": opt_type,
                        "strike": float(getattr(row, "strike")),
                        "iv": float(getattr(row, "iv")),
                        "oi": float(getattr(row, "oi")),
                    }
                )

    if not contracts:
        return pd.DataFrame(
            columns=[
                "ts",
                "symbol",
                "expiry",
                "type",
                "strike",
                "bid",
                "ask",
                "mid",
                "iv",
                "delta",
                "gamma",
                "theta",
                "vega",
                "oi",
                "volume",
            ]
        )

    end_plus = end + timedelta(days=1)
    all_frames: list[pd.DataFrame] = []
    for c in contracts:
        hist = yf.Ticker(c["contract"]).history(
            start=start.isoformat(),
            end=end_plus.isoformat(),
            auto_adjust=False,
        )
        if hist.empty:
            continue
        df = hist.rename(columns={"Close": "mid", "Volume": "volume"}).reset_index()
        df["ts"] = pd.to_datetime(df["Date"], utc=True)
        df["symbol"] = symbol
        df["expiry"] = c["expiry"]
        df["type"] = c["type"]
        df["strike"] = c["strike"]
        df["bid"] = df["mid"]
        df["ask"] = df["mid"]
        df["iv"] = c["iv"]
        df["oi"] = c["oi"]
        df["underlying"] = df["ts"].dt.date.map(close_map)
        df["dte"] = (pd.to_datetime(df["expiry"], utc=True) - df["ts"]).dt.days.clip(lower=1)
        greeks = df.apply(
            lambda row: _black_scholes_greeks(
                row["type"],
                float(row["underlying"]) if pd.notnull(row["underlying"]) else np.nan,
                float(row["strike"]),
                float(row["iv"]),
                float(row["dte"]) / 365.0,
            ),
            axis=1,
            result_type="expand",
        )
        greeks.columns = ["delta", "gamma", "theta", "vega"]
        df = pd.concat([df, greeks], axis=1)
        all_frames.append(
            df[
                [
                    "ts",
                    "symbol",
                    "expiry",
                    "type",
                    "strike",
                    "bid",
                    "ask",
                    "mid",
                    "iv",
                    "delta",
                    "gamma",
                    "theta",
                    "vega",
                    "oi",
                    "volume",
                ]
            ]
        )
    if not all_frames:
        return pd.DataFrame(
            columns=[
                "ts",
                "symbol",
                "expiry",
                "type",
                "strike",
                "bid",
                "ask",
                "mid",
                "iv",
                "delta",
                "gamma",
                "theta",
                "vega",
                "oi",
                "volume",
            ]
        )
    out = pd.concat(all_frames, ignore_index=True)
    out = out.dropna(subset=["mid"]).sort_values(["ts", "expiry", "strike", "type"]).reset_index(drop=True)
    return out


def ingest(config: IngestConfig) -> IngestionResult:
    ticker = yf.Ticker(config.symbol)
    end_plus = config.end_date + timedelta(days=1)
    history = ticker.history(
        start=config.start_date.isoformat(), end=end_plus.isoformat(), auto_adjust=False
    )
    if history.empty:
        raise ValueError(f"No equity history for {config.symbol}")
    equity_df = _format_equity(history, config.symbol)
    close_map = {row.ts.date(): float(row.close) for row in equity_df.itertuples()}
    options_df = _build_options_history(
        symbol=config.symbol,
        start=config.start_date,
        end=config.end_date,
        ticker=ticker,
        close_map=close_map,
        min_oi=config.min_open_interest,
        min_volume=config.min_volume,
        max_expiries=config.max_expiries,
        max_contracts_per_expiry=config.max_contracts_per_expiry,
    )
    partition = f"{config.symbol}/{config.start_date}_{config.end_date}"
    base_uri = f"{config.dest_uri}/{partition}"
    equity_uri = f"{base_uri}/equity.parquet"
    options_uri = f"{base_uri}/options.parquet"
    metadata_uri = f"{base_uri}/metadata.json"
    write_parquet(equity_df, equity_uri)
    write_parquet(options_df, options_uri)
    meta = {
        "symbol": config.symbol,
        "start_date": config.start_date.isoformat(),
        "end_date": config.end_date.isoformat(),
        "rows_equity": int(equity_df.shape[0]),
        "rows_options": int(options_df.shape[0]),
    }
    write_json(meta, metadata_uri)
    return IngestionResult(
        symbol=config.symbol,
        partition=partition,
        equity_uri=equity_uri,
        options_uri=options_uri,
        metadata_uri=metadata_uri,
    )


def _parse_args(args: Optional[Iterable[str]] = None) -> IngestConfig:
    parser = argparse.ArgumentParser(description="Ingest data from Yahoo Finance.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--dest-uri", default="data/raw")
    parser.add_argument("--min-oi", type=int, default=100)
    parser.add_argument("--min-volume", type=int, default=10)
    parser.add_argument("--max-expiries", type=int, default=None)
    parser.add_argument("--max-contracts-per-expiry", type=int, default=None)
    parsed = parser.parse_args(args=args)
    return IngestConfig(
        symbol=parsed.symbol,
        start_date=date.fromisoformat(parsed.start_date),
        end_date=date.fromisoformat(parsed.end_date),
        dest_uri=parsed.dest_uri,
        min_open_interest=parsed.min_oi,
        min_volume=parsed.min_volume,
        max_expiries=parsed.max_expiries,
        max_contracts_per_expiry=parsed.max_contracts_per_expiry,
    )


def main(argv: Optional[Iterable[str]] = None) -> IngestionResult:
    config = _parse_args(argv)
    return ingest(config)


if __name__ == "__main__":
    main()
