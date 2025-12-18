from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import Iterable, Optional

from common.io import write_json, write_parquet
from common.schema import IngestConfig
from pipelines.ingest_yf import _black_scholes_greeks, IngestionResult

def generate_geometric_brownian_motion(
    n_days: int,
    start_price: float,
    mu: float,
    sigma: float,
    seed: int = 42
) -> np.ndarray:
    np.random.seed(seed)
    dt = 1/252
    drift = (mu - 0.5 * sigma**2) * dt
    shock = sigma * np.sqrt(dt) * np.random.normal(size=n_days)
    returns = drift + shock
    price_path = start_price * np.exp(np.cumsum(returns))
    return price_path

def generate_synthetic_data(config: IngestConfig) -> IngestionResult:
    # 1. Generate Equity Data
    dates = pd.date_range(start=config.start_date, end=config.end_date, freq="B")
    n_days = len(dates)
    
    prices = generate_geometric_brownian_motion(
        n_days=n_days,
        start_price=150.0,
        mu=0.10,  # 10% annual drift
        sigma=0.20, # 20% annual vol
        seed=config.seed
    )
    
    equity_df = pd.DataFrame({
        "ts": dates,
        "symbol": config.symbol,
        "open": prices,
        "high": prices * 1.01,
        "low": prices * 0.99,
        "close": prices,
        "adj_close": prices,
        "volume": 1000000
    })
    equity_df["ts"] = pd.to_datetime(equity_df["ts"], utc=True)

    # 2. Generate Options Data
    # For each day, generate a chain of options
    option_records = []
    
    # Fixed strikes relative to initial price to keep it simple, or dynamic?
    # Let's make strikes dynamic around the spot price
    
    for day_idx, row in equity_df.iterrows():
        spot = row["close"]
        trade_date = row["ts"]
        
        # Generate expiries: 1 week, 1 month, 3 months out
        expiries = [
            trade_date + timedelta(days=7),
            trade_date + timedelta(days=30),
            trade_date + timedelta(days=90)
        ]
        
        for expiry in expiries:
            dte = (expiry - trade_date).days
            if dte <= 0: continue
            
            # Strikes: ATM, +/- 5%, +/- 10%
            strikes = [spot * m for m in [0.9, 0.95, 1.0, 1.05, 1.1]]
            strikes = [round(s, 1) for s in strikes]
            
            for strike in strikes:
                for opt_type in ["C", "P"]:
                    # Randomize IV slightly
                    iv = 0.20 + np.random.normal(0, 0.02)
                    iv = max(0.1, iv)
                    
                    # Calculate Greeks & Price
                    delta, gamma, theta, vega = _black_scholes_greeks(
                        opt_type, spot, strike, iv, dte/365.0
                    )
                    
                    # BS Price approximation (simplified)
                    # We don't have a full pricer in ingest_yf, but we can approximate or use the greeks?
                    # Actually let's just use a simple BS pricer here for 'mid'
                    
                    d1 = (np.log(spot / strike) + (0.01 + 0.5 * iv ** 2) * (dte/365.0)) / (iv * np.sqrt(dte/365.0))
                    d2 = d1 - iv * np.sqrt(dte/365.0)
                    
                    from math import erf, exp, sqrt, log
                    def cdf(x): return 0.5 * (1 + erf(x / sqrt(2)))
                    
                    if opt_type == "C":
                        price = spot * cdf(d1) - strike * exp(-0.01 * dte/365.0) * cdf(d2)
                    else:
                        price = strike * exp(-0.01 * dte/365.0) * cdf(-d2) - spot * cdf(-d1)
                        
                    mid = max(0.01, price)
                    spread = mid * 0.02 # 2% spread
                    bid = mid - spread/2
                    ask = mid + spread/2
                    
                    option_records.append({
                        "ts": trade_date,
                        "symbol": config.symbol,
                        "expiry": expiry,
                        "type": opt_type,
                        "strike": strike,
                        "bid": bid,
                        "ask": ask,
                        "mid": mid,
                        "iv": iv,
                        "delta": delta,
                        "gamma": gamma,
                        "theta": theta,
                        "vega": vega,
                        "oi": np.random.randint(100, 1000),
                        "volume": np.random.randint(10, 100)
                    })

    options_df = pd.DataFrame(option_records)
    
    # Save
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
        "source": "synthetic"
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
    parser = argparse.ArgumentParser(description="Ingest synthetic data.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--dest-uri", default="data/raw")
    parsed = parser.parse_args(args=args)
    return IngestConfig(
        symbol=parsed.symbol,
        start_date=date.fromisoformat(parsed.start_date),
        end_date=date.fromisoformat(parsed.end_date),
        dest_uri=parsed.dest_uri,
    )

def main(argv: Optional[Iterable[str]] = None) -> IngestionResult:
    config = _parse_args(argv)
    return generate_synthetic_data(config)

if __name__ == "__main__":
    main()
