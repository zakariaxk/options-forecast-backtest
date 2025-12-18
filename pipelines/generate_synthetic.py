import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from common.io import write_parquet, write_json
from common.schema import IngestConfig

def generate_synthetic_data(symbol: str, start_date: str, end_date: str, output_uri: str):
    print(f"Generating synthetic data for {symbol} from {start_date} to {end_date}...")
    
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    n_days = len(dates)
    
    # Generate Equity Data
    np.random.seed(42)
    price = 150.0
    prices = []
    for _ in range(n_days):
        change = np.random.normal(0, 0.02)
        price *= (1 + change)
        prices.append(price)
        
    equity_df = pd.DataFrame({
        'ts': dates,
        'symbol': symbol,
        'open': [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
        'high': [p * (1 + np.random.uniform(0, 0.01)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
        'close': prices,
        'adj_close': prices,
        'volume': np.random.randint(1000000, 5000000, size=n_days)
    })
    
    # Generate Options Data
    options_data = []
    for date, price in zip(dates, prices):
        # Generate expiries
        expiries = [date + timedelta(days=x) for x in [7, 30, 60, 90]]
        
        for expiry in expiries:
            # Generate strikes
            strikes = [price * (1 + x) for x in np.arange(-0.1, 0.11, 0.05)]
            
            for strike in strikes:
                for opt_type in ['C', 'P']:
                    dte = (expiry - date).days
                    moneyness = (price - strike) / strike
                    
                    # Simple dummy pricing logic
                    iv = 0.2 + np.random.uniform(-0.05, 0.05)
                    intrinsic = max(0, price - strike) if opt_type == 'C' else max(0, strike - price)
                    time_value = price * iv * np.sqrt(dte/365) * 0.4
                    mid_price = intrinsic + time_value
                    
                    options_data.append({
                        'ts': date,
                        'symbol': symbol,
                        'expiry': expiry,
                        'type': opt_type,
                        'strike': round(strike, 2),
                        'bid': mid_price * 0.95,
                        'ask': mid_price * 1.05,
                        'mid': mid_price,
                        'iv': iv,
                        'delta': 0.5 + (moneyness if opt_type == 'C' else -moneyness), # Dummy delta
                        'gamma': 0.05,
                        'theta': -0.05,
                        'vega': 0.1,
                        'oi': np.random.randint(100, 10000),
                        'volume': np.random.randint(10, 1000)
                    })
                    
    options_df = pd.DataFrame(options_data)
    
    # Save Data
    partition = f"{symbol}/{start_date}_{end_date}"
    base_uri = f"{output_uri}/{partition}"
    
    write_parquet(equity_df, f"{base_uri}/equity.parquet")
    write_parquet(options_df, f"{base_uri}/options.parquet")
    
    meta = {
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "rows_equity": len(equity_df),
        "rows_options": len(options_df),
        "is_synthetic": True
    }
    write_json(meta, f"{base_uri}/metadata.json")
    print(f"Synthetic data saved to {base_uri}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2023-03-31")
    parser.add_argument("--output-uri", default="data/raw")
    args = parser.parse_args()
    
    generate_synthetic_data(args.symbol, args.start_date, args.end_date, args.output_uri)
