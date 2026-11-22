"""
This file downloads OHLCV data for major cryptocurrencies from Binance's public
REST API.
"""

import time
import numpy as np
import pandas as pd
import requests

from pathlib import Path
from datetime import datetime, timedelta, timezone


YEARS = 5
DAYS = 729
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "Datasets" / "Raw_Datasets" / "Crypto"
BASE_URL = "https://api.binance.com/api/v3/klines"
LIMIT = 1000  

FEATURES = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "n_trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]

#Method to fetch symbols between two points of time
def fetch_symbol(symbol, interval, start_ms, end_ms):
    rows = []
    cursor = start_ms
    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": LIMIT,
        }
        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        new_rows = resp.json()
        if not new_rows:
            break
        rows.extend(new_rows)
        cursor = rows[-1][6] + 1  
        time.sleep(0.3)  
    return rows

#Method to convert from list of lists to pd Data frame
def convert_to_df(rows):
    df = pd.DataFrame(rows, columns=FEATURES)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(np.float64)
    df = df[["open_time", "open", "high", "low", "close", "volume"]]
    return df.drop_duplicates("open_time").sort_values("open_time")



def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    curr = datetime.now(timezone.utc)
    end_ms = int(curr.timestamp() * 1000)
    daily_start_ms = int((curr - timedelta(days=365 * YEARS)).timestamp() * 1000)
    hourly_start_ms = int((curr - timedelta(days=DAYS)).timestamp() * 1000)

    for symbol in SYMBOLS:
        print(f"Started etching {symbol}")

        #Fetching daily data
        print("Daily data:")
        d_rows = fetch_symbol(symbol, "1d", daily_start_ms, end_ms)
        if d_rows:
            df_d = convert_to_df(d_rows)
            path = OUTPUT_DIR / f"{symbol}_1d.csv"
            df_d.to_csv(path, index=False)
            print(f"Successfully saved at {path}")

        #Fetching hourly data
        print("Hourly data:")
        h_rows = fetch_symbol(symbol, "1h", hourly_start_ms, end_ms)
        if h_rows:
            df_h = convert_to_df(h_rows)
            path = OUTPUT_DIR / f"{symbol}_1h.csv"
            df_h.to_csv(path, index=False)
            print(f"Successfully saved at {path}")


if __name__ == "__main__":
    main()