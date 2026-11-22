"""
Shared helpers all fetch scripts (except crypto) for pulling data from Yahoo Finance (via the `yfinance`
package) for FX, commodities, equities and ETFs.
Each script below pulls 2 versions of the data:

 - fetch_hourly() - 2 years of hourly data
 - fetch_daily()  - 5 years of daily data

Quick note: the method fix_names is needed as sometimes data is stored with different symbols - ^, = are common, and we want similar
style accross all datasets
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

YEARS = 5
DAYS = 729

#Method to convert from raw pd Data Frame to OHLCV dataset format
def convert_raw_to_ohlcv(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index.name = "datetime"

    entries = [var for var in ["Open", "High", "Low", "Close", "Volume"] if var in df.columns]
    df = df[entries]
    df.columns = [var.lower() for var in df.columns]
    return df.astype(np.float64)

#Method to ensure proper name convention
def fix_names(ticker):
    return ticker.replace("=", "_").replace("^", "")

def fetch_hourly(ticker, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = yf.download(ticker, period=f"{DAYS}d", interval="1h",
                      auto_adjust=False, progress=False)
    df = convert_raw_to_ohlcv(df)
    path = out_dir / f"{fix_names(ticker)}_1h.csv"
    df.to_csv(path)
    print(f"  {ticker}: {len(df)} hourly rows -> {path}")
    return df

def fetch_daily(ticker, out_dir, years=YEARS):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = yf.download(ticker, period=f"{years}y", interval="1d",
                      auto_adjust=False, progress=False)
    df = convert_raw_to_ohlcv(df)
    path = out_dir / f"{fix_names(ticker)}_1d.csv"
    df.to_csv(path)
    print(f"  {ticker}: {len(df)} daily rows -> {path}")
    return df