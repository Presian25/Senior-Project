"""
Shared helpers all fetch scripts (except crypto) for pulling data from Yahoo Finance (via the `yfinance`
package) for FX, commodities, equities and ETFs.
Each script below pulls 2 versions of the data:
  1) fetch_hourly() - 2 years of hourly data
  2) fetch_daily()  - 5 years of daily data
"""
import os

import numpy as np
import pandas as pd
import yfinance as yf

YEARS = 5
DAYS = 729
MAX_DAYS = DAYS 

#Method to convert from raw pd Data Frame to OHLCV dataset format, consistent with crypto formatting
def convert(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index.name = "datetime"

    #Selecting the data
    entries = [var for var in ["Open", "High", "Low", "Close", "Volume"] if var in df.columns]
    df = df[entries]
    df.columns = [var.lower() for var in df.columns]
    return df.astype(np.float64)

#A cosmetic method, sometimes names contain symbols like =,^, so we ensure consistency
def fix_names(ticker):
    return ticker.replace("=", "_").replace("^", "")

#Method for fetching hourly data
def fetch_hourly(ticker, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df = yf.download(ticker, period=f"{DAYS}d", interval="1h",
                      auto_adjust=False, progress=False)
    df = convert(df)
    path = os.path.join(out_dir, f"{fix_names(ticker)}_1h.csv")
    df.to_csv(path)
    print(f"  {ticker}: {len(df)} hourly rows -> {path}")
    return df

#Method for fetching daily data
def fetch_daily(ticker, out_dir, years=YEARS):
    os.makedirs(out_dir, exist_ok=True)
    df = yf.download(ticker, period=f"{years}y", interval="1d",
                      auto_adjust=False, progress=False)
    df = convert(df)
    path = os.path.join(out_dir, f"{fix_names(ticker)}_1d.csv")
    df.to_csv(path)
    print(f"  {ticker}: {len(df)} daily rows -> {path}")
    return df