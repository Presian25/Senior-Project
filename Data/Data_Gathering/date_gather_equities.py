"""
Downloads equity data for 5 large-cap stocks, from diverse sectors, from Yahoo Finance.
Data span: 2 years hourly + full 5 years daily.
"""
import time

from pathlib import Path
from gather_config import fetch_hourly, fetch_daily

STOCKS = ["AAPL", "JPM", "XOM", "JNJ", "NVDA"]  # tech, financials, energy, healthcare, semiconductors
OUTPUT_DIR = Path(__file__).resolve().parent.parent/"Datasets"/"Raw_Datasets"/"Equities"


def main():
    for asset in STOCKS:
        print(f"Started fetching {asset}")
        fetch_hourly(asset, OUTPUT_DIR)
        fetch_daily(asset, OUTPUT_DIR)
        time.sleep(1)


if __name__ == "__main__":
    main()