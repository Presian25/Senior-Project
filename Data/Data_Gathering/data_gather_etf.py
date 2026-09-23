"""
Downloads ETF data for 5 diversified funds from Yahoo Finance. 
Data span: 2 years hourly + full 5 years daily.
The assets observed are broad equity index, a tech index, a commodity, a bond
fund and an emerging-markets fund.
"""
import time

from pathlib import Path
from gather_config import fetch_hourly, fetch_daily

ETFS = ["SPY", "QQQ", "GLD", "TLT", "EEM"]  # S&P 500, Nasdaq-100, gold, long-term Treasuries, emerging markets
OUTPUT_DIR = Path(__file__).resolve().parent.parent/"Datasets"/"Raw_Datasets"/"ETFs"


def main():
    for asset in ETFS:
        print(f"Started fetching {asset}")
        fetch_hourly(asset, OUTPUT_DIR)
        fetch_daily(asset, OUTPUT_DIR)
        time.sleep(1)


if __name__ == "__main__":
    main()