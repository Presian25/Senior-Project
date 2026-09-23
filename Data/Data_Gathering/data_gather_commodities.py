"""
Downloads commodity futures data for 5 diversified contracts from Yahoo Finance.
Data span: 2 years hourly + full 5 years daily.
The observed assets are precious metals, energy and industrial metals so the sample is diverse.
"""
import time

from pathlib import Path
from gather_config import fetch_hourly, fetch_daily

COMMODITIES = ["GC=F", "CL=F", "SI=F", "NG=F", "HG=F"]  # gold, WTI crude, silver, natural gas, copper
OUTPUT_DIR = Path(__file__).resolve().parent.parent/"Datasets"/"Raw_Datasets"/"Commodities"


def main():
    for asset in COMMODITIES:
        print(f"Started fetching {asset}")
        fetch_hourly(asset, OUTPUT_DIR)
        fetch_daily(asset, OUTPUT_DIR)
        time.sleep(1)


if __name__ == "__main__":
    main()