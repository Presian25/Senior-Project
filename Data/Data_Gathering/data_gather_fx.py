"""
Downloads FX data for 5 major currency pairs from Yahoo Finance. 
Data span: 2 years hourly + full 5 years daily. 
The pairs chosen are across USD, EUR, GBP, JPY, CHF and AUD exposure.
"""
import time

from pathlib import Path
from gather_config import fetch_hourly, fetch_daily

PAIRS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X"]
OUTPUT_DIR = Path(__file__).resolve().parent.parent/"Datasets"/"Raw_Datasets"/"FXs"


def main():
    for asset in PAIRS:
        print(f"Started fetching {asset}")
        fetch_hourly(asset, OUTPUT_DIR)
        fetch_daily(asset, OUTPUT_DIR)
        time.sleep(1)  

if __name__ == "__main__":
    main()