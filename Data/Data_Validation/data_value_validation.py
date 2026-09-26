"""
The purpose of this script is to validate the date as appropriate 
to be used for later research. For this purpose, the script tests
each dataset on 5 things:

1) All prices > 0 (for all open, close, high, low)
2) Volume >= 0
3) No time stamp has higher price than "high" and lower than "low"
4) Returns between timestamps are not beyond 50 %
5) No price movement with 0 volume 
"""

import pandas as pd
from pathlib import Path

INPUT = Path(__file__).resolve().parent.parent / "Datasets" / "Raw_Datasets"
MAX_DELTA = 0.5    #Threshold for abnormal price moves

#Method to ensure that all prices are positive
def validation_prices(df, filename):
    columns = [ cols for cols in ["open", "high", "low", "close"] if cols in df.columns]
    neg = (df[columns] <= 0).any(axis=1)

    if neg.any():
        print(f"The dataset {filename} has {neg.sum()} rows with non-positive prices")
        return False
    
    print(f"The dataset {filename} has no non-positive prices")
    return True

#Method to ensure that all volume values are non-negative
def validation_volume(df, filename):
    if "volume" not in df.columns:
        return True
    non_pos = df["volume"] < 0

    if non_pos.any():
        print(f"The dataset {filename} has {non_pos.sum()} rows with negative volume")
        return False
        
    print(f"The dataset {filename} has no negative volume")
    return True

#Method to check that "open" and "close" are within the interval ["low", "high"]
def validation_range(df, filename):
    required_cols = ["open", "close", "high", "low"]
    if not all(cols in df.columns for cols in required_cols):
        return True

    too_high = df["high"] < df[["open", "close", "low"]].max(axis=1)
    too_low = df["low"] > df[["open", "close", "high"]].min(axis=1)
    fault = too_high | too_low
    if fault.any(): 
        print(f"The dataset {filename} has {fault.sum()} rows with inconsistent OHLC values")
        return False

    print(f"The dataset {filename} has consistent OHLC values")
    return True

#Method to check for abnormal single bar returns
def validation_returns(df, filename, threshold = MAX_DELTA):
    if "close" not in df.columns:
        return True

    returns = df["close"].pct_change().dropna()
    abnormal = returns[returns.abs() > threshold]

    if len(abnormal) > 0:
        print(f"The dataset {filename} has {len(abnormal)} single-bar returns beyond {threshold:.0%}")
        return False

    print(f"The dataset {filename} has no abnormal returns")
    return True

#Method to check if there are bars with price movement, but zero volume
def validation_price_volume(df, filename):
    if "volume" not in df.columns or "open" not in df.columns or "close" not in df.columns:
        return True

    fault = (df["volume"] == 0) & (df["open"] != df["close"])
    if fault.any(): 
        print(f"The dataset {filename} has {fault.sum()} rows with price movement and 0 volume change")
        return False
    
    print(f"The dataset {filename} has no zero-volume rows with price movement")
    return True

def main():

    csv_files = sorted(INPUT.rglob("*.csv"))
    print(f"Found {len(csv_files)} files to validate.\n")

    for file in csv_files:
        filename = file.name
        print(f"--- {filename} ---")

        df = pd.read_csv(file, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df[df.index.notna()]

        validation_prices(df, filename)
        validation_volume(df, filename)
        validation_range(df, filename)
        validation_returns(df, filename)
        validation_price_volume(df, filename)
        print()


if __name__ == "__main__":
    main()
