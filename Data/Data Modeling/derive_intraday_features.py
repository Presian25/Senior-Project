"""
The purpose of the script is to derive the intraday (or hourly) features.
The file follows similar logic to the derive_daily_features file, so check its docstring as well.

A difference is the new method - derive_daily_jump_components. 
Every row of the daily file is one day, so shifting by one row is enough to avoid information leaking for time t.
Here a row is one hour, so RV_pos, RV_neg and BV are first added up per UTC calendar day,
then shifted by one day, and every hourly row gets the totals of the previous day.
This way a row never sees later hours of its own day.
For BV, a product of two consecutive returns is only counted when both are from the same day.

The previous methodology of simply shifting by 1 row cannot work here, as the shift could result in still
a timestamp of the current day if it is for example 13:00
"""

import numpy as np
import pandas as pd
from pathlib import Path

INPUT = Path(__file__).resolve().parent.parent / "Datasets" / "Raw_Datasets"
OUTPUT = Path(__file__).resolve().parent.parent / "Datasets" / "Derived_Datasets"

HORIZONS_HOURLY = [1, 3, 6, 13, 24, 48, 72, 168]
CONST_TERM = np.pi / 2


#Method to load a dataset
def load_dataset(filepath):
    df = pd.read_csv(filepath, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df[df.index.notna()]
    return df.sort_index()


#Method to derive the intraday features
def derive_features_1h(df, horizons):
    out = pd.DataFrame(index=df.index)
    out["close"] = df["close"]
    out["log_return"] = np.log(df["close"]).diff()

    sq_return = out["log_return"] ** 2
    for h in horizons:
        out[f"RV_{h}h"] = sq_return.rolling(h).sum().shift(1)

    d_rv_pos, d_rv_neg, d_bv = derive_daily_jump_components(out["log_return"])
    row_dates = pd.to_datetime(out.index.date, utc=True)

    out["RV_pos"] = d_rv_pos.shift(1).reindex(row_dates).values
    out["RV_neg"] = d_rv_neg.shift(1).reindex(row_dates).values
    out["BV"] = d_bv.shift(1).reindex(row_dates).values

    return out


#Method to compute the daily jump-related components (check the main docstring for more info)
def derive_daily_jump_components(log_return):
    sq_return = log_return ** 2
    is_pos = log_return > 0

    d_rv_pos = sq_return.where(is_pos, 0.0).groupby(log_return.index.date).sum()
    d_rv_neg = sq_return.where(~is_pos, 0.0).groupby(log_return.index.date).sum()

    abs_return = log_return.abs()
    prod = abs_return * abs_return.shift(1)
    same_d = log_return.index.date == pd.Series(log_return.index.date, index=log_return.index).shift(1)
    prod = prod.where(same_d, 0.0)
    d_bv = (CONST_TERM * prod).groupby(log_return.index.date).sum()

    d_rv_pos.index = pd.to_datetime(d_rv_pos.index, utc=True)
    d_rv_neg.index = pd.to_datetime(d_rv_neg.index, utc=True)
    d_bv.index = pd.to_datetime(d_bv.index, utc=True)

    return d_rv_pos, d_rv_neg, d_bv


def main():
    csv_files = sorted(INPUT.rglob("*_1h.csv"))
    print(f"Found {len(csv_files)} hourly-data files.\n")

    for file in csv_files:
        filename = file.name
        print(f"--- {filename} ---")

        df = load_dataset(file)
        features = derive_features_1h(df, HORIZONS_HOURLY)

        r_path = file.relative_to(INPUT).parent
        ticker = file.stem.replace("_1h", "")
        out_path = OUTPUT / r_path / f"{ticker}_features_1h.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(out_path)

        print(f"Saved {len(features)} rows to {out_path}")
        print()


if __name__ == "__main__":
    main()