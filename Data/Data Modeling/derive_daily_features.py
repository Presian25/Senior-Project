"""
The purpose of the script is to derive the following input features from daily data"
log_return, RV_Xd, RV_pos, RV_neg, BV 
Horizons: 1d, 2d, 3d, 5d, 7d, 10d, 14d, 30d.

Quick note on the financial interpretation of some vars in the code:
RV_Xd - lagged rolling sum of squared returns over horizons
RV_POS/NEG - positive and negative squared return components, used in SemiVarHar
BV - bipower variation proxy variable, used for HAR_CJ

Second note, on code design. The features shifted by 1 is a necessity in the code. This prevents information of time t to be accessible to
the models when making thew forecasts at the same time, which means they only use information before t. If this measure
is not taken, the forecasts would be unrealistically good.
"""

import numpy as np
import pandas as pd
from pathlib import Path

INPUT = Path(__file__).resolve().parent.parent / "Datasets" / "Raw_Datasets"
OUTPUT = Path(__file__).resolve().parent.parent / "Datasets" / "Derived_Datasets"

HORIZONS_DAILY = [1, 2, 3, 5, 7, 10, 14, 30]
CONST_TERM = np.pi / 2


#Method to load a dataset from a filepath
def load_dataset(filepath):
    df = pd.read_csv(filepath, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df[df.index.notna()]
    return df.sort_index()


#Method to derive the features
def derive_features_1d(df, horizons):
    """
    The features derived for daily are the following:RV_Hd, rv_pos, rv_neg, BV
    For their meaning and financial interpretation, check the main docstring on top of the file
    """
    out = pd.DataFrame(index=df.index)
    out["close"] = df["close"]
    out["log_return"] = np.log(df["close"]).diff()

    sq_return = out["log_return"] ** 2
    for h in horizons:
        out[f"RV_{h}d"] = sq_return.rolling(h).sum().shift(1)

    is_pos = out["log_return"] > 0
    abs_return = out["log_return"].abs()

    rv_pos = pd.Series(np.where(is_pos, sq_return, 0.0), index=out.index)
    rv_neg = pd.Series(np.where(~is_pos, sq_return, 0.0), index=out.index)
    out["RV_pos"] = rv_pos.shift(1)
    out["RV_neg"] = rv_neg.shift(1)
    out["BV"] = (CONST_TERM * (abs_return * abs_return.shift(1))).shift(1)

    return out


def main():
    csv_files = sorted(INPUT.rglob("*_1d.csv"))
    print(f"Found {len(csv_files)} daily-data files.\n")

    for file in csv_files:
        filename = file.name
        print(f"--- {filename} ---")

        df = load_dataset(file)
        features = derive_features_1d(df, HORIZONS_DAILY)

        r_path = file.relative_to(INPUT).parent
        ticker = file.stem.replace("_1d", "")
        out_path = OUTPUT / r_path / f"{ticker}_features_1d.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(out_path)

        print(f"Saved {len(features)} rows to {out_path}")
        print()


if __name__ == "__main__":
    main()