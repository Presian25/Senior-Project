"""
The purpose of the script is to derive the following input features from daily data:
1) log_return = log(close).diff() - used by GARCH-type and ARIMA-type models
2) RV_Xd = rolling sum of squared log_return over the trailing X days,
           shifted by 1 bar to avoid lookahead - used by HAR-type models
3) RV_pos, RV_neg = signed squared returns (positive / non-positive return),
                    shifted by 1 bar to avoid lookahead
4) BV = (pi/2) * |r_t| * |r_{t-1}| single-day bipower proxy,
        shifted by 1 bar to avoid lookahead

Horizons: 1d, 2d, 3d, 5d, 7d, 10d, 14d, 30d.

Reads every *_1d.csv under Datasets/Raw_Datasets and writes a corresponding
*_features_1d.csv per ticker.
"""

import numpy as np
import pandas as pd
from pathlib import Path

INPUT = Path(__file__).resolve().parent.parent / "Datasets" / "Raw_Datasets"
OUTPUT = Path(__file__).resolve().parent.parent / "Datasets" / "Derived_Datasets"

HORIZONS_DAILY = [1, 2, 3, 5, 7, 10, 14, 30]
const_term = np.pi / 2


#Method to load a dataset
def load_dataset(filepath):
    df = pd.read_csv(filepath, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df[df.index.notna()]
    return df.sort_index()


#Method to derive the features
def derive_features_1d(df, horizons):
    out = pd.DataFrame(index=df.index)
    out["close"] = df["close"]
    out["log_return"] = np.log(df["close"]).diff()

    squared_return = out["log_return"] ** 2
    for h in horizons:
        out[f"RV_{h}d"] = squared_return.rolling(h).sum().shift(1)

    # Every HAR-type feature is shifted by one bar, so row t only contains information
    # available strictly before t.
    is_pos = out["log_return"] > 0
    abs_return = out["log_return"].abs()

    rv_pos = pd.Series(np.where(is_pos, squared_return, 0.0), index=out.index)
    rv_neg = pd.Series(np.where(~is_pos, squared_return, 0.0), index=out.index)
    out["RV_pos"] = rv_pos.shift(1)
    out["RV_neg"] = rv_neg.shift(1)
    out["BV"] = (const_term * (abs_return * abs_return.shift(1))).shift(1)

    return out


def main():
    csv_files = sorted(INPUT.rglob("*_1d.csv"))
    print(f"Found {len(csv_files)} daily-data files.\n")

    for file in csv_files:
        filename = file.name
        print(f"--- {filename} ---")

        df = load_dataset(file)
        features = derive_features_1d(df, HORIZONS_DAILY)

        rel_path = file.relative_to(INPUT).parent
        ticker = file.stem.replace("_1d", "")
        out_path = OUTPUT / rel_path / f"{ticker}_features_1d.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(out_path)

        print(f"Saved {len(features)} rows to {out_path}")
        print()


if __name__ == "__main__":
    main()