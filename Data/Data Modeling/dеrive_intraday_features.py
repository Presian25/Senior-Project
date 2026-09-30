"""
The purpose of the script is to derive the following input features from intraday data:
1) log_return = log(close).diff() - used by GARCH-type and ARIMA-type models
2) RV_Xh = rolling sum of squared log_return over the trailing - used by HAR-type models
                X hours, shifted by 1 bar to avoid lookahead

Horizons: 1h, 3h, 6h, 13h, 24h.

Reads every *_1h.csv under Datasets/Raw_Datasets and writes a corrseponding
*_features_1h.csv per ticker.
"""

import numpy as np
import pandas as pd
from pathlib import Path

INPUT = Path(__file__).resolve().parent.parent / "Datasets" / "Raw_Datasets"
OUTPUT = Path(__file__).resolve().parent.parent / "Datasets" / "Derived_Datasets"

HORIZONS_HOURLY = [1, 3, 6, 13, 24, 48, 72, 168]


#Method to load a dataset
def load_dataset(filepath):
    df = pd.read_csv(filepath, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df[df.index.notna()]
    return df.sort_index()


#Method to derive the features
def derive_features_1h(df, horizons):
    out = pd.DataFrame(index=df.index)
    out["close"] = df["close"]
    out["log_return"] = np.log(df["close"]).diff()

    squared_return = out["log_return"] ** 2
    for h in horizons:
        out[f"RV_{h}h"] = squared_return.rolling(h).sum().shift(1)

    return out


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