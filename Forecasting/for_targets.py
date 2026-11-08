"""
This file defines the targets for the forecasting
Targets for period p:
  abs_ret_pct : |log_return[p]| * 100   - absolute return percentage
  rv_hf_pct: used only for daily data forecasting
"""

import numpy as np
import pandas as pd

from Forecasting.config import get_assets, rv_col

#Columns for specific models only
EXTRA_COLS = ("RV_pos", "RV_neg", "BV")

HF_CLASSES = ("Equities", "ETFs", "Crypto")


def load(path):
    return pd.read_csv(path, index_col=0, parse_dates=True)

#Method to build a master panel dataset. It ensures that all models forecast on the same data entries, keeps only full rows
def master_frame(path, freq):
    df = load(path)
    cols = ["log_return", rv_col(freq), *EXTRA_COLS]
    full = df[cols].notna().all(axis=1).to_numpy()
    not_full = np.flatnonzero(~full)
    start = int(not_full[-1]) + 1 if len(not_full) else 0
    first_full = int(np.argmax(full)) if full.any() else 0
    info = {
        "Total_rows": len(df),
        "Used_rows": len(df) - start,
        "leading_dropped": first_full,
        "inside_dropped": int((not_full >= first_full).sum()),   
    }
    return df.iloc[start:][cols].copy(), info

#Method for deriving clear day forecasting target from hourly data
def hf_daily_target(hourly_path):
    r = load(hourly_path)["log_return"].dropna()
    g = (r ** 2).groupby(r.index.normalize())
    total = g.sum()
    count = g.count()
    complete = count >= 0.7 * count.median()          # drop partial days
    return (np.sqrt(total.where(complete)) * 100.0).rename("rv_hf_pct")

#Method to load targets for all assets, indexed like the feature files
def load_targets(item, assets_index=None):
    df = load(item["path"])
    t = pd.DataFrame({"abs_ret_pct": df["log_return"].abs() * 100.0})
    if item["freq"] == "1d" and item["cls"] in HF_CLASSES:
        hourly = item["path"].with_name(item["path"].name.replace("_1d.csv", "_1h.csv"))
        if hourly.exists():
            t["rv_hf_pct"] = hf_daily_target(hourly).reindex(t.index)
    return t

#Method to validate completeness and check for lookahead bias
def report():
    rows = []
    for item in get_assets():
        m, info = master_frame(item["path"], item["freq"])
        r = m["log_return"]
        row = {"asset": item["asset"], "class": item["cls"], "freq": item["freq"], **info,
               "zero_ret_pct": round(float((r == 0).mean()) * 100, 2),
               "start": m.index[0].date(), "end": m.index[-1].date()}
        if item["freq"] == "1d" and item["cls"] in HF_CLASSES:
            t = load_targets(item)
            both = t.dropna()
            row["hf_days"] = len(both)
            if len(both) > 50:
                r2 = (t["abs_ret_pct"] ** 2)
                for k in (-1, 0, 1):
                    v = pd.concat([t["rv_hf_pct"] ** 2, r2.shift(k)], axis=1, sort=False).dropna()
                    row[f"corr_rvhf_vs_r2_shift{k}"] = round(float(v.corr().iloc[0, 1]), 3)
        rows.append(row)
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(out.to_string(index=False))
    return out


if __name__ == "__main__":
    report()
