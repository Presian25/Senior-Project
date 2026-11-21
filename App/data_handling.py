"""
Loads the data for the dashboard
"""

import pandas as pd
from Forecasting.config import get_assets
from Forecasting.evaluate import FLOOR_FRACTION
from Blending.utilities import BLEND_DIR, BLEND_TABLE_DIR, BLEND_NAMES, prep

class LoadError(Exception):
    """Used to raise errors if files are missing"""

#Method to find the assets for an x freq
def list_assets(freq):
    try:
        found = get_assets([freq])
    except FileNotFoundError:
        raise LoadError("Dataset not found")

    items = {}
    for i in found:
        items[i["asset"]] = i
    return items

#Method to load blend datasets
def load_blnds(set):
    cols = {}
    for name in BLEND_NAMES:
        p = BLEND_DIR / set["freq"] / set["asset"] / f"{name}.csv"
        if p.exists():
            df = pd.read_csv(p, index_col="date", parse_dates=["date"])
            cols[name] = df["sigma_hat_pct"]
    return pd.DataFrame(cols)

#Method to load forecasts, naive, realized return for asset
def load_data(set):
    forec, act, naive = prep(set)
    if forec is None:
        raise LoadError(f"Forecasts not found for {set['asset']} {set['freq']}")
    
    blnds = load_blnds(set)
    blnds = blnds.reindex(forec.index)
    blnds = blnds.clip(lower=FLOOR_FRACTION*naive.median())

    return forec.join(blnds), act, naive

def load_period(set):
    p = BLEND_TABLE_DIR / "blend_metrics.csv"
    if not p.exists():
        return None
    rows = []
    table = pd.read_csv(p)
    for _, row in table.iterrows():
        if row["sample"] != "full" or row["target"] != "abs" or row["freq"] != set["freq"] or row["asset"] != set["asset"] or row["kind"] == "reference":
            continue
        rows.append({"model": row["model"], "QLIKE": row["QLIKE"], "corr": row["corr"]})
    if len(rows) == 0:
        return None

    dataset = pd.DataFrame(rows)
    return dataset.sort_values("QLIKE").reset_index(drop=True)
