"""
The parts shared by all the blends: paths, settings and helper methods.
To add a new blend: make a file <name>_blend.py (opt_blend.py is the simplest example), and add the name of
the blend to BLEND_NAMES and the file to run_all.py.
"""

import warnings

import numpy as np
from scipy.optimize import minimize

from Forecasting.config import RESULTS_DIR, get_assets
from Forecasting.for_targets import load
from Forecasting.evaluate import load_forec, WINDOW as NAIVE_WINDOW, FLOOR_FRACTION

BLEND_DIR = RESULTS_DIR / "blends"                
BLEND_TABLE_DIR = RESULTS_DIR / "blend_tables"    
BLEND_NAMES = ["BLEND_OPT", "BLEND_REG", "BLEND_HIER2G"]

FREQS = None
ASSETS = None
SHRINK = 0.75
VAR_FLOOR = 1e-8


#Method to find the weights with the lowest QLIKE at period p
def best_ws(variances, actual_sq, start):
    def mean_qlike(w):
        ratio = actual_sq / (variances @ w)
        return np.mean(ratio - np.log(ratio) - 1)
    result = minimize(mean_qlike, start, method="SLSQP", bounds=[(0, 1)] * len(start),
                      constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], options={"maxiter": 60})
    if result.success and np.isfinite(result.x).all():
        return result.x
    return start


#Method to load input
def prep(item):
    set = load_forec(item)
    if set.empty:
        return None, None, None
    freq = item["freq"]
    returns = load(item["path"])["log_return"]
    naive = np.sqrt((returns ** 2).rolling(NAIVE_WINDOW[freq]).mean().shift(1)) * 100
    y = (returns.abs() * 100).reindex(S.index)
    S = S[y.notna()]
    y = y[y.notna()]
    naive = naive.reindex(S.index)
    S = S.clip(lower=FLOOR_FRACTION * naive.median())    
    return S, y, naive


#Method to save forecasts
def save_forec(name, freq, asset, series):
    out_dir = BLEND_DIR / freq / asset
    out_dir.mkdir(parents=True, exist_ok=True)
    out = series.dropna().rename("sigma_hat_pct")
    out.index.name = "date"
    out.to_csv(out_dir / f"{name}.csv")


def run_per_asset(name, method):
    warnings.filterwarnings("ignore")
    for item in get_assets(FREQS, ASSETS):
        S, y, naive = prep(item)
        if S is None:
            continue
        save_forec(name, item["freq"], item["asset"], method(S, y, naive, item["freq"]))
        print("done", name, item["asset"], item["freq"], flush=True)