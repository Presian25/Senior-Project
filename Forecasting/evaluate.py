"""
Shared evaluation for all models. Previous version was cleaned up as most of it was used for generating graphs
"""

import numpy as np
import pandas as pd
from scipy import stats

from Forecasting.config import ABS_SCALE, FORECAST_DIR
from Forecasting.models import MODELS
from Forecasting.for_targets import load_targets, load

INPUT_DIR = FORECAST_DIR

WINDOW = {"1h": 24, "1d": 22}
FLOOR_FRACTION = 0.1
ROLLING = 500
Y_FLOOR = 1e-6
VAR_FLOOR = 1e-8
MIN_GROUP = 30
MODEL_NAMES = [m.name for m in MODELS]
REFS = ["NAIVE", "ENS_MEAN", "ENS_MEDIAN"]


#Method to read the forecasts of model x
def load_forec(item):
    cols = {}
    for name in MODEL_NAMES:
        path = INPUT_DIR / item["freq"] / item["asset"] / f"{name}.csv"
        if path.exists():
            df = pd.read_csv(path, usecols=["date", "sigma_hat_pct"], parse_dates=["date"], index_col="date")
            cols[name] = df["sigma_hat_pct"]
    if len(cols) == 0:
        return pd.DataFrame()
    return pd.concat(cols, axis=1, sort=False).dropna(how="any")


#Method to add the naive forecast and the two ensembles next to the models
def add_ref_cols(S, item):
    real = list(S.columns)
    returns = load(item["path"])["log_return"]
    naive = np.sqrt((returns ** 2).rolling(WINDOW[item["freq"]]).mean().shift(1)) * 100
    S = S.copy()
    S["NAIVE"] = naive.reindex(S.index)
    S["ENS_MEAN"] = S[real].mean(axis=1)
    S["ENS_MEDIAN"] = S[real].median(axis=1)
    return S.dropna(), real


#Method to build the targets of oan x dataset
def prep_targets(item, S):
    t = load_targets(item)
    common = S.index.intersection(t.index)
    S = S.loc[common]
    y = t["abs_ret_pct"].loc[common]
    ok = y.notna()
    yield "abs", y[ok], S[ok], ABS_SCALE, t["abs_ret_pct"].dropna()

    if "rv_hf_pct" in t:
        y2 = t["rv_hf_pct"].reindex(S.index)
        ok = y2.notna()
        if ok.sum() > MIN_GROUP:
            yield "rvhf", y2[ok], S[ok], 1.0, None

    if item["freq"] == "1h":
        r = load(item["path"])["log_return"].dropna()
        full = r.groupby(r.index.normalize()).count()
        day = S.index.normalize()
        have = pd.Series(1, index=S.index).groupby(day).count()
        complete = have.index[have.reindex(full.index).reindex(have.index) == full.reindex(have.index)]
        keep = np.isin(day, complete)
        if keep.sum() > 0:
            S_keep = S[keep]
            d = S_keep.index.normalize()
            agg_forecast = np.sqrt((S_keep ** 2).groupby(d).sum())
            squared = (r.reindex(S_keep.index) * 100) ** 2
            agg_actual = np.sqrt(squared.groupby(d).sum())
            if len(agg_actual) > MIN_GROUP:
                yield "agg", agg_actual, agg_forecast, 1.0, None


#Methods for the two QLIKE versions 
def qlike_var(y, s):
    p = np.maximum(np.asarray(y, float) ** 2, VAR_FLOOR)
    h = np.maximum(np.asarray(s, float) ** 2, VAR_FLOOR)
    return p / h - np.log(p / h) - 1.0


def qlike_abs(y, f):
    a = np.maximum(np.asarray(y, float), Y_FLOOR)
    b = np.maximum(np.asarray(f, float), Y_FLOOR)
    return a / b - np.log(a / b) - 1.0


#Method for the Diebold-Mariano statistic
def dm_stat(loss_i, loss_j):
    d = np.asarray(loss_i, float) - np.asarray(loss_j, float)
    n = len(d)
    if n < MIN_GROUP or np.allclose(d, d[0]):
        return np.nan
    centered = d - d.mean()
    lag = int(n ** (1 / 3))
    var = np.dot(centered, centered) / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * np.dot(centered[k:], centered[:-k]) / n
    if var <= 0:
        return np.nan
    return d.mean() / np.sqrt(var / n)


#Method to calculate all the metrics of one model
def score(y, s, scale, y_hist=None):
    f = s * scale
    e = f - y
    abs_e = np.abs(e)
    over = e[e > 0]
    under = -e[e < 0]
    row = {
        "n": len(y),
        "MAE": abs_e.mean(),
        "MSE": (e ** 2).mean(),
        "RMSE": np.sqrt((e ** 2).mean()),
        "QLIKE": qlike_var(y, s).mean(),
        "QLIKE_abs": qlike_abs(y, f).mean(),
        "corr": np.corrcoef(f, y)[0, 1] if f.std() > 0 else np.nan,
        "bias": e.mean(),
        "rel_bias": e.mean() / y.mean(),
        "over_pct": (e > 0).mean() * 100,
        "under_pct": (e < 0).mean() * 100,
        "mean_over": over.mean() if len(over) else np.nan,
        "mean_under": under.mean() if len(under) else np.nan,
        "over_share_MAE": over.sum() / abs_e.sum() if abs_e.sum() > 0 else np.nan,
        "err_skew": stats.skew(e),
        "err_kurt": stats.kurtosis(e),
    }
    for q in (5, 25, 50, 75, 95):
        row[f"err_q{q:02d}"] = np.percentile(e, q)

    X = np.column_stack([np.ones(len(f)), f])
    beta = np.linalg.lstsq(X, y.to_numpy(float), rcond=None)[0]
    total = np.sum((y - y.mean()) ** 2)
    row["MZ_intercept"] = beta[0]
    row["MZ_slope"] = beta[1]
    row["MZ_R2"] = 1 - np.sum((y - X @ beta) ** 2) / total if total > 0 else np.nan

    #share of the error calc
    if y_hist is not None:
        shock_limit = y_hist.rolling(ROLLING).quantile(0.90).shift(1).reindex(y.index)
        quiet_level = y_hist.rolling(ROLLING).median().shift(1).reindex(y.index)
        ok = shock_limit.notna() & quiet_level.notna()
        if ok.any() and abs_e[ok].sum() > 0:
            shock = (y[ok] > shock_limit[ok]) & (f[ok] < 0.7 * y[ok])
            ghost = (y[ok] < quiet_level[ok]) & (f[ok] > y[ok] + 0.15 * quiet_level[ok])
            row["shock_pct_MAE"] = abs_e[ok][shock].sum() / abs_e[ok].sum() * 100
            row["ghost_pct_MAE"] = abs_e[ok][ghost].sum() / abs_e[ok].sum() * 100
    return row