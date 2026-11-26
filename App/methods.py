"""
Contains the methods needed for the dashboard
"""

import numpy as np
import pandas as pd


from Forecasting.config import ABS_SCALE
from Forecasting.evaluate import qlike_var, dm_stat
from Blending.blend_reg import reg_labels

PREV = {"1h": 500, "1d": 60}
MIN_PER = 30
REG_NAMES = {0: "low", 1: "mid", 2: "high", -1: "unknown"}

class MinPeriodError(Exception):
    """Used to raise errors if not enough periods are found in the window"""

#Method to get the window for reccomendation
def get_rec_win(forec, rr, ts, win):
    end = forec.index.searchsorted(ts, side="left")
    begin = max(end-win, 0)
    period = forec.iloc[begin:end].dropna(axis=1)
    return period, rr.reindex(period.index)

#Method to recommend model on an x window
def recommend_model(forec, rr, ts, win, n):
    period, period_act = get_rec_win(forec, rr, ts, win)
    if len(period) < MIN_PER:
        raise MinPeriodError("Not enough Observations for this window  in order to recommend a model")

    rows = []
    for name in period.columns:
        rows.append({"model": name,
                     "QLIKE": qlike_var(period_act, period[name]).mean(),
                     "corr": period[name].corr(period_act),
                     "MAE": (period_act - ABS_SCALE * period[name]).abs().mean()})
    
    init_table = pd.DataFrame(rows)
    sorted = init_table.sort_values("QLIKE").reset_index(drop=True)
    sorted.index = range(1, len(sorted) + 1)
    sorted.index.name = "place"

    n = min(max(int(n), 1), len(sorted))
    recommended_models = []
    stat_val = []
    for i in range(1, n+1):
        curr = sorted.loc[i, "model"]
        recommended_models.append(curr)
        if i == len(sorted):
            stat_val.append(False)
            continue

        next = sorted.loc[i+1, "model"]
        dm = dm_stat(qlike_var(period_act, period[curr]), qlike_var(period_act, period[next]))

        if not np.isnan(dm) and dm < -1.96:
           stat_val.append(True)
        else:
            stat_val.append(False)

    return {"models": recommended_models, "significant": stat_val, "scores": sorted, "num_obs": len(period)}
    
#Method to get forecast for model x at time t
def get_forec(forec, rr, naive, ts):
    out = pd.DataFrame({"forec_pct": forec.loc[ts]})
    out.loc["NAIVE", "forec_pct"] = naive.loc[ts]
    out["abs_err"] = (rr.loc[ts] - out["forec_pct"] * ABS_SCALE).abs()
    return out, rr.loc[ts]

#Method to map regime to label at time t
def label_reg(naive, freq, ts):
    reg = reg_labels(naive,freq)
    pos = naive.index.get_loc(ts)
    reg_num = REG_NAMES[int(reg[pos])]
    return reg_num

#Method to get periods around requested time t
def crop_periods(forec, rr, ts, bef, aft):
    pos = forec.index.get_loc(ts)
    start = max(0, pos-bef)
    end = pos+aft+1
    forec_req = forec.iloc[start:end]
    rr_req = rr.iloc[start:end]
    return forec_req, rr_req
    

#Method to get closest period to user's request
def get_closest_per(forec, ts):
    pos = forec.index.searchsorted(ts, side="right") - 1
    for i in range(168):
        if pos - i < 0:
            return None
        if forec.index[pos - i] <= ts:
            return forec.index[pos - i]
    return None
    
#Method to get available hours at date d
def get_hs(forec, d):
    return list(forec.index[forec.index.date == d])


def summary(forec, rr, naive, freq, win):
    prev = forec.index[-1]
    try:
        rec = recommend_model(forec, rr, prev + pd.Timedelta(seconds=1), win, 1)
    except MinPeriodError:
        return {"recommended": "not enough data"}
    best = rec["models"][0]

    return {"recommended": best,
        "recent QLIKE": rec["scores"].loc[1, "QLIKE"],
        "last forecast (%)": forec.iloc[-1][best],
        "regime": label_reg(naive, freq, prev),
        "last period": prev}