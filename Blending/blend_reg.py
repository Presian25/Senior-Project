"""
2nbd blend: takes into account the particular volatility regime. 
Weights are 50% based on past regime performance of an x model, the other 50 is general recent performance
"""

import numpy as np
import pandas as pd

from Blending.utilities import SHRINK, VAR_FLOOR, best_ws, run_per_asset

PERIODS = {"1h": 1500, "1d": 150}      
RECOMPUTE = {"1h": 120, "1d": 5}
PREV = {"1h": 1000, "1d": 250}       
MIN_ROWS = {"1h": 100, "1d": 25}        
REG_W = 0.5                         


#Method to CLASSIFY REGIMES
def reg_labels(naive, freq):
    q = PREV[freq]
    l = naive.rolling(q, min_periods=q // 2).quantile(1 / 3).shift(1)
    h = naive.rolling(q, min_periods=q // 2).quantile(2 / 3).shift(1)
    labels = np.where(naive <= l, 0, np.where(naive > h, 2, 1))
    labels[(l.isna() | naive.isna()).to_numpy()] = -1          
    return labels


#Method to blend with weights for every regime
def reg_ws(set, y, naive, freq):
    p = PERIODS[freq]
    vars = set.to_numpy() ** 2
    act_sq = np.maximum(y.to_numpy() ** 2, VAR_FLOOR)
    labels = reg_labels(naive, freq)
    eq = np.full(set.shape[1], 1 / set.shape[1])

    result = np.full(len(set), np.nan)
    for t in range(p, len(set)):
        if (t - p) % RECOMPUTE[freq] == 0:
            v, a, lab = vars[t - p:t], act_sq[t - p:t], labels[t - p:t]
            w_all = SHRINK * best_ws(v, a, eq) + (1 - SHRINK) * eq
            w_reg = {}
            for r in (0, 1, 2):
                rows = lab == r
                if rows.sum() >= MIN_ROWS[freq]:
                    w_reg[r] = REG_W * best_ws(v[rows], a[rows], eq) + (1 - REG_W) * w_all
                else:
                    w_reg[r] = w_all
        w = w_reg.get(labels[t], w_all)          
        result[t] = np.sqrt(vars[t] @ w)
    return pd.Series(result, index=set.index)


def main():
    run_per_asset("BLEND_REG", reg_ws)


if __name__ == "__main__":
    main()