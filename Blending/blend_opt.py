"""
1st blend, aims to minimize the QLike by weighting the forecasts of the models with the lowest Qlike
"""

import numpy as np
import pandas as pd

from Blending.utilities import SHRINK, VAR_FLOOR, best_ws, run_per_asset

PERIODS = {"1h": 500, "1d": 60}
RECOMPUTE = {"1h": 24, "1d": 5}


#Method for Qlike optimal weights
def optimal_weights(set, y, naive, freq):
    k = PERIODS[freq]
    vars = set.to_numpy() ** 2
    act_sq = np.maximum(y.to_numpy() ** 2, VAR_FLOOR)
    eq = np.full(set.shape[1], 1 / set.shape[1])

    result = np.full(len(set), np.nan)
    for t in range(k, len(set)):
        if (t - k) % RECOMPUTE[freq] == 0:          
            best = best_ws(vars[t - k:t], act_sq[t - k:t], eq)
            w = SHRINK * best + (1 - SHRINK) * eq
        result[t] = np.sqrt(vars[t] @ w)
    return pd.Series(result, index=set.index)


def main():
    run_per_asset("BLEND_OPT", optimal_weights)


if __name__ == "__main__":
    main()