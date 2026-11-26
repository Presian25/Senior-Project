"""
3rd blend: calcs weights like this:
50%  its own QLIKE-optimal weights
25%  the QLIKE-optimal weights of the other assets of its group.
25%  eq weights

Important to note is that the groups are 2 - (equities, ETFs), (commodities, crypto, futures). Therefore,
an ETF asset for example is influenced by all other ETFs and equities for example.

This blend needs all assets at once, so it has its own loop.
"""

import warnings
import numpy as np
import pandas as pd

from Blending.utilities import FREQS, VAR_FLOOR, best_ws, get_assets, prep, save_forec

PERIODS = {"1h": 500, "1d": 60}        
REFIT_DAYS = 7
OWN_PCT, GROUP_PCT, EQ_PCT = 0.5, 0.25, 0.25
EQ_LIKE = ["Equities", "ETFs"]


#Method to find the optimal weights of a group of assets
def group_ws(names, data, date, k):
    vars, act_sq = [], []
    for name in names:
        V, Y2, index = data[name]
        end = index.searchsorted(date, side="left")        
        if end >= k:
            vars.append(V[end - k:end])
            act_sq.append(Y2[end - k:end])
    if not vars:
        return None
    n_models = vars[0].shape[1]
    return best_ws(np.vstack(vars), np.concatenate(act_sq), np.full(n_models, 1 / n_models))


def main():
    warnings.filterwarnings("ignore")
    items = get_assets(FREQS, None)
    for freq in sorted({item["freq"] for item in items}):
        p = PERIODS[freq]
        data, group = {}, {}
        for item in items:
            if item["freq"] == freq:
                set, y, _ = prep(item)
                if set is not None:
                    data[item["asset"]] = (set.to_numpy() ** 2, np.maximum(y.to_numpy() ** 2, VAR_FLOOR), set.index)
                    group[item["asset"]] = "equity-like" if item["cls"] in EQ_LIKE else "other"
        n_models = next(iter(data.values()))[0].shape[1]
        eq = np.full(n_models, 1 / n_models)
        start = min(index[0] for _, _, index in data.values())
        end = max(index[-1] for _, _, index in data.values())
        dates = pd.date_range(start, end, freq=f"{REFIT_DAYS}D")

        for asset, (V, _, index) in data.items():
            mates = [a for a in data if group[a] == group[asset] and a != asset]
            own = [group_ws([asset], data, d, p) for d in dates]
            others = [group_ws(mates, data, d, p) for d in dates]

            out = np.full(len(index), np.nan)
            a = dates.searchsorted(index, side="right") - 1        
            for j in range(len(dates)):
                if own[j] is None or others[j] is None:
                    continue
                w = OWN_PCT * own[j] + GROUP_PCT * others[j] + EQ_PCT * eq
                rows = np.where(a == j)[0]
                out[rows] = np.sqrt(V[rows] @ w)

            save_forec("BLEND_HIER2G", freq, asset, pd.Series(out, index=index))
            print("done BLEND_HIER2G", freq, asset, flush=True)


if __name__ == "__main__":
    main()