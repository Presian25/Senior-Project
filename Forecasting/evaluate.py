"""
Evaluation of the saved forecasts (Data/Results/forecasts). Tables and a log go to Data/Results/tables (no plots).


Targets (all scored from the same forecasts, on the same timestamps):
  abs   |log return| * 100 of the period, the forecast is sigma_hat * sqrt(2/pi)
  rvhf  daily only (equities, ETFs, crypto): sqrt(sum of squared hourly returns of the day) * 100
  agg   hourly models only: the hourly forecasts are summed to the day and compared with the same daily target


Tables outputted: metrics_all (all stats of every model), breakdown_all, error_corr_all, dm_all, ranking_by_asset,
ranking_overall, ranking_by_class, shocks_all, shock_summary.

Log for the visualization: rolling_all, error_hist_all, calibration_all, cumulative_all, shock_paths_all,
periods/ (actual and forecasts) and run_info.json.
"""

import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from Forecasting.config import ABS_SCALE, FORECAST_DIR, TABLE_DIR, get_assets
from Forecasting.models import MODELS
from Forecasting.for_targets import load_targets, load

FREQS = None          
ASSETS = None         #for example ["SPY", "BTCUSDT"]
INPUT_DIR = FORECAST_DIR
OUT_DIR = TABLE_DIR

WINDOW = {"1h": 24, "1d": 22}   
FLOOR_FRACTION = 0.1  #elevates forecasts beyond this 
ROLLING = 500         #history used for the shock / ghost thresholds
Y_FLOOR = 1e-6
VAR_FLOOR = 1e-8
MIN_GROUP = 30        #groups with fewer periods are not reported
MODEL_NAMES = [m.name for m in MODELS]
REFS = ["NAIVE", "ENS_MEAN", "ENS_MEDIAN"]

#settings of the shock analysis
S_QUANTILE = {"1h": 0.98, "1d": 0.95}
S_HISTORY = 500       #periods used for the trailing quantile
MIN_HISTORY = 60
DECAY_WINDOW = {"1h": 12, "1d": 5}     #periods looked at after a shock starts or ends
BASE_WINDOW = {"1h": 24, "1d": 10}     #periods before a shock that give its normal level
PERSIST_MARGIN = 1.2
CATCH_UP = 0.5        #periods until half of the rise in volatility is in the forecast
CALM_TOLERANCE = 0.10 
PATH_BEFORE = 3       
MIN_EVENTS = 5        

#settings of the log
SAVE_LOG = True
SAVE_PERIODS = True  
HIST_RANGE = (-3, 3)  
HIST_BINS = 60
CALIB_BINS = 10       #forecast groups in the calibration table
ROLL_WINDOW = {"1h": 500, "1d": 60}   #periods in the rolling MAE / MSE / corr
ROLL_STEP = {"1h": 24, "1d": 5}       #one rolling value is kept every this many periods


#Method to read the forecasts of one dataset
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
    naive = np.sqrt((returns ** 2).rolling(WINDOW[item["freq"]]).mean().shift(1)) * 100   #only past returns
    S = S.copy()
    S["NAIVE"] = naive.reindex(S.index)
    S["ENS_MEAN"] = S[real].mean(axis=1)
    S["ENS_MEDIAN"] = S[real].median(axis=1)
    return S.dropna(), real


#Method to build the targets of one dataset
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
        full = r.groupby(r.index.normalize()).count()      #bars in every day
        day = S.index.normalize()
        have = pd.Series(1, index=S.index).groupby(day).count()  #forecasted bars in every day
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


#Method for the Diebold-Mariano statistic of two loss series 
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
    e = f - y                      #signed error, positive = over-prediction
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

    #Mincer-Zarnowitz regression: actual = a + b * forecast
    X = np.column_stack([np.ones(len(f)), f])
    beta = np.linalg.lstsq(X, y.to_numpy(float), rcond=None)[0]
    total = np.sum((y - y.mean()) ** 2)
    row["MZ_intercept"] = beta[0]
    row["MZ_slope"] = beta[1]
    row["MZ_R2"] = 1 - np.sum((y - X @ beta) ** 2) / total if total > 0 else np.nan

    #share of the error from missed shocks and from ghost errors
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


#Method to score the models inside groups of periods 
def breakdown(y, S, scale, real, by_hour):
    loss_q = pd.DataFrame({m: qlike_var(y, S[m]) for m in real}, index=y.index)
    loss_s = pd.DataFrame({m: (S[m] * scale - y) ** 2 for m in real})
    winner_q = loss_q.idxmin(axis=1)
    winner_s = loss_s.idxmin(axis=1)

    quarter = pd.Series(y.index.year.astype(str) + "Q" + y.index.quarter.astype(str), index=y.index)
    regime = pd.qcut(S["NAIVE"].rank(method="first"), 3, labels=["low", "mid", "high"]).astype(str)
    schemes = {"all": pd.Series("all", index=y.index), "regime": regime, "quarter": quarter}
    if by_hour:
        schemes["hour"] = pd.Series(y.index.hour, index=y.index).map(lambda h: f"{h:02d}")
    else:
        schemes["weekday"] = pd.Series(y.index.dayofweek, index=y.index).astype(str)     #0 = Monday

    rows = []
    for kind, groups in schemes.items():
        for name in sorted(groups.unique()):
            mask = (groups == name).to_numpy()
            if mask.sum() < MIN_GROUP:
                continue
            for m in S.columns:
                e = S[m][mask] * scale - y[mask]
                row = {"group_type": kind, "group": name, "model": m, "n": int(mask.sum()),
                       "MAE": np.abs(e).mean(), "bias": e.mean(), "over_pct": (e > 0).mean() * 100,
                       "QLIKE": qlike_var(y[mask], S[m][mask]).mean()}
                if m in real:
                    row["win_QLIKE_pct"] = (winner_q[mask] == m).mean() * 100
                    row["win_MSE_pct"] = (winner_s[mask] == m).mean() * 100
                rows.append(row)
    return rows


#Method to see how similar the models are: error_corr (signed errors) and deviation_corr (forecast - naive forecast)
def error_correlation(y, S, scale, real):
    errors = pd.DataFrame({m: S[m] * scale - y for m in real}).corr()
    deviations = pd.DataFrame({m: S[m] - S["NAIVE"] for m in real}).corr()
    rows = []
    for i in real:
        for j in real:
            if i != j:
                rows.append({"model_i": i, "model_j": j,
                             "error_corr": errors.loc[i, j], "deviation_corr": deviations.loc[i, j]})
    return rows


#Method to run the Diebold-Mariano test for every pair of models (QLIKE loss and squared error loss)
def dm_matrix(y, S, scale, real):
    losses = {"QLIKE": {m: qlike_var(y, S[m]) for m in real},
              "sq_error": {m: ((S[m] * scale - y) ** 2).to_numpy() for m in real}}
    rows = []
    for loss_name, L in losses.items():
        for i in real:
            for j in real:
                if i != j:
                    rows.append({"loss": loss_name, "model_i": i, "model_j": j, "DM": dm_stat(L[i], L[j])})
    return rows


#Method to flag the shock periods 
def shock_flags(y, y_hist, freq):
    history = y_hist if y_hist is not None else y
    limit = history.rolling(S_HISTORY, min_periods=MIN_HISTORY).quantile(S_QUANTILE[freq]).shift(1)
    return (y > limit.reindex(y.index)).fillna(False).astype(bool)


#Method to find  consecutive shock periods, returns the (first, last) position 
def episodes(shock):
    padded = np.concatenate([[0], np.asarray(shock, int), [0]])
    change = np.diff(padded)
    return list(zip(np.flatnonzero(change == 1), np.flatnonzero(change == -1) - 1))


#Method to look at the forecasts around the shocks
def event_study(y, F, shock, freq):
    K, B = DECAY_WINDOW[freq], BASE_WINDOW[freq]
    yv, Fv = y.to_numpy(float), F.to_numpy(float)
    n, m = Fv.shape
    out = {"react": [], "calm": [], "paths_react": [], "paths_calm": []}

    for start, end in episodes(shock):
        if start < max(B, PATH_BEFORE):
            continue
        base = yv[start - B:start].mean()
        pre = Fv[start - B:start].mean(axis=0)
        if base <= 0 or (pre <= 0).any():
            continue

        if start + K < n:
            level = yv[start + 1:start + K + 1].mean()
            if level > PERSIST_MARGIN * base:
                out["react"].append((Fv[start + 1:start + K + 1] / pre - 1) / (level / base - 1))
                rows = slice(start - PATH_BEFORE, start + K + 1)
                out["paths_react"].append(np.column_stack([Fv[rows] / pre, yv[rows] / base]))

        if end + K < n:
            level = yv[end + 1:end + K + 1].mean()
            if 0 < level <= PERSIST_MARGIN * base:
                out["calm"].append(Fv[end + 1:end + K + 1] / pre - 1)
                rows = slice(end - PATH_BEFORE, end + K + 1)
                out["paths_calm"].append(np.column_stack([Fv[rows] / pre, yv[rows] / base]))

    for key in ("react", "calm"):
        out[key] = np.array(out[key]) if out[key] else np.empty((0, K, m))
    for key in ("paths_react", "paths_calm"):
        out[key] = np.array(out[key]) if out[key] else np.empty((0, PATH_BEFORE + K + 1, m + 1))
    return out


#Method to get the first lag where the condition holds (1 = one period after), K + 1 if it never does
def first_lag(hit):
    return np.where(hit.any(axis=0), hit.argmax(axis=0) + 1, hit.shape[0] + 1)


#Method to turn the event study into numbers per model 
def shock_metrics(ev, models):
    out = pd.DataFrame(index=models)
    out["n_persistent"] = len(ev["react"])
    out["n_calm"] = len(ev["calm"])
    for col in ("react_k1", "react_k2", "react_half_life", "overstay_k1", "overstay_mean", "calm_time"):
        out[col] = np.nan
    if len(ev["react"]) >= MIN_EVENTS:
        profile = np.median(ev["react"], axis=0)       #lags x models
        out["react_k1"] = profile[0]
        out["react_k2"] = profile[1]
        out["react_half_life"] = first_lag(profile >= CATCH_UP)
    if len(ev["calm"]) >= MIN_EVENTS:
        profile = np.median(ev["calm"], axis=0)
        out["overstay_k1"] = profile[0]
        out["overstay_mean"] = profile.mean(axis=0)
        out["calm_time"] = first_lag(profile <= CALM_TOLERANCE)
    return out


#Method to split the total error of every model (in %) into three parts - jump, calm, normal
def decompose(y, F, S, shock, freq):
    H = DECAY_WINDOW[freq]
    after = (~shock) & (shock.astype(int).rolling(H, min_periods=1).max().shift(1).fillna(0) > 0)
    rows = {}
    for m in F.columns:
        e = F[m] - y
        losses = {"MAE": e.abs(), "MSE": e ** 2, "QLIKE": pd.Series(qlike_var(y, S[m]), index=y.index)}
        jump = shock & (e < 0)
        calm = after & (e > 0)
        row = {}
        for name, loss in losses.items():
            total = loss.sum()
            row[f"{name}_jump_pct"] = loss[jump].sum() / total * 100
            row[f"{name}_calm_pct"] = loss[calm].sum() / total * 100
            row[f"{name}_normal_pct"] = 100 - row[f"{name}_jump_pct"] - row[f"{name}_calm_pct"]
        rows[m] = row
    out = pd.DataFrame(rows).T
    out["shock_periods_pct"] = shock.mean() * 100
    out["after_periods_pct"] = after.mean() * 100
    return out


#Method to summarise the shock table over the assets 
def shock_summary(shocks):
    event_cols = ["react_k1", "react_k2", "react_half_life", "overstay_k1", "overstay_mean", "calm_time"]
    share_cols = [c for c in shocks.columns if c.endswith("_pct") and c.split("_")[0] in ("MAE", "MSE", "QLIKE")]
    how = {}
    for c in event_cols:
        how[c] = "median"
    for c in share_cols:
        how[c] = "mean"
    how["n_persistent"] = "sum"
    how["n_calm"] = "sum"
    return shocks.groupby(["target", "freq", "model"]).agg(how).round(3)


#Method to group the forecasts of every model by size and compare the mean forecast with the mean actual
def calibration_table(tag, y, F):
    rows = []
    for m in F.columns:
        group = pd.qcut(F[m].rank(method="first"), CALIB_BINS, labels=False)
        for g in range(CALIB_BINS):
            mask = (group == g).to_numpy()
            rows.append({**tag, "model": m, "group": g + 1, "n": int(mask.sum()),
                         "mean_forecast": F[m][mask].mean(), "mean_actual": y[mask].mean()})
    return pd.DataFrame(rows)


#Method to get the cumulative gain of every model over NAIVE (positive = better than the naive forecast)
def cumulative_gain(tag, y, F, Sc, freq):
    step = ROLL_STEP[freq]
    naive_q = qlike_var(y, Sc["NAIVE"])
    naive_se = ((F["NAIVE"] - y) ** 2).to_numpy()
    parts = []
    for m in F.columns:
        if m == "NAIVE":
            continue
        df = pd.DataFrame({"date": y.index, "model": m,
                           "cum_QLIKE_gain": np.cumsum(naive_q - qlike_var(y, Sc[m])),
                           "cum_SE_gain": np.cumsum(naive_se - ((F[m] - y) ** 2).to_numpy())})
        parts.append(df.iloc[step - 1::step])
    out = pd.concat(parts, ignore_index=True)
    for key, value in reversed(list(tag.items())):
        out.insert(0, key, value)
    return out


#Method to get the median path of all the shocks of one kind ("react" or "calm"), None if there are too few
def median_path(p, kind):
    if not p[kind]:
        return None
    data = np.concatenate(p[kind])
    if data.shape[0] < MIN_EVENTS:
        return None
    return np.median(data, axis=0), data.shape[0]


#Method to count the signed errors of every model in fixed bins
def error_hist(tag, y, F):
    edges = np.linspace(HIST_RANGE[0], HIST_RANGE[1], HIST_BINS + 1)
    rows = []
    for m in F.columns:
        e = ((F[m] - y) / y.mean()).clip(HIST_RANGE[0], HIST_RANGE[1])
        counts, _ = np.histogram(e, bins=edges)
        for k in range(HIST_BINS):
            rows.append({**tag, "model": m, "bin_left": edges[k], "bin_right": edges[k + 1],
                         "count": counts[k], "share": counts[k] / len(e)})
    return pd.DataFrame(rows)


#Method to get the rolling MAE, MSE, QLIKE, corr and relative bias of every model over time
def rolling_scores(tag, y, F, Sc, freq):
    w, step = ROLL_WINDOW[freq], ROLL_STEP[freq]
    parts = []
    for m in F.columns:
        e = F[m] - y
        qlike = pd.Series(qlike_var(y, Sc[m]), index=y.index)
        df = pd.DataFrame({"date": y.index, "model": m,
                           "roll_MAE": e.abs().rolling(w).mean().to_numpy(),
                           "roll_MSE": (e ** 2).rolling(w).mean().to_numpy(),
                           "roll_QLIKE": qlike.rolling(w).mean().to_numpy(),
                           "roll_corr": y.rolling(w).corr(F[m]).to_numpy(),
                           "roll_rel_bias": (e.rolling(w).mean() / y.rolling(w).mean()).to_numpy(),
                           "roll_target": y.rolling(w).mean().to_numpy()})
        parts.append(df.iloc[w - 1::step])
    out = pd.concat(parts, ignore_index=True)
    for key, value in reversed(list(tag.items())):
        out.insert(0, key, value)
    return out


#Method to write the median shock paths to a table
def shock_path_table(paths):
    rows = []
    for (target, freq), p in paths.items():
        names = p["names"] + ["ACTUAL"]
        lags = np.arange(-PATH_BEFORE, DECAY_WINDOW[freq] + 1)
        for kind in ("react", "calm"):
            result = median_path(p, kind)
            if result is None:
                continue
            middle, n_events = result
            for i, name in enumerate(names):
                for k, lag in enumerate(lags):
                    rows.append({"target": target, "freq": freq, "kind": kind, "model": name,
                                 "lag": lag, "median_path": middle[k, i], "n_events": n_events})
    return pd.DataFrame(rows)


#Method to rank the models FOR inside every asset, also average the ranks over all
def rank(metrics):
    rows = metrics[metrics["kind"] == "model"].copy()      
    keys = ["target", "freq", "asset"]
    rows["rank_QLIKE"] = rows.groupby(keys)["QLIKE"].rank()
    rows["rank_MAE"] = rows.groupby(keys)["MAE"].rank()
    rows["rank_RMSE"] = rows.groupby(keys)["RMSE"].rank()
    rows["rank_corr"] = rows.groupby(keys)["corr"].rank(ascending=False)    

    per_asset = rows[["target", "freq", "class", "asset", "model", "corr", "rank_corr", "MAE", "rank_MAE",
                      "RMSE", "rank_RMSE", "QLIKE", "rank_QLIKE"]]
    per_asset = per_asset.sort_values(["target", "freq", "asset", "rank_corr"])

    cols = ["rank_QLIKE", "rank_MAE", "rank_RMSE", "rank_corr", "skill_MAE", "skill_QLIKE", "corr"]
    how = {}
    for c in cols:
        how[c] = "mean" if c.startswith("rank") else "median"     
    overall = rows.groupby(["target", "freq", "model"])[cols].agg(how).round(3)
    by_class = rows.groupby(["target", "freq", "class", "model"])[cols].agg(how).round(3)
    return per_asset, overall, by_class


def main():
    warnings.filterwarnings("ignore")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    metric_rows, group_rows, corr_rows, dm_rows, coverage, shock_rows = [], [], [], [], [], []
    hist_rows, roll_rows, calib_rows, cum_rows = [], [], [], []
    shock_paths = {}
    log_dir = OUT_DIR / "log"
    if SAVE_LOG:
        (log_dir / "periods").mkdir(parents=True, exist_ok=True)

    for item in get_assets(FREQS, ASSETS):
        S = load_forec(item)
        if S.empty:
            continue
        S, real = add_ref_cols(S, item)

        for target, y, Sc, scale, y_hist in prep_targets(item, S):
            tag = {"asset": item["asset"], "class": item["cls"], "freq": item["freq"], "target": target}

            #a forecast of (almost) zero makes QLIKE explode, so forecasts are floored and the share of floored ones is saved
            raw = Sc * scale
            floor = FLOOR_FRACTION * Sc["NAIVE"].median()
            floored = (Sc < floor).mean() * 100
            Sc = Sc.clip(lower=floor)

            #all the metrics
            rows = []
            for m in Sc.columns:
                kind = "reference" if m in REFS else "model"
                rows.append({**tag, "model": m, "kind": kind, **score(y, Sc[m], scale, y_hist), "floored_pct": floored[m]})
            res = pd.DataFrame(rows)
            #health of the raw forecasts (before flooring): a flat or exploding model shows up here
            res["f_mean"] = res["model"].map(raw.mean())
            res["f_std"] = res["model"].map(raw.std())
            res["f_min"] = res["model"].map(raw.min())
            res["f_max"] = res["model"].map(raw.max())
            naive = res[res["model"] == "NAIVE"].iloc[0]
            res["skill_MAE"] = 1 - res["MAE"] / naive["MAE"]
            res["skill_QLIKE"] = naive["QLIKE"] - res["QLIKE"]
            metric_rows.append(res)

            #shocks
            F = Sc * scale
            flags = shock_flags(y, y_hist, item["freq"])
            ev = event_study(y, F, flags, item["freq"])
            shock_df = shock_metrics(ev, list(Sc.columns)).join(decompose(y, F, Sc, flags, item["freq"]))
            shock_df.insert(0, "kind", ["reference" if m in REFS else "model" for m in shock_df.index])
            shock_df.insert(0, "model", shock_df.index)
            for key, value in reversed(list(tag.items())):
                shock_df.insert(0, key, value)
            shock_rows.append(shock_df.reset_index(drop=True))

            stored = shock_paths.setdefault((target, item["freq"]), {"react": [], "calm": [], "names": list(Sc.columns)})
            if stored["names"] == list(Sc.columns):
                for kind, key in (("react", "paths_react"), ("calm", "paths_calm")):
                    if len(ev[key]):
                        stored[kind].append(ev[key])

            #groups, similarity of the models, Diebold-Mariano
            group_rows += [{**tag, **r} for r in breakdown(y, Sc, scale, real, item["freq"] == "1h" and target == "abs")]
            corr_rows += [{**tag, **r} for r in error_correlation(y, Sc, scale, real)]
            dm_rows += [{**tag, **r} for r in dm_matrix(y, Sc, scale, real)]
            coverage.append({**tag, "n_common": len(y), "models": len(real)})

            #log
            if SAVE_LOG:
                hist_rows.append(error_hist(tag, y, F))
                roll_rows.append(rolling_scores(tag, y, F, Sc, item["freq"]))
                calib_rows.append(calibration_table(tag, y, F))
                cum_rows.append(cumulative_gain(tag, y, F, Sc, item["freq"]))
                if SAVE_PERIODS:
                    periods = F.copy()
                    periods.insert(0, "shock", flags.astype(int))
                    periods.insert(0, "actual", y)
                    periods.to_csv(log_dir / "periods" / f"{item['freq']}_{item['asset']}_{target}.csv", index_label="date")
        print("scored", item["asset"], item["freq"])

    if len(metric_rows) == 0:
        print("no forecasts found in", INPUT_DIR)
        return

    metrics = pd.concat(metric_rows, ignore_index=True)
    metrics.to_csv(OUT_DIR / "metrics_all.csv", index=False)
    pd.DataFrame(group_rows).to_csv(OUT_DIR / "breakdown_all.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(OUT_DIR / "error_corr_all.csv", index=False)
    pd.DataFrame(dm_rows).to_csv(OUT_DIR / "dm_all.csv", index=False)
    pd.DataFrame(coverage).to_csv(OUT_DIR / "coverage.csv", index=False)
    shocks = pd.concat(shock_rows, ignore_index=True)
    shocks.to_csv(OUT_DIR / "shocks_all.csv", index=False)
    shock_summary(shocks).to_csv(OUT_DIR / "shock_summary.csv")

    per_asset, overall, by_class = rank(metrics)
    per_asset.to_csv(OUT_DIR / "ranking_by_asset.csv", index=False)
    overall.to_csv(OUT_DIR / "ranking_overall.csv")
    by_class.to_csv(OUT_DIR / "ranking_by_class.csv")

    if SAVE_LOG:
        pd.concat(hist_rows, ignore_index=True).to_csv(log_dir / "error_hist_all.csv", index=False)
        pd.concat(roll_rows, ignore_index=True).to_csv(log_dir / "rolling_all.csv", index=False)
        pd.concat(calib_rows, ignore_index=True).to_csv(log_dir / "calibration_all.csv", index=False)
        pd.concat(cum_rows, ignore_index=True).to_csv(log_dir / "cumulative_all.csv", index=False)
        shock_path_table(shock_paths).to_csv(log_dir / "shock_paths_all.csv", index=False)
        settings = {}
        for key, value in globals().items():
            if key.isupper() and key != "MODELS" and isinstance(value, (int, float, str, bool, dict, tuple, list, type(None))):
                settings[key] = value
        settings["run_time"] = pd.Timestamp.now().isoformat(timespec="seconds")
        with open(log_dir / "run_info.json", "w") as f:
            json.dump(settings, f, indent=1, default=str)

    pd.set_option("display.width", 200)
    print("\nAverage rank over the assets, abs target (1 = best):")
    print(overall.loc["abs"].sort_values(["freq", "rank_QLIKE"]).to_string())
    print("\nPer asset, abs target, models ordered by correlation (best first), full table in ranking_by_asset.csv:")
    for (freq, asset), g in per_asset[per_asset["target"] == "abs"].groupby(["freq", "asset"]):
        order = " > ".join(f"{m} {c:.3f}" for m, c in zip(g["model"], g["corr"]))
        print(f"  {freq} {asset:10s} {order}")
    print("\nShocks, abs target (median over the assets), reaction speed and calm-down time in periods:")
    summary = shock_summary(shocks)
    print(summary.loc["abs"][["react_k1", "react_half_life", "overstay_mean", "calm_time",
                              "MSE_jump_pct", "MSE_calm_pct", "MSE_normal_pct"]].to_string())
    print("\ntables written to", OUT_DIR)


if __name__ == "__main__":
    main()