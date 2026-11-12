"""
Walk-forward forecasting: for every asset, frequency and model, fit on the last WINDOW
observations, forecast the next period, move one step forward and repeat.
Saves Data/Results/forecasts/<freq>/<asset>/<model>.csv with the columns date, sigma_hat_pct
(forecast std of that period's log return, in percent).
"""

import warnings

import numpy as np
import pandas as pd

from Forecasting.config import WINDOW, REFIT_EVERY, ABS_SCALE, FORECAST_DIR, get_assets, rv_col
from Forecasting.models import MODELS
from Forecasting.for_targets import master_frame, load

FREQS = None          
ASSETS = None         
MODEL_NAMES = None    
LIMIT = None          #threshold for observations
OUT = FORECAST_DIR


#Method to get length of a season, used only for SARIMA
def seasonal_period(item):
    if item["freq"] == "1d":
        return 7 if item["cls"] == "Crypto" else 5
    returns = load(item["path"])["log_return"].dropna()
    return int(returns.groupby(returns.index.normalize()).count().median())


#Method to turn the raw model output into std in percent
def to_sigma(family, raw):
    if not np.isfinite(raw):
        return np.nan
    if family == "arima":
        return max(raw, 0.0) / ABS_SCALE        
    return float(np.sqrt(max(raw, 0.0)) * 100.0)


#Run model x on asset y
def run(item, spec, path):
    family, freq = spec.family, item["freq"]
    refit_every = REFIT_EVERY[(family, freq)]

    frame, _ = master_frame(item["path"], freq)
    kwargs = spec.kwargs({"rv_col": rv_col(freq), "season": seasonal_period(item)})
    returns = frame["log_return"].to_numpy(dtype=float)
    abs_returns = np.abs(returns) * 100.0

    stop = len(frame) if LIMIT is None else min(len(frame), WINDOW + LIMIT)
    model = None
    rows = []

    for p in range(WINDOW, stop):
        #the data the model is allowed to see (the HAR columns are already lagged by one row)
        if family == "har":
            window = frame.iloc[p - WINDOW + 1: p + 1]
        elif family == "garch":
            window = returns[p - WINDOW: p]
        else:
            window = abs_returns[p - WINDOW: p]

        #re-estimate the parameters every refit_every steps, if the fit fails keep the old model
        fitted_now = False
        if model is None or (p - WINDOW) % refit_every == 0:
            try:
                new_model = spec.cls(**kwargs)
                new_model.fit(window)
                if np.isfinite(new_model.params).all():
                    model, fitted_now = new_model, True
            except Exception:
                pass

        raw = np.nan
        if model is not None:
            try:
                if family == "har":
                    raw = model.forecast(window)
                else:
                    #between refits the parameters stay fixed, but the model has to see the new window
                    if not fitted_now and family == "garch":
                        model.series = window
                        model.fitted_values = model.variance_recursion(model.params, window)
                    elif not fitted_now:
                        converted = model.convert(window)
                        model.series = window
                        model.converted_series = converted
                        model.fitted_values, model.residuals = model.recursion(model.params, converted)
                    raw = model.forecast()
            except Exception:
                pass

        rows.append((frame.index[p], to_sigma(family, raw)))

    pd.DataFrame(rows, columns=["date", "sigma_hat_pct"]).to_csv(path, index=False)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    for item in get_assets(FREQS, ASSETS):
        for spec in MODELS:
            if MODEL_NAMES and spec.name not in MODEL_NAMES:
                continue
            path = OUT / item["freq"] / item["asset"] / f"{spec.name}.csv"
            if path.exists():
                print("skip", item["asset"], item["freq"], spec.name)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            print("run ", item["asset"], item["freq"], spec.name, flush=True)
            run(item, spec, path)