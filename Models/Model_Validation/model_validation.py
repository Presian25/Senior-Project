"""
This file contains a method to load series and a class to validate the implemented models. It requires one model from the implemented ones,
as well as optionally, its counterpart from a respective library. The optionality comes from the fact that
there is no library for the HAR models. The class validates 1) the fitting of the implemented models by comparing them 
to the fitting params of the library model and 2) the forecasting ability of the implemented models
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DERIVED_DIR = ROOT / "Data" / "Datasets" / "Derived_Datasets"
RESULTS_DIR = ROOT / "Data" / "Datasets" / "Model_Validation_Datasets"

ASSETS = [
    ("SPY", DERIVED_DIR / "ETFs" / "SPY_features_1h.csv", "low_vol"),
    ("BTCUSDT", DERIVED_DIR / "Crypto" / "BTCUSDT_features_1h.csv", "high_vol"),
]

WINDOW = 1000
REFIT = 1000               #count of data entries on which to refit the models

#Method to load a required column
def load_series(features_path, column):
    df = pd.read_csv(features_path, index_col=0, parse_dates=True)
    return df[column].dropna()

class ModelValidator:
    """
    name:    name of the model
    model: arg to call an unfit instance of the implemented model
    fit_implement:   (window) -> (params_dict, loglik_or_none)
    fit_library:    (window) -> (params_dict, loglik_or_none), or None
                   if no library equivalent exists (e.g. HAR family)
    true_rv:     the actual rv at time t
    """

    def __init__(self, name, model, fit_implement,
                 fit_library=None, true_rv=None,
                 window=WINDOW, refit=REFIT):
        self.name = name
        self.model = model
        self.fit_implement = fit_implement
        self.fit_library = fit_library
        self.true_rv = true_rv
        self.window = window
        self.refit = refit

    #Method to check fitting correctness against the library equivalent
    def validate_fit(self, values, idx, ticker="", label=""):
        if self.fit_library is None:
            print(f"[{self.name}] no library equivalent -- skipping fit validation")
            return pd.DataFrame()

        n = len(values)
        rows = []
        for t in range(self.window, n + 1, self.refit):
            window = values[t - self.window: t]
            i_params, i_score = self.fit_implement(window)
            l_params, l_score = self.fit_library(window)

            row = {"model": self.name, "ticker": ticker, "label": label,
                   "fit_end": idx[t - 1], "implemented_score": i_score, "library_score": l_score,
                   "diff": i_score - l_score if i_score is not None and l_score is not None else None}
            row.update({f"implemented_{k}": v for k, v in i_params.items()})
            row.update({f"library_{k}": v for k, v in l_params.items()})
            rows.append(row)

        df = pd.DataFrame(rows)
        self.save(df, f"{self.name}_fit_validation")
        if "diff" in df and df["diff"].notna().any():
            print(f"[{self.name}] FIT: mean diff={df['diff'].mean():.4f}  "
                  f"max abs diff={df['diff'].abs().max():.4f}  n={len(df)}")
        return df

    #Method to check forecasting ability via walk-forward one-step-ahead forecasts
    def validate_forecast(self, values, idx, ticker="", label=""):
        if self.true_rv is None:
            raise ValueError(f"[{self.name}] rv is required for forecast validation")

        n = len(values)
        rows = []
        model = None

        for t in range(self.window, n - 1):
            if model is None or (t - self.window) % self.refit == 0:
                model = self.model()
                model.fit(values[t - self.window: t])

            forecast = model.forecast()
            rv = self.true_rv(values, t)

            rows.append({"model": self.name, "ticker": ticker, "label": label,
                          "t": idx[t], "forecast": forecast, "actual": rv,
                          "error": rv - forecast})

        df = pd.DataFrame(rows)
        self.save(df, f"{self.name}_{ticker}_forecasts")

        mse = (df["error"] ** 2).mean()
        mae = df["error"].abs().mean()
        qlike = self._qlike(df["forecast"], df["actual"])
        corr = df["forecast"].corr(df["actual"])

        print(f"[{self.name}] {ticker} Metrics: MSE={mse:.8f}  MAE={mae:.8f}  "
          f"QLIKE={qlike:.6f}  corr={corr:.4f}  n={len(df)}")
        return df

    def _qlike(self, forecast, actual):
        valid = (forecast > 0) & (actual > 0)
        f = forecast[valid]
        a = actual[valid]
        ratio = a / f
        return (ratio - np.log(ratio) - 1).mean()

    #Internal: save a results dataframe to the shared results directory
    def save(self, df, filename):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(RESULTS_DIR / f"{filename}.csv", index=False)