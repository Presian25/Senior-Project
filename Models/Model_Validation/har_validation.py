from Models.HAR.har import HAR
from statsmodels.api import OLS
from Models.Model_Validation.model_validation import ModelValidator, load_series, ASSETS

import numpy as np
import pandas as pd

WINDOWS = (1, 5, 22)     #rv horizons

RV_COLUMN = {
    "low_vol_1h": "RV_1h",
    "high_vol_1h": "RV_1h",
    "low_vol_1d": "RV_1d",
    "high_vol_1d": "RV_1d",
}


class HARAdapter:

    def __init__(self, col):
        self.col = col
        self.model = HAR(rv_col=col, windows=WINDOWS)
        self._df = None

    def fit(self, window):
        self._df = pd.DataFrame({self.col: np.asarray(window, dtype=float)})
        self.model.fit(self._df)
        return self

    def forecast(self):
        return self.model.forecast(self._df)

    @property
    def params(self):
        return np.asarray(self.model.params, dtype=float)


def design(rv):
    rv = np.asarray(rv, dtype=float)
    n, H = len(rv), max(WINDOWS)
    c = np.concatenate([[0.0], np.cumsum(rv)])
    t = np.arange(H, n)
    cols = [(c[t] - c[t - w]) / w for w in WINDOWS]      # mean of rv[t-w:t]
    X = np.column_stack([np.ones(len(t))] + cols)
    return X, rv[t]


def next_x(rv):
    rv = np.asarray(rv, dtype=float)
    return np.array([1.0] + [rv[-w:].mean() for w in WINDOWS])


def gaussian_llf(resid):
    n = len(resid)
    return -0.5 * n * (np.log(2 * np.pi * np.mean(resid ** 2)) + 1)


def named(b):
    out = {"const": b[0]}
    out.update({f"beta_{w}": v for w, v in zip(WINDOWS, b[1:])})
    return out

#This function is needed as there is no library to validate HAR
def fns(col):
    def fit_implemented_fn(window):
        m = HARAdapter(col).fit(window)
        b = m.params
        X, y = design(window)
        return named(b), gaussian_llf(y - X @ b)

    def fit_library_fn(window):
        X, y = design(window)
        r = OLS(y, X).fit()
        return named(r.params), r.llf

    def forecast_library(window):
        X, y = design(window)
        r = OLS(y, X).fit()
        return float(r.predict(next_x(window).reshape(1, -1))[0])

    return fit_implemented_fn, fit_library_fn, forecast_library


def get_actual_rv(values, t):
    return values[t]        


if __name__ == "__main__":
    for ticker, path, label in ASSETS:
        col = RV_COLUMN[label]
        fit_i, fit_l, fc_l = fns(col)
        validator = ModelValidator(
            name="HAR",
            model=lambda col=col: HARAdapter(col),
            fit_implement=fit_i,
            fit_library=fit_l,
            forecast_library=fc_l,
            true_rv=get_actual_rv,
        )
        series = load_series(path, col)
        validator.validate_fit(series.values, series.index, ticker, label)
        validator.validate_forecast(series.values, series.index, ticker, label)