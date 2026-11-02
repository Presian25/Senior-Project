"""
As HAR models do not have a benchmark library, we validate the models with custom implementation.
This script handles all this, but tests no model in particular.
"""

import numpy as np
import pandas as pd
from statsmodels.api import OLS

from Models.Model_Validation.model_validation import ModelValidator, ASSETS

WINDOWS = (1, 5, 22)     #rv horizons

COLS = {
    "rv": {"low_vol_1h": "RV_1h", "high_vol_1h": "RV_1h",
           "low_vol_1d": "RV_1d", "high_vol_1d": "RV_1d"},
    "bv": {"low_vol_1h": "BV", "high_vol_1h": "BV",
           "low_vol_1d": "BV", "high_vol_1d": "BV"},
}


#helpers

def target_i(n):
    """Row indices that have a full history for every window."""
    return np.arange(max(WINDOWS), n)

#Trailing means
def t_means(x, t):
    """mean(x[t-w:t]) for every target index t and every window w (vectorised)."""
    c = np.concatenate([[0.0], np.cumsum(np.asarray(x, dtype=float))])
    return [(c[t] - c[t - w]) / w for w in WINDOWS]


def prev_means(x):
    """mean(x[-w:]) for every window w: the regressors for the next step."""
    x = np.asarray(x, dtype=float)
    return [x[-w:].mean() for w in WINDOWS]


def gaussian_llf(resid):
    n = len(resid)
    return -0.5 * n * (np.log(2 * np.pi * np.mean(resid ** 2)) + 1)


#adapts the models to 1 interface
class HARAdapter:
    def __init__(self, data, model_cls, kwargs):
        self.data = data
        self.model = model_cls(**kwargs)
        self._df = None

    def observe(self, window):
        self._df = self.data.iloc[np.asarray(window, dtype=int)].reset_index(drop=True)

    def fit(self, window):
        self.observe(window)
        self.model.fit(self._df)
        return self

    def forecast(self):
        return self.model.forecast(self._df)

    @property
    def params(self):
        return np.asarray(self.model.params, dtype=float)


#base HAR validator
class HARValidation:
    name = None
    model_cls = None
    roles = ("rv",)          

    #Methods to be overridden in classes
    def design(self, d):
        """d: dict role -> 1-D array for the window. Returns (X, y)."""
        raise NotImplementedError

    def next_x(self, d):
        """Regressor vector for the step after the window."""
        raise NotImplementedError

    def param_names(self):
        return ["const"] + [f"beta_{w}" for w in WINDOWS]

    def model_kwargs(self, cols):
        return {"rv_col": cols["rv"], "windows": WINDOWS}

    def transform(self, pred):
        return pred          

    # --- shared ---
    def named(self, b):
        names = self.param_names()
        assert len(names) == len(b), f"{self.name}: expected {len(names)} params, got {len(b)}"
        return dict(zip(names, b))

    def run(self):
        for ticker, path, label in ASSETS:
            cols = {role: COLS[role][label] for role in self.roles}
            data = pd.read_csv(path, index_col=0, parse_dates=True)[list(cols.values())].dropna()
            arrays = {role: data[col].values for role, col in cols.items()}
            kwargs = self.model_kwargs(cols)

            def cut(window, arrays=arrays):
                pos = np.asarray(window, dtype=int)
                return {role: arr[pos] for role, arr in arrays.items()}

            def fit_implemented(window, data=data, kwargs=kwargs, cut=cut):
                b = HARAdapter(data, self.model_cls, kwargs).fit(window).params
                X, y = self.design(cut(window))
                return self.named(b), gaussian_llf(y - X @ b)

            def fit_library(window, cut=cut):
                X, y = self.design(cut(window))
                r = OLS(y, X).fit()
                return self.named(r.params), r.llf

            def forecast_library(window, cut=cut):
                d = cut(window)
                X, y = self.design(d)
                r = OLS(y, X).fit()
                pred = float(r.predict(np.asarray(self.next_x(d)).reshape(1, -1))[0])
                return self.transform(pred)

            def actual(values, t, rv_all=arrays["rv"]):
                return rv_all[int(values[t])]    

            validator = ModelValidator(
                name=self.name,
                model=lambda data=data, kwargs=kwargs: HARAdapter(data, self.model_cls, kwargs),
                fit_implement=fit_implemented,
                fit_library=fit_library,
                forecast_library=forecast_library,
                true_rv=actual,
            )
            positions = np.arange(len(data))     
            validator.validate_fit(positions, data.index, ticker, label)
            validator.validate_forecast(positions, data.index, ticker, label)