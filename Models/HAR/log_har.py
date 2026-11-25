"""
Similar to bas HAR, however, uses log variance in order to stabilize the usual heavry right skew of the vol
log(RV_t) = c + sum_i beta_i * log(mean(RV_{t-window_i : t-1})) + eps_t
Importantly, this implementation allows for both daily and hourly data input
"""

import numpy as np
from Models.HAR.base_har_model import BaseHARModel


class LogHAR(BaseHARModel):

    def __init__(self, rv_col="RV_1d", windows=(1, 5, 22), window_labels=None):
        super().__init__()
        self.rv_col = rv_col
        self.windows = list(windows)
        self.window_labels = (
            list(window_labels) if window_labels is not None
            else [str(w) for w in self.windows]
        )

    #Method to build input matrix/dataset
    def build_input(self, df):
        rv = df[self.rv_col]
        log_rv = np.log(rv)

        cols = [np.ones(len(df))]
        for w in self.windows:
            rol_mean = rv.shift(1).rolling(w).mean()
            cols.append(np.log(rol_mean).values)

        reg = np.column_stack(cols)
        target = log_rv.values

        valid = np.isfinite(reg).all(axis=1) & np.isfinite(target)
        self._floor = rv[rv > 0].min()

        rm = reg[valid]
        y = target[valid]
        return rm, y

    #Method to produce a forecast
    def forecast(self, df):
        self._check_is_fitted()
        rv = df[self.rv_col]
        reg_vals = [1.0]
        for w in self.windows:
            rol_mean = max(rv.iloc[-w:].mean(), self._floor)
            reg_vals.append(np.log(rol_mean))

        log_forecast = float(np.dot(self.params, reg_vals))
        smear = float(np.mean(np.exp(self.series - self.fitted_values)))
        return np.exp(log_forecast) * smear

