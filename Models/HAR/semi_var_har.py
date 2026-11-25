"""
A complex HAR model, which needs specific features - negative and positive RV.
This is done in order to capture potential assymetry between positive and negative shocks.
Its benefits are similar to the GJR-GARCH model.

RV_t = c + beta_pos * RV_pos_{t-1} + beta_neg * RV_neg_{t-1}
         + sum_i beta_i * mean(RV_{t-window_i : t-1}) + eps_t

Importantly, this implementation allows for both daily and hourly data input
"""

import numpy as np
from Models.HAR.base_har_model import BaseHARModel


class SemiVarHAR(BaseHARModel):

    def __init__(self, rv_col="RV_1d", rv_pos_col="RV_pos_1d", rv_neg_col="RV_neg_1d", windows=(5, 22), window_labels=None):
        super().__init__()
        self.rv_col = rv_col
        self.rv_pos_col = rv_pos_col
        self.rv_neg_col = rv_neg_col
        self.windows = list(windows)
        self.window_labels = (
            list(window_labels) if window_labels is not None
            else [str(w) for w in self.windows]
        )

    #Method to build input matrix/dataset
    def build_input(self, df):
        rv = df[self.rv_col]
        rv_pos = df[self.rv_pos_col]
        rv_neg = df[self.rv_neg_col]

        cols = [np.ones(len(df))]
        cols.append(rv_pos.shift(1).values)
        cols.append(rv_neg.shift(1).values)
        for w in self.windows:
            cols.append(rv.shift(1).rolling(w).mean().values)

        reg = np.column_stack(cols)
        target = rv.values
        valid = ~np.isnan(reg).any(axis=1) & ~np.isnan(target)
        rm = reg[valid]
        y = target[valid]
        return rm, y

    #Method to produce a forecast
    def forecast(self, df):
        self._check_is_fitted()
        rv = df[self.rv_col]
        rv_pos = df[self.rv_pos_col]
        rv_neg = df[self.rv_neg_col]
        reg_vals = [1.0, rv_pos.iloc[-1], rv_neg.iloc[-1]]
        for w in self.windows:
            reg_vals.append(rv.iloc[-w:].mean())

        return float(np.dot(self.params, reg_vals))
