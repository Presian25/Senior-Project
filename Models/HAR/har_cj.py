"""
HAR-CJ decomposes rv into a continuous and jump component using bipower variation to seperate the 2
RV_t = c + beta_d^C * C_{t-1} + beta_w^C * mean(C_{t-5:t-1})
         + beta_m^C * mean(C_{t-22:t-1}) + beta_J * J_{t-1} + eps_t

Importantly, this implementation allows for both daily and hourly data input
"""

import numpy as np
import csv
from pathlib import Path

from Models.HAR.base_har_model import BaseHARModel


class HAR_CJ(BaseHARModel):

    def __init__(self, rv_col="RV_1d", bv_col="BV_1D",windows=(1, 5, 22), window_labels=None):
        super().__init__()
        self.rv_col = rv_col
        self.bv_col = bv_col
        self.windows = list(windows)
        self.window_labels = (
            list(window_labels) if window_labels is not None
            else [str(w) for w in self.windows]
        )
    #Method to split the components
    def decompose(self, df):
        rv = df[self.rv_col]
        bv = df[self.bv_col]
        c_term = np.minimum(rv, bv)
        j_term = np.maximum(rv-bv, 0.0)
        return c_term, j_term
    
    #Method to build input matrix/dataset
    def build_input(self, df):
        c_term, j_term = self.decompose(df)
        rv = df[self.rv_col]

        cols = [np.ones(len(df))]
        for w in self.windows:
            cols.append(c_term.shift(1).rolling(w).mean().values)
        cols.append(j_term.shift(1).values)

        reg = np.column_stack(cols)
        target = rv.values

        valid = ~np.isnan(reg).any(axis=1) & ~np.isnan(target)
        valid &= np.isfinite(reg).all(axis=1)

        rm = reg[valid]
        y = target[valid]

        self._valid_index = df.index[valid]
        return rm, y

    #Method to produce a forecast
    def forecast(self, df):
        self._check_is_fitted()
        c_term, j_term = self.decompose(df)
        reg_vals = [1.0]
        for w in self.windows:
            reg_vals.append(c_term.iloc[-w:].mean())
        reg_vals.append(j_term.iloc[-1])

        return float(np.dot(self.params, reg_vals))

    #Method to collect fit data
    def fit_validation(self, log_path=None, ticker=None):
        self._check_is_fitted()

        row = {
            "model": "HAR_CJ",
            "ticker": ticker,
            "rv_col": self.rv_col,
            "bv_col": self.bv_col,
            "windows": str(self.windows),
            "fit_start": self.fit_start,
            "fit_end": self.fit_end,
            "const": self.params[0],
        }
        for label, coef in zip(self.window_labels, self.params[1:-1]):
            row[f"beta_{label}"] = coef
        row["beta_J"] = self.params[-1]

        if log_path is not None:
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_exists = log_path.exists()

            with open(log_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)

        return row