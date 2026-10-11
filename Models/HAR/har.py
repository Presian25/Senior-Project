"""
The most common HAR model, uses lagged variables of realized volatility for different horizons

RV_t = c + sum_i beta_i * mean(RV_{t-window_i : t-1}) + eps_t

Importantly, this implementation allows for both daily and hourly data input
"""

import numpy as np
import csv
from pathlib import Path

from Models.HAR.base_har_model import BaseHARModel


class HAR(BaseHARModel):

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

        cols = [np.ones(len(df))]
        for w in self.windows:
            cols.append(rv.shift(1).rolling(w).mean().values)

        reg = np.column_stack(cols)
        target = rv.values

        valid = ~np.isnan(reg).any(axis=1) & ~np.isnan(target)

        rm = reg[valid]
        y = target[valid]

        self._valid_index = df.index[valid]
        return rm, y

    #Method to produce a forecast
    def forecast(self, df):
        self._check_is_fitted()
        rv = df[self.rv_col]
        reg_vals = [1.0]
        for w in self.windows:
            reg_vals.append(rv.iloc[-w:].mean())

        return float(np.dot(self.params, reg_vals))

    #Method to collect fit data
    def fit_validation(self, log_path=None, ticker=None):
        self._check_is_fitted()

        row = {
            "model": "HAR",
            "ticker": ticker,
            "rv_col": self.rv_col,
            "windows": str(self.windows),
            "fit_start": self.fit_start,
            "fit_end": self.fit_end,
            "const": self.params[0],
        }
        for label, coef in zip(self.window_labels, self.params[1:]):
            row[f"beta_{label}"] = coef

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