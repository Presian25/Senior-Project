"""
Model's formula:
GJR-GARCH(1,1): sigma_t^2 = omega + alpha*eps_{t-1}^2 + gamma*I_{t-1}*eps_{t-1}^2 + beta*sigma_{t-1}^2
where I_{t-1} = 1 if eps_{t-1} < 0, else 0.

This model's benefit is to capture levergae effect as negative shocks increase future variance
more than positive shocks, even at the same scale
"""

import numpy as np
import csv
from pathlib import Path

from Models.GARCH.base_garch_model import BaseGARCHModel


class GJR_GARCH_1_1(BaseGARCHModel):

    #Method to eval variance recursion
    def variance_recursion(self, params, returns):
        o, a, b, g = params           #omega, alpha, beta, gamma
        period = len(returns)
        variance = np.zeros(period)
        variance[0] = np.var(returns)

        for t in range(1, period):
            if returns[t-1] < 0:
                ind = 1.0
            else:
                ind = 0.0

            variance[t] = o + a * returns[t-1]**2 + g * ind * returns[t-1]**2 +  b * variance[t-1]

        return variance

    #Method to set a the starting guess for the parameters
    def initial_params(self, returns):
        sample_var = np.var(returns)
        a_0 = 0.003
        g_0 = 0.05
        b_0 = 0.90
        o_0 = sample_var * (1 - a_0 -g_0/2 - b_0)
        return [o_0, a_0, b_0, g_0]

    #Method to set parameter limits
    def param_bounds(self):
        return [(1e-8, None), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

    #Method to produce a variance forecast at time t for time t+1
    def forecast(self):
        self._check_is_fitted()
        o, a, b, g = self.params
        prev_return = self.series[-1]
        prev_variance = self.fitted_values[-1]
        if prev_return < 0:
            ind = 1.0
        else:
            ind = 0.0
        return o + a * prev_return**2 + g*ind*prev_return**2 +  b * prev_variance

    #Method to save fit data
    def fit_validation(self, log_path=None, ticker=None):
        self._check_is_fitted()
        o, a, b, g= self.params

        row = {
            "model": "GJR_GARCH_1_1",
            "ticker": ticker,
            "fit_start": self.fit_start,
            "fit_end": self.fit_end,
            "omega": o,
            "alpha": a,
            "beta": b,
            "gamma": g,
            "persistence": a + g/2 + b,
        }

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