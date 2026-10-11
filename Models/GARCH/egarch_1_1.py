"""
Model's formula:
EGARCH(1,1): omega + beta*log(sigma_{t-1}^2)
                 + alpha*(|eps_{t-1}|/sigma_{t-1} - sqrt(2/pi))
                 + gamma*(eps_{t-1}/sigma_{t-1})

The model specializes at capturing volatility shocks
"""

import numpy as np
import csv
from pathlib import Path

from Models.GARCH.base_garch_model import BaseGARCHModel


class EGARCH_1_1(BaseGARCHModel):

    #Method to eval variance recursion
    def variance_recursion(self, params, returns):
        o, a, b, g = params           #omega, alpha, beta
        period = len(returns)
        log_var = np.zeros(period)
        log_var[0] = np.log(np.var(returns))

        const_term = np.sqrt(2/np.pi)

        for t in range(1, period):
            prev_std = np.sqrt(np.exp(log_var[t-1]))
            stand_returns = returns[t-1]/prev_std
            log_var[t] = (o + b*log_var[t-1] + a*(np.abs(stand_returns) - const_term) + g * stand_returns)
        return np.exp(log_var)

    #Method to set a the starting guess for the parameters
    def initial_params(self, returns):
        sample_var = np.var(returns)
        a_0 = 0.1
        b_0 = 0.90
        o_0 = np.log(sample_var) * (1-b_0)
        g_0 = 0.05
        return [o_0, a_0, b_0, g_0]

    #Method to set parameter limits
    def param_bounds(self):
        return [(None, None), (None, None), (-0.999, 0.999), (None, None),]

    #Method to produce a variance forecast at time t for time t+1
    def forecast(self):
        self._check_is_fitted()
        o, a, b, g = self.params
        prev_return = self.series[-1]
        prev_variance = self.fitted_values[-1]
        prev_std = np.sqrt(prev_variance)
        stand = prev_return / prev_std
        const_term = np.sqrt(2/np.pi)

        forecast = (o + b* np.log(prev_variance) + a*(np.abs(stand) - const_term) + g * stand)
        return np.exp(forecast)

    #Method to save fit data
    def fit_validation(self, log_path=None, ticker=None):
        self._check_is_fitted()
        o, a, b, g= self.params

        row = {
            "model": "EGARCH_1_1",
            "ticker": ticker,
            "fit_start": self.fit_start,
            "fit_end": self.fit_end,
            "omega": o,
            "alpha": a,
            "beta": b,
            "gamma": g,
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