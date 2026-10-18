"""
ARFIMA(p,d,q): similar to ARIMA, however, uses fractional differencing in order
to capture long memory vol dynamics better

Differencing: y'_t = sum_{k=0}^{t} w_k * y_{t-k}w_0 = 1, w_k = w_{k-1} * (k-1-d)/k
Mean eq: y'_t = c + sum_i phi_i * y'_{t-i} + eps_t + sum_j theta_j * eps_{t-j}
"""

import numpy as np

from Models.ARIMA.base_arima_model import BaseARIMAModel

class ARFIMA(BaseARIMAModel):
    def __init__(self, p=1, d=0.3, q=1):
        super().__init__()
        self.p = p
        self.d = d
        self.q = q
        self.d_weights = None            #last val of each diff level, need for inverting

    #Method to return num of AR and MA terms
    def param_num(self):
        return self.p, self.q
    
    #Method for fractional diff weights
    def frac_d_weights(self, d, l):
        w = np.zeros(l)
        w[0] = 1.0
        for k in range(1, l):
            w[k] = w[k-1] * (k-1 -d )/k
        return w

    #Method for fractional integration weights
    def frac_i_weights(self, d, l):
        w = np.zeros(l)
        w[0] = 1.0
        for k in range(1, l):
            w[k] = w[k-1] * (k-1  +d )/k
        return w

    #Method to apply the differencing
    def convert(self, series):
        series = np.asarray(series, dtype=float)
        n = len(series)
        curr = series
        w = self.frac_d_weights(self.d, n)
        self.d_weights = w

        d_series = np.zeros(n)
        for t in range(n):
            d_series[t] = np.dot(w[:t+1], series[t::-1][:t+1])

        return d_series

    #Method to invert the differencing for 1 forecasted val
    def invert_conversion(self, last_vals, forecast_val):
        T = len(self.converted_series)
        v = self.frac_i_weights(self.d, T+1)

        prev = self.converted_series[::-1]
        total = v[0] * forecast_val
        total += np.dot(v[1:T+1], prev[:T])
        return total

    #Method to set a starting guess
    def set_initial_params(self, converted_series):
        c_0 = np.mean(converted_series)
        p_0 = [0.0]*self.p
        t_0 = [0.0]*self.q
        return [c_0] + p_0 + t_0
    
    #Method to set param bounds within (-1, 1)
    def param_bounds(self):
        bounds = [(None, None)]
        bounds += [(-0.999, 0.999)] * self.p
        bounds += [(-0.999, 0.999)] * self.q
        return bounds

    def fit_validation(self, log_path=None, ticker=None):
        self._check_is_fitted()
        c = self.params[0]
        phi = self.params[1:1+self.p]
        theta = self.params[1+self.p:1+self.p+self.q]

        row = {
            "model": "ARFIMA",
            "ticker": ticker,
            "p": self.p,
            "d": self.d,
            "q": self.q,
            "fit_start": self.fit_start,
            "fit_end": self.fit_end,
            "const": c,
        }
        for i, coef in enumerate(phi, start=1):
            row[f"phi_{i}"] = coef
        for j, coef in enumerate(theta, start=1):
            row[f"theta_{j}"] = coef

        if log_path is not None:
            import csv
            from pathlib import Path
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_exists = log_path.exists()
            with open(log_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)

        return row