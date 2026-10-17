"""
Note that this is not ht emultiplicative SARIMA model, but rather the additive versiion of it

Differencing: y'_t = (1-L)^d (1-L^s)^D y_t
Mean: y'_t = c + sum_i phi_i y'_{t-i} + sum_k Phi_k y'_{t-(k+1)s}
         + eps_t + sum_j theta_j eps_{t-j} + sum_l Theta_l eps_{t-(l+1)s}
"""

import numpy as np

from Models.ARIMA.base_arima_model import BaseARIMAModel

class SARIMA(BaseARIMAModel):
    def __init__(self, p=1, d=1, q=1, P=1, D=1, Q=1, s=1):
        super().__init__()
        self.p = p
        self.d = d
        self.q = q
        self.P = P
        self.Q = Q
        self.D = D
        self.s = s
        self.dif_tails = None            #last val of each diff level, need for inverting
        self.s_tails = None

    #Method to return num of AR and MA terms
    def param_num(self):
        return self.p, self.q

    #Method to apply the differencing
    def convert(self, series):
        series = np.asarray(series, dtype=float)

        tails = []
        curr = series
        for i in range(self.d):
            tails.append(curr[-1])
            curr = np.diff(curr)
        self.dif_tails = tails

        s_tails = []
        for i in range(self.D):
            s_tails.append(curr[-self.s:].copy())
            curr = curr[self.s:] - curr[:-self.s]
        self.s_tails = s_tails
        return curr

    #Method to invert the differencing for 1 forecasted val
    def invert_conversion(self, last_vals, forecast_val):
        val = forecast_val
        for t in reversed(self.s_tails):
            val = val + t[0]
        for t in reversed(self.dif_tails):
            val = val+t
        return val
    
    #Method to retrieve all params from the param vector
    def set_params(self, params):
        p, q, P, Q = self.p, self.q, self.P, self.Q
        c = params[0]
        phi = params[1:1+p]
        Phi = params[1+p:1+p+P]
        theta = params[1+p+P:1+p+P+q]
        Theta = params[1+p+P+q:1+p+P+q+Q]
        return c, phi, Phi, theta, Theta

    #Method to set a starting guess
    def set_initial_params(self, converted_series):
        c_0 = np.mean(converted_series)
        return [c_0] + [0.0]*self.p + [0.0]*self.P + [0.0]*self.q + [0.0]*self.Q

    def recursion(self, params, series):
        c, phi, Phi, theta, Theta = self.set_params(params)
        p, q, P, Q, s = self.p, self.q, self.P, self.Q, self.s

        period = len(series)
        fit = np.zeros(period)
        res = np.zeros(period)

        for t in range(period):
            ar = sum(phi[i] * series[t-1-i] for i in range(p) if t-1-i >= 0)
            sar = sum(Phi[k] * series[t-(k+1)*s] for k in range(P) if t-(k+1)*s >= 0)
            ma = sum(theta[j] * res[t-1-j] for j in range(q) if t-1-j >= 0)
            sma = sum(Theta[l] * res[t-(l+1)*s] for l in range(Q) if t-(l+1)*s >= 0)
            fit[t] = c + ar + sar + ma + sma
            res[t] = series[t] - fit[t]

        return fit, res

    def forecast(self):
        self._check_is_fitted()
        c, phi, Phi, theta, Theta = self.set_params(self.params)
        p, q, P, Q, s = self.p, self.q, self.P, self.Q, self.s

        series = self.converted_series
        residuals = self.residuals
        n = len(series)
        ar = sum(phi[i] * series[n-1-i] for i in range(p))
        sar = sum(Phi[k] * series[n-(k+1)*s] for k in range(P) if n-(k+1)*s >= 0)
        ma = sum(theta[j] * residuals[n-1-j] for j in range(q))
        sma = sum(Theta[l] * residuals[n-(l+1)*s] for l in range(Q) if n-(l+1)*s >= 0)

        forecast_val = c + ar + sar + ma + sma
        return self.invert_conversion(self.series, forecast_val)
    #Method to set param bounds within (-1, 1)
    def param_bounds(self):
        bounds = [(None, None)]
        bounds += [(-0.999, 0.999)] * self.p
        bounds += [(-0.999, 0.999)] * self.P
        bounds += [(-0.999, 0.999)] * self.q
        bounds += [(-0.999, 0.999)] * self.Q
        return bounds

    def fit_validation(self, log_path=None, ticker=None):
        self._check_is_fitted()
        c, phi, Phi, theta, Theta = self.set_params(self.params)

        row = {
            "model": "SARIMA",
            "ticker": ticker,
            "p": self.p,
            "d": self.d,
            "q": self.q,
            "P": self.P,
            "D": self.D,
            "Q": self.Q,
            "s": self.s,
            "fit_start": self.fit_start,
            "fit_end": self.fit_end,
            "const": c,
        }
        for i, coef in enumerate(phi, start=1):
            row[f"phi_{i}"] = coef
        for k, coef in enumerate(Phi, start=1):
            row[f"Phi_{k}"] = coef
        for j, coef in enumerate(theta, start=1):
            row[f"theta_{j}"] = coef
        for l, coef in enumerate(Theta, start=1):
            row[f"Theta_{l}"] = coef

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