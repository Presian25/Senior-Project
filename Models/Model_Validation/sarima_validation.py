from Models.ARIMA.sarima import SARIMA
from statsmodels.tsa.arima.model import ARIMA as SM_ARIMA
from Models.Model_Validation.model_validation import ModelValidator, load_series, ASSETS

import numpy as np
import warnings

p, d, q = 1, 1, 1
P, D, Q = 1, 0, 1      # D=0: statsmodels' trend='t' is differenced away when d+D >= 2

# Seasonal period per dataset label (bars per season). Adjust to your data.
SEASON = {
    "low_vol_1h": 7,     # SPY: bars per trading day
    "high_vol_1h": 24,   # BTCUSDT: hours per day
    "low_vol_1d": 5,     # SPY: trading days per week
    "high_vol_1d": 7,    # BTCUSDT: days per week
}


def lags(n_nonseasonal, n_seasonal, s):
    # additive lag set, e.g. p=1, P=1, s=7 -> [1, 7]
    return sorted(set(range(1, n_nonseasonal + 1)) | {s * (k + 1) for k in range(n_seasonal)})


def make_sm_model(window, s):
    # Passing lag lists makes statsmodels additive (no phi*Phi cross-terms),
    # which matches Models.ARIMA.sarima.SARIMA.
    return SM_ARIMA(
        window,
        order=(lags(p, P, s), d, lags(q, Q, s)),
        seasonal_order=(0, D, 0, s) if D > 0 else (0, 0, 0, 0),
        trend='t' if (d + D) > 0 else 'c',
    )


def make_fit_implemented_fn(s):
    def fit_implemented_fn(window):
        m = SARIMA(p=p, d=d, q=q, P=P, D=D, Q=Q, s=s)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(window)
        loglik = -m.neg_log_likelihood(m.params, m.converted_series)

        c, phi, Phi, theta, Theta = m.set_params(m.params)
        params = {"const": c}
        params.update({f"phi_{i+1}": v for i, v in enumerate(phi)})
        params.update({f"Phi_{k+1}": v for k, v in enumerate(Phi)})
        params.update({f"theta_{j+1}": v for j, v in enumerate(theta)})
        params.update({f"Theta_{l+1}": v for l, v in enumerate(Theta)})
        return params, loglik
    return fit_implemented_fn


def make_fit_library_fn(s):
    def fit_library_fn(window):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = make_sm_model(window, s).fit()
            params = {name: value for name, value in zip(r.param_names, r.params)
                      if name != "sigma2"}
            return params, r.llf
        except Exception:
            return {}, None
    return fit_library_fn


def make_forecast_library(s):
    def forecast_library(window):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = make_sm_model(window, s).fit()
            value = np.asarray(r.forecast(steps=1))[-1]
            return value if np.isfinite(value) else np.nan
        except Exception:
            return np.nan
    return forecast_library


def get_actual_return(values, t):
    return values[t + 1]


if __name__ == "__main__":
    for ticker, path, label in ASSETS:
        s = SEASON[label]
        validator = ModelValidator(
            name="SARIMA",
            model=lambda s=s: SARIMA(p=p, d=d, q=q, P=P, D=D, Q=Q, s=s),
            fit_implement=make_fit_implemented_fn(s),
            fit_library=make_fit_library_fn(s),
            forecast_library=make_forecast_library(s),
            true_rv=get_actual_return,
        )
        series = load_series(path, "log_return")
        validator.validate_fit(series.values, series.index, ticker, label)
        validator.validate_forecast(series.values, series.index, ticker, label)