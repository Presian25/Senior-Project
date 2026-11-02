from Models.ARIMA.arfima import ARFIMA
from statsmodels.tsa.arima.model import ARIMA as SM_ARIMA
from Models.Model_Validation.model_validation import ModelValidator, load_series, ASSETS

import numpy as np
import warnings

p, q = 1, 1
D_FRAC = 0.3     


def make_fit_implemented_fn(d):
    def fit_implemented_fn(window):
        m = ARFIMA(p=p, d=d, q=q)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(window)
        loglik = -m.neg_log_likelihood(m.params, m.converted_series)

        params = {"const": m.params[0]}
        params.update({f"phi_{i+1}": v for i, v in enumerate(m.params[1:1 + p])})
        params.update({f"theta_{j+1}": v for j, v in enumerate(m.params[1 + p:1 + p + q])})
        return params, loglik
    return fit_implemented_fn


def fit_library_fn(window):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = SM_ARIMA(window, order=(p, 0, q), trend='c').fit()
        params = {n: v for n, v in zip(r.param_names, r.params) if n != "sigma2"}
        return params, r.llf
    except Exception:
        return {}, None


def forecast_library(window):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = SM_ARIMA(window, order=(p, 0, q), trend='c').fit()
        value = np.asarray(r.forecast(steps=1))[-1]
        return value if np.isfinite(value) else np.nan
    except Exception:
        return np.nan


def get_actual_return(values, t):
    return values[t + 1]


red_validator = ModelValidator(
    name="ARFIMA_d0_vs_ARIMA",
    model=lambda: ARFIMA(p=p, d=0.0, q=q),
    fit_implement=make_fit_implemented_fn(0.0),
    fit_library=fit_library_fn,
    forecast_library=forecast_library,
    true_rv=get_actual_return,
)

frac_validator = ModelValidator(
    name="ARFIMA",
    model=lambda: ARFIMA(p=p, d=D_FRAC, q=q),
    fit_implement=make_fit_implemented_fn(D_FRAC),
    fit_library=None,
    forecast_library=None,
    true_rv=get_actual_return,
)

if __name__ == "__main__":
    for ticker, path, label in ASSETS:
        series = load_series(path, "log_return")
        # 1) d = 0 must reproduce ARIMA(p, 0, q) in statsmodels
        red_validator.validate_fit(series.values, series.index, ticker, label)
        red_validator.validate_forecast(series.values, series.index, ticker, label)
        # 2) real fractional run, forecasting ability only
        frac_validator.validate_forecast(series.values, series.index, ticker, label)