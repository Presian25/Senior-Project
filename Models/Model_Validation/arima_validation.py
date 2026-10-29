from Models.ARIMA.arima import ARIMA
from statsmodels.tsa.arima.model import ARIMA as SM_ARIMA
from Models.Model_Validation.model_validation import ModelValidator, load_series, ASSETS

import numpy as np
import warnings

P, D, Q = 1, 1, 1


def fit_implemented_fn(window):
    m = ARIMA(p=P, d=D, q=Q)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(window)
    loglik = -m.neg_log_likelihood(m.params, m.converted_series)

    c = m.params[0]
    phi = m.params[1:1 + P]
    theta = m.params[1 + P:1 + P + Q]
    params = {"const": c}
    params.update({f"phi_{i+1}": v for i, v in enumerate(phi)})
    params.update({f"theta_{j+1}": v for j, v in enumerate(theta)})
    return params, loglik

def fit_library_fn(window):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sm_model = SM_ARIMA(window, order=(P, D, Q), trend='t' if D > 0 else 'c')
            r = sm_model.fit()
        params = {name: value for name, value in zip(r.param_names, r.params) if name != "sigma2"}
        return params, r.llf
    except Exception:
        return {}, None

def forecast_library(window):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sm_model = SM_ARIMA(window, order=(P, D, Q), trend='t' if D > 0 else 'c')
            r = sm_model.fit()
        f = r.forecast(steps=1)
        value = np.asarray(f)[-1]
        if not np.isfinite(value):
            return np.nan
        return value
    except Exception:
        return np.nan

def get_actual_return(values, t):
    return values[t + 1]

validator = ModelValidator(
    name="ARIMA",
    model=lambda: ARIMA(p=P, d=D, q=Q),
    fit_implement=fit_implemented_fn,
    fit_library=fit_library_fn,
    forecast_library=forecast_library,
    true_rv=get_actual_return,
)

if __name__ == "__main__":
    for ticker, path, label in ASSETS:
        series = load_series(path, "log_return")
        validator.validate_fit(series.values, series.index, ticker, label)
        validator.validate_forecast(series.values, series.index, ticker, label)