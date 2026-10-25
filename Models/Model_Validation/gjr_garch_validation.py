from Models.GARCH.gjr_garch_1_1 import GJR_GARCH_1_1
from arch import arch_model
from Models.Model_Validation.model_validation import ModelValidator, load_series, ASSETS

import numpy as np


def fit_implemented_fn(window):
    m = GJR_GARCH_1_1()
    m.fit(window)
    
    loglik = -m.neg_log_likelihood(m.params, window)
    o, a, b, g = m.params
    return {"omega": o, "alpha": a, "beta": b, "gamma": g}, loglik

def fit_library_fn(window):
    am = arch_model(window, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="normal", rescale=False)
    r = am.fit(disp="off", options={'maxiter': 500}, show_warning=False)
    return {"omega": r.params["omega"], "alpha": r.params["alpha[1]"],
             "gamma": r.params["gamma[1]"], "beta": r.params["beta[1]"]}, r.loglikelihood

def forecast_library(window):
    lib_model = arch_model(window, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="normal", rescale=False)
    r = lib_model.fit(disp="off", options={'maxiter': 500}, show_warning=False)
    f = r.forecast(horizon=1, reindex=False)
    value = f.variance.values[-1, 0]
    if not np.isfinite(value) or value <= 0 or value > 1.0:
        return np.nan   
    return value

def get_rv(values, t):
    return values[t + 1] ** 2

validator = ModelValidator(
    name="GJR_GARCH_1_1",
    model=GJR_GARCH_1_1,
    fit_implement=fit_implemented_fn,
    fit_library=fit_library_fn,
    forecast_library=forecast_library,
    true_rv=get_rv,
)

if __name__ == "__main__":
    for ticker, path, label in ASSETS:
        series = load_series(path, "log_return")
        validator.validate_fit(series.values, series.index, ticker, label)
        validator.validate_forecast(series.values, series.index, ticker, label)