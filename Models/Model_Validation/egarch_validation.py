from Models.GARCH.egarch_1_1 import EGARCH_1_1
from arch import arch_model
from Models.Model_Validation.model_validation import ModelValidator, load_series, ASSETS

import warnings


def fit_implemented_fn(window):
    m = EGARCH_1_1()
    #with warnings.catch_warnings():
        #warnings.simplefilter("ignore", RuntimeWarning)
    m.fit(window)

    loglik = -m.neg_log_likelihood(m.params, window)
    o, a, b, g = m.params
    return {"omega": o, "alpha": a, "gamma": g, "beta": b}, loglik

def fit_library_fn(window):
    am = arch_model(window, mean="Zero", vol="EGARCH", p=1, o=1, q=1, dist="normal", rescale=False)
    r = am.fit(disp="off", options={'maxiter': 500}, show_warning=False)
    return {"omega": r.params["omega"], "alpha": r.params["alpha[1]"],
             "gamma": r.params["gamma[1]"], "beta": r.params["beta[1]"]}, r.loglikelihood

def forecast_library(window):
    lib_model = arch_model(window, mean="Zero", vol="EGARCH", p=1, o=1, q=1, dist="normal", rescale=False)
    r = lib_model.fit(disp="off", options={'maxiter': 500}, show_warning=False)
    f = r.forecast(horizon=1, reindex=False)
    return f.variance.values[-1, 0]

def get_rv(values, t):
    return values[t + 1] ** 2

validator = ModelValidator(
    name="EGARCH_1_1",
    model=EGARCH_1_1,
    fit_implement=fit_implemented_fn,
    fit_library=fit_library_fn,
    forecast_library = forecast_library,
    true_rv = get_rv,
)

if __name__ == "__main__":
    for ticker, path, label in ASSETS:
        series = load_series(path, "log_return")
        validator.validate_fit(series.values, series.index, ticker, label)
        validator.validate_forecast(series.values, series.index, ticker, label)