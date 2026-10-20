from Models.GARCH.garch_1_1 import GARCH_1_1
from arch import arch_model
from Models.Model_Validation.model_validation import ModelValidator, load_series, ASSETS

def fit_implemented_fn(window):
    m = GARCH_1_1()
    m.fit(window)
    loglik = -m.neg_log_likelihood(m.params, window)
    return {"omega": m.params[0], "alpha": m.params[1], "beta": m.params[2]}, loglik

def fit_library_fn(window):
    am = arch_model(window, mean="Zero", vol="Garch", p=1, q=1, dist="normal", rescale=False)
    r = am.fit(disp="off")
    return {"omega": r.params["omega"], "alpha": r.params["alpha[1]"], "beta": r.params["beta[1]"]}, r.loglikelihood

def get_rv(values, t):
    return values[t + 1] ** 2

validator = ModelValidator(
    name="GARCH_1_1",
    model=GARCH_1_1,
    fit_implement=fit_implemented_fn,
    fit_library=fit_library_fn,
    true_rv = get_rv,
)

if __name__ == "__main__":
    for ticker, path, label in ASSETS:
        series = load_series(path, "log_return")
        validator.validate_fit(series.values, series.index, ticker, label)
        validator.validate_forecast(series.values, series.index, ticker, label)