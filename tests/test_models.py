"""
Tests the model part of the project: the GARCH, HAR and ARIMA families. The validation is not tested as it
is unnecessary.
Same as in data testing, the data is not real, rather created for the purpose oftesting only


"""

import numpy as np
import pandas as pd
import pytest

from Models.GARCH.garch_1_1 import GARCH_1_1
from Models.GARCH.gjr_garch_1_1 import GJR_GARCH_1_1
from Models.GARCH.egarch_1_1 import EGARCH_1_1
from Models.HAR.har import HAR
from Models.HAR.log_har import LogHAR
from Models.HAR.har_cj import HAR_CJ
from Models.HAR.semi_var_har import SemiVarHAR
from Models.ARIMA.arima import ARIMA
from Models.ARIMA.sarima import SARIMA
from Models.ARIMA.arfima import ARFIMA


#Method to simulate a base garch model
def sim_garch(n, o=2e-6, a=0.08, b=0.90, seed=0):
    rng = np.random.default_rng(seed)
    returns = np.zeros(n)
    variance = o / (1 - a - b)
    for t in range(n):
        returns[t] = np.sqrt(variance) * rng.normal()
        variance = o + a * returns[t] ** 2 + b * variance
    return returns


#Method to build a HAR table
def har_df(n=60, constant=False, seed=0):
    rng = np.random.default_rng(seed)
    if constant:
        return pd.DataFrame({"rv": 2.0, "pos": 1.0, "neg": 1.0, "bv": 2.0}, index=range(n))
    pos = rng.uniform(0.5, 1.5, n)
    neg = rng.uniform(0.5, 1.5, n)
    return pd.DataFrame({"rv": pos + neg, "pos": pos, "neg": neg, "bv": rng.uniform(1, 3, n)})


#One fresh instance per HAR model, all pointed at the columns of make_har_df
HAR_MODELS = {
    "HAR": lambda: HAR(rv_col="rv", windows=(1, 5)),
    "LogHAR": lambda: LogHAR(rv_col="rv", windows=(1, 5)),
    "HAR_CJ": lambda: HAR_CJ(rv_col="rv", bv_col="bv", windows=(1, 5)),
    "SemiVarHAR": lambda: SemiVarHAR(rv_col="rv", rv_pos_col="pos", rv_neg_col="neg", windows=(5,)),
}


#GARCH 

#Test the variance recursion
def test_garch_recursion():
    returns = np.array([0.01, -0.02, 0.03])
    o, a, b = 1e-6, 0.1, 0.8
    out = GARCH_1_1().variance_recursion([o, a, b], returns)

    v0 = np.var(returns)
    v1 = o + a * 0.01 ** 2 + b * v0
    v2 = o + a * 0.02 ** 2 + b * v1
    assert out == pytest.approx([v0, v1, v2])


#Test that GJR reacts stronger to a negative shock than to a positive one of the same size
def test_gjr_leverage():
    o, a, b, g = 1e-6, 0.05, 0.9, 0.1
    model = GJR_GARCH_1_1()
    #same numbers in a different order, so the starting variance is identical
    after_pos = model.variance_recursion([o, a, b, g], np.array([0.02, -0.02, 0.0]))
    after_neg = model.variance_recursion([o, a, b, g], np.array([-0.02, 0.02, 0.0]))
    assert after_neg[1] - after_pos[1] == pytest.approx(g * 0.02 ** 2)


#Test the EGARCH recursion
def test_egarch_recursion():
    """
    The cap on the parameters, which was discovered to be needed in development
    is considered here
    """
    returns = np.array([0.01, 0.5, 0.0])
    o, a, b, g = -5.0, 0.1, 0.9, 0.05
    out = EGARCH_1_1().variance_recursion([o, a, b, g], returns)

    c = np.sqrt(2 / np.pi)
    log_v0 = np.log(np.var(returns))
    z0 = returns[0] / np.sqrt(np.exp(log_v0))
    log_v1 = o + b * log_v0 + a * (abs(z0) - c) + g * z0

    raw_z1 = returns[1] / np.sqrt(np.exp(log_v1))
    assert raw_z1 > 5.0                                   
    log_v2 = o + b * log_v1 + a * (5.0 - c) + g * 5.0     
    assert out == pytest.approx(np.exp([log_v0, log_v1, log_v2]))


#Test garch fit nd forecast
def test_garch_fit_forec():
    returns = sim_garch(1500)
    model = GARCH_1_1()
    model.fit(returns)
    o, a, b = model.params

    assert 0.9 < a + b < 1.0
    expected = o + a * returns[-1] ** 2 + b * model.fitted_values[-1]
    assert model.forecast() == pytest.approx(expected)
    assert model.forecast() > 0

#All models

#Test that no forecasting before fitting
@pytest.mark.parametrize("model", [GARCH_1_1(), GJR_GARCH_1_1(), EGARCH_1_1(), ARIMA(), SARIMA(s=4), ARFIMA()])
def test_forecast_needs_fit(model):
    with pytest.raises(RuntimeError):
        model.forecast()


@pytest.mark.parametrize("make", HAR_MODELS.values(), ids=HAR_MODELS.keys())
def test_har_forecast_needs_fit(make):
    with pytest.raises(RuntimeError):
        make().forecast(har_df())


#HAR

#Test that data is only taken before time t
def test_har_info_scope():
    rv = np.arange(1.0, 11.0)
    df = pd.DataFrame({"rv": rv})
    rm, y = HAR(rv_col="rv", windows=(1, 3)).build_input(df)

    assert y[0] == rv[3]
    assert rm[0] == pytest.approx([1.0, rv[2], rv[0:3].mean()])
    assert len(y) == 7



#Test that LogHAR always predicts pos, even if last val is 0
def test_loghar_pos():
    df = har_df(60)
    model = LogHAR(rv_col="rv", windows=(1, 5))
    model.fit(df)

    df.loc[df.index[-6:], "rv"] = 0.0
    forecast = model.forecast(df)
    assert np.isfinite(forecast) and forecast > 0


#Test the continuous and jump split in HAR-CJ
def test_har_cj_():
    df = pd.DataFrame({"rv": [2.0, 1.0], "bv": [1.0, 2.0]})
    c, j = HAR_CJ(rv_col="rv", bv_col="bv").decompose(df)
    assert list(c) == [1.0, 1.0]
    assert list(j) == [1.0, 0.0]  


#ARIMA family

#Test that arime inverses differences properly
def test_arima_diff_invers():
    model = ARIMA(p=1, d=2, q=1)
    series = np.arange(10.0) ** 2
    assert model.convert(series) == pytest.approx([2.0] * 8)
    assert model.invert_conversion(series, 2.0) == pytest.approx(100.0)


#Same test as above, just for seasonal component in sarima
def test_sarima_seasonal():
    model = SARIMA(p=1, d=1, q=1, P=1, D=1, Q=1, s=4)
    t = np.arange(20)
    series = 2.0 * t + np.array([0, 1, 0, -1])[t % 4]

    converted = model.convert(series)
    assert converted == pytest.approx(np.zeros(len(converted)))
    assert model.invert_conversion(series, 0.0) == pytest.approx(40.0)


#Test the fractional weights and differencing in arfima
def test_arfima_ws():
    model = ARFIMA(p=1, d=0.5, q=1)
    assert model.frac_d_weights(1.0, 4) == pytest.approx([1, -1, 0, 0])
    assert model.frac_d_weights(0.5, 3) == pytest.approx([1, -0.5, -0.125])

    series = np.cumsum(np.random.default_rng(0).normal(size=31))
    model.converted_series = model.convert(series[:-1])
    next_diff = model.convert(series)[-1]
    assert model.invert_conversion(series[:-1], next_diff) == pytest.approx(series[-1])


#Test base arima fit
def test_arima_fit():
    rng = np.random.default_rng(1)
    series = np.zeros(600)
    for t in range(1, 600):
        series[t] = 0.6 * series[t - 1] + rng.normal()

    model = ARIMA(p=1, d=0, q=0)
    model.fit(series)
    assert model.params[1] == pytest.approx(0.6, abs=0.1)
    assert np.isfinite(model.forecast())