"""
Tests the forecasting section of the project. Once again, data is forged for the tests
"""

import numpy as np
import pandas as pd
import pytest

from Forecasting import run_for, evaluate, for_targets
from Forecasting.config import ABS_SCALE
from Forecasting.models import MODELS
from Models.GARCH.garch_1_1 import GARCH_1_1

SPECS = {m.name: m for m in MODELS}


#Method to write a features file
def write_features(path, close):
    r = np.log(close).diff()
    df = pd.DataFrame({"close": close, "log_return": r})
    df["RV_1d"] = (r ** 2).shift(1)
    df["RV_pos"] = (r.clip(lower=0) ** 2).shift(1)
    df["RV_neg"] = (r.clip(upper=0) ** 2).shift(1)
    df["BV"] = (np.pi / 2 * r.abs() * r.abs().shift(1)).shift(1)
    df.index.name = "datetime"
    df.to_csv(path)


#Method to make a close price series
def make_close(n=200, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2022-01-03", periods=n, freq="D", tz="UTC")
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=index)


#Method to run model x on an asset y series
def run_model(spec, path, out_path, monkeypatch, window=100, limit=12):
    monkeypatch.setattr(run_for, "WINDOW", window)
    monkeypatch.setattr(run_for, "LIMIT", limit)
    item = {"asset": "X", "cls": "ETFs", "freq": "1d", "path": path}
    run_for.run(item, spec, out_path)
    return pd.read_csv(out_path, index_col="date", parse_dates=True)["sigma_hat_pct"]

#run_for.py

#Test that a forecast at tim t uses info only before time t
@pytest.mark.parametrize("name", ["GARCH_1_1", "ARIMA", "HAR"])
def test_run_info_scope(name, tmp_path, monkeypatch):
    close = make_close()
    write_features(tmp_path / "a.csv", close)
    first = run_model(SPECS[name], tmp_path / "a.csv", tmp_path / "fa.csv", monkeypatch)
    k = 3 + 106
    changed = close.copy()
    changed.iloc[k] *= 1.2
    write_features(tmp_path / "b.csv", changed)
    second = run_model(SPECS[name], tmp_path / "b.csv", tmp_path / "fb.csv", monkeypatch)

    day = close.index[k]
    assert first.notna().all() and second.notna().all()
    assert first[first.index <= day].equals(second[second.index <= day])
    assert not first[first.index > day].equals(second[second.index > day])


#Test that exists one forecast for every period
def test_run_dates(tmp_path, monkeypatch):
    close = make_close()
    write_features(tmp_path / "a.csv", close)
    out = run_model(SPECS["HAR"], tmp_path / "a.csv", tmp_path / "f.csv", monkeypatch)

    frame, _ = for_targets.master_frame(tmp_path / "a.csv", "1d")
    assert list(out.index) == list(frame.index[100:112])


#Test that if refir fails, model uses prev params
def test_run_no_refit(tmp_path, monkeypatch):
    calls = []
    class Temp(GARCH_1_1):
        def fit(self, series):
            calls.append(1)
            if len(calls) > 1:
                raise RuntimeError("optimizer failed")
            return super().fit(series)

    spec = SPECS["GARCH_1_1"].__class__("Flaky", "garch", Temp, lambda c: {})
    write_features(tmp_path / "a.csv", make_close())
    out = run_model(spec, tmp_path / "a.csv", tmp_path / "f.csv", monkeypatch)

    assert len(calls) > 1
    assert out.notna().all() and (out > 0).all()


#Test the conversion to std
def test_to_sigma():
    assert run_for.to_sigma("garch", 4e-4) == pytest.approx(2.0)
    assert run_for.to_sigma("har", 4e-4) == pytest.approx(2.0)
    assert run_for.to_sigma("arima", 0.8) == pytest.approx(0.8 / ABS_SCALE)
    assert run_for.to_sigma("har", -1.0) == 0.0
    assert np.isnan(run_for.to_sigma("garch", np.nan))


#evaluate.py

#Test the QLIKE calc
def test_qlike():
    """
    Here specifically we check if 1) perfect forecastin results in QLIKE 0 and\
    2) that under forecasting is perceived as worse to overestimations
    """
    assert evaluate.qlike_var([2.0], [2.0])[0] == pytest.approx(0.0)
    under = evaluate.qlike_var([2.0], [1.0])[0]
    over = evaluate.qlike_var([2.0], [4.0])[0]
    assert under > over > 0


#Test the Diebold-Mariano stat
def test_dm_stat():
    rng = np.random.default_rng(0)
    good = rng.uniform(0.5, 1.5, 200)
    bad = good + 0.5 + rng.normal(0, 0.05, 200)
    assert evaluate.dm_stat(good, bad) < -5
    assert evaluate.dm_stat(bad, good) > 5
    assert np.isnan(evaluate.dm_stat(good, good))

