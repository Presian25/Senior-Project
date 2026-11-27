"""
Tests the functionalities of the dashboard. This includes 1) the recommendation logic;
2) data loading, 3) picking periods to use and others. Data is made up for the tests as 
on the other tests

"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from App import asset_info, data_handling, methods
from Blending import blend_reg

ROOT = Path(__file__).resolve().parent.parent


#Method to build forged forecasts of 3 models, similar to blending tests
def make_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    true = rng.uniform(0.5, 2.0, n)
    rr = pd.Series(np.abs(true * rng.normal(size=n)), index=index)
    forec = pd.DataFrame({"exact": true, "noisy": true * rng.uniform(0.5, 1.5, n), "high": np.full(n, true.mean() * 3)}, index=index)
    naive = pd.Series(true, index=index)
    return forec, rr, naive



#Recommendation
#Tests the recommendation logic
def test_rec():
    """
    Three features are particularly tested:
    1) best model is first, 2)all models are sorted properly, 3) last model cannot be significant
    """
    forec, rr, _ = make_data()
    rec = methods.recommend_model(forec, rr, forec.index[150], 100, 3)

    assert rec["models"][0] == "exact"
    assert list(rec["scores"].index) == [1, 2, 3]
    assert rec["scores"]["QLIKE"].is_monotonic_increasing
    assert rec["num_obs"] == 100
    assert rec["significant"][-1] is False


#Test that that only significantly better models get the label for that
def test_rec_significance():
    """
    The models that should get the label are the ones that pass the dm test
    """
    forec, rr, _ = make_data()
    far = pd.DataFrame({"exact": forec["exact"], "bad": forec["exact"] * 3})
    close = pd.DataFrame({"exact": forec["exact"], "same": forec["exact"] * 1.001})
    assert methods.recommend_model(far, rr, forec.index[150], 100, 2)["significant"] == [True, False]
    assert methods.recommend_model(close, rr, forec.index[150], 100, 2)["significant"][0] is False


#Test that the range(models_recommended) is between 1 and all models
def test_rec_range():
    forec, rr, _ = make_data()
    assert len(methods.recommend_model(forec, rr, forec.index[150], 100, 0)["models"]) == 1
    assert len(methods.recommend_model(forec, rr, forec.index[150], 100, 9)["models"]) == 3


#Test that when history is not enough, error is raised, prev period is used
def test_rec_short_hist():
    forec, rr, _ = make_data()
    with pytest.raises(methods.MinPeriodError):
        methods.recommend_model(forec, rr, forec.index[10], 100, 1)

    ts = forec.index[150]
    changed_forec, changed_rr = forec.copy(), rr.copy()
    changed_forec.loc[ts] *= 50
    changed_rr.loc[ts] *= 50
    first = methods.recommend_model(forec, rr, ts, 100, 3)["scores"]
    second = methods.recommend_model(changed_forec, changed_rr, ts, 100, 3)["scores"]
    pd.testing.assert_frame_equal(first, second)


#Helpers
#Test the regime label shown in the dashboard
def test_reg_label(monkeypatch):
    monkeypatch.setattr(blend_reg, "PREV", {"1d": 6})
    index = pd.date_range("2024-01-01", periods=7, freq="D")
    naive = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 3.5], index=index)
    assert [methods.label_reg(naive, "1d", ts) for ts in index] == ["unknown"] * 3 + ["high"] * 3 + ["mid"]



#data_handling
#Test that assets are sorted by name, and that missing data raises err
def test_list_assets(monkeypatch):
    monkeypatch.setattr(data_handling, "get_assets", lambda freqs: [{"asset": "AAA"}, {"asset": "BBB"}])
    assert list(data_handling.list_assets("1d")) == ["AAA", "BBB"]

    def missing(freqs):
        raise FileNotFoundError
    monkeypatch.setattr(data_handling, "get_assets", missing)
    with pytest.raises(data_handling.LoadError):
        data_handling.list_assets("1d")



#Test that every asset of the project has a name and a description, and that an unknown one is still shown safely
def test_asset_names():
    assert asset_info.label("ZZZ") == "ZZZ (ZZZ)" and asset_info.describe("ZZZ") == ""
    try:
        found = {a for freq in ("1d", "1h") for a in data_handling.list_assets(freq)}
    except data_handling.LoadError:
        pytest.skip("real datasets not available")
    assert len(found) == 25
    assert found <= set(asset_info.ASSETS)


#Test that the whole dashboard starts, and that pressing Apply for one asset shows its result without errors
def test_dashboard_runs():
    try:
        data_handling.list_assets("1h")
    except data_handling.LoadError:
        pytest.skip("real datasets not available")
    at = AppTest.from_file(str(ROOT / "App" / "app.py"), default_timeout=120).run()
    assert not at.exception

    at.sidebar.selectbox[0].select("SPY").run()
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert at.subheader[0].value.startswith("S&P 500 ETF (SPY)")