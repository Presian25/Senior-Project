"""
Tests the blending part of the project. The data for the tests is forged
"""
import numpy as np
import pandas as pd
import pytest

from Blending import utilities, blend_opt, blend_reg, blend_class
from Blending import eval as blend_eval
from Blending.utilities import best_ws, save_forec


#Method to build forged forecasts of 3 models (exact, noisy, far too high) and the realized |return| they are scored against
def make_forec(n=200, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    t = rng.uniform(0.5, 2.0, n)
    y = pd.Series(np.abs(t * rng.normal(size=n)), index=index)
    forec = pd.DataFrame({"exact": t, "noisy": t * rng.uniform(0.5, 1.5, n), "high": np.full(n, t.mean() * 3)}, index=index)
    return forec, y


#Tests good model -> higher weights
def test_ws_model():
    forec, y = make_forec(400)
    w = best_ws(forec.to_numpy() ** 2, y.to_numpy() ** 2, np.full(3, 1 / 3))
    assert w.argmax() == 0
    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0).all() and (w <= 1).all()

#QLIKE blend
#Test that a forecast never depends on what happens in or after its own period
def test_bqlike_info_scope():
    forec, y = make_forec(200)
    changed = y.copy()
    changed.iloc[100] *= 20
    first = blend_opt.opt_ws(forec, y, None, "1d")
    second = blend_opt.opt_ws(forec, changed, None, "1d")

    assert first.iloc[:101].equals(second.iloc[:101])
    assert not first.iloc[101:].equals(second.iloc[101:])


#Test that the weights add up to 1.
#Also if all forecasts are perfect, the blended should be perfect too
def test_bqlike_equal_models():
    forec, y = make_forec(200)
    same = pd.DataFrame({name: forec["exact"] for name in ["a", "b", "c"]})
    out = blend_opt.opt_ws(same, y, None, "1d").dropna()
    assert out.to_numpy() == pytest.approx(same.loc[out.index, "a"].to_numpy())


#Test that the lowest forecast < blend < highest forecast
def test_bqlike_range():
    forec, y = make_forec(200)
    out = blend_opt.opt_ws(forec, y, None, "1d").dropna()
    assert len(out) == 200 - blend_opt.PERIODS["1d"]
    assert (out >= forec.loc[out.index].min(axis=1) - 1e-9).all()
    assert (out <= forec.loc[out.index].max(axis=1) + 1e-9).all()




#Blend by regime

#Test reg labels assignment
def test_breg_labels_(monkeypatch):
    monkeypatch.setattr(blend_reg, "PREV", {"1d": 6})
    naive = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 3.5])
    assert list(blend_reg.reg_labels(naive, "1d")) == [-1, -1, -1, 2, 2, 2, 1]


#Test for lookahead, same as prior blend
def test_breg_info_scope(monkeypatch):
    monkeypatch.setattr(blend_reg, "PERIODS", {"1d": 40})
    monkeypatch.setattr(blend_reg, "RECOMPUTE", {"1d": 5})
    monkeypatch.setattr(blend_reg, "PREV", {"1d": 20})
    monkeypatch.setattr(blend_reg, "MIN_ROWS", {"1d": 5})
    forec, y = make_forec(200)
    naive = forec["exact"]
    changed = y.copy()
    changed.iloc[120] *= 20
    first = blend_reg.reg_ws(forec, y, naive, "1d")
    second = blend_reg.reg_ws(forec, changed, naive, "1d")

    assert first.iloc[:121].equals(second.iloc[:121])
    assert not first.iloc[121:].equals(second.iloc[121:])


#Blend by asset group

#Method to build forged data
def forged_assets(monkeypatch, changed=None):
    items = [{"asset": "AAA", "cls": "Equities", "freq": "1d"}, {"asset": "BBB", "cls": "ETFs", "freq": "1d"},
             {"asset": "CCC", "cls": "Crypto", "freq": "1d"}, {"asset": "DDD", "cls": "FXs", "freq": "1d"}]
    data = {item["asset"]: make_forec(150, seed=i) for i, item in enumerate(items)}
    if changed is not None:
        data[changed] = make_forec(150, seed=99)
    saved = {}
    monkeypatch.setattr(blend_class, "PERIODS", {"1d": 30})
    monkeypatch.setattr(blend_class, "get_assets", lambda freqs, assets: items)
    monkeypatch.setattr(blend_class, "prep", lambda item: (*data[item["asset"]], None))
    monkeypatch.setattr(blend_class, "save_forec", lambda name, freq, asset, series: saved.update({asset: series}))
    blend_class.main()
    return saved


#Test that asset is not affected by ones outide its group
def test_asset_groups(monkeypatch):
    base = forged_assets(monkeypatch)
    other_group = forged_assets(monkeypatch, changed="DDD")
    same_group = forged_assets(monkeypatch, changed="BBB")

    assert base["AAA"].dropna().size > 0
    assert base["AAA"].equals(other_group["AAA"]) and base["BBB"].equals(other_group["BBB"])
    assert not base["CCC"].equals(other_group["CCC"])
    assert not base["AAA"].equals(same_group["AAA"])


#Test for the lookahead
def test_bgroup_info_scope():
    forec, y = make_forec(100)
    data = {"x": (forec.to_numpy() ** 2, y.to_numpy() ** 2, forec.index)}
    date = forec.index[60]
    first = blend_class.group_ws(["x"], data, date, 30)

    V, Y2, index = data["x"]
    V2, Y22 = V.copy(), Y2.copy()
    V2[60:], Y22[60:] = 5.0, 5.0
    assert blend_class.group_ws(["x"], {"x": (V2, Y22, index)}, date, 30) == pytest.approx(first)
    V2[59], Y22[59] = 50.0, 0.001
    assert not np.allclose(blend_class.group_ws(["x"], {"x": (V2, Y22, index)}, date, 30), first)


#Test that saved blends are processsed properly
def test_save_load_bs(monkeypatch, tmp_path):
    monkeypatch.setattr(utilities, "BLEND_DIR", tmp_path)
    monkeypatch.setattr(blend_eval, "BLEND_DIR", tmp_path)
    index = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    save_forec("BLEND_OPT", "1d", "AAA", pd.Series([np.nan, np.nan, 1.0, 2.0, 3.0], index=index))

    loaded = blend_eval.load_b({"freq": "1d", "asset": "AAA"})
    assert list(loaded.columns) == ["BLEND_OPT"]
    assert list(loaded["BLEND_OPT"]) == [1.0, 2.0, 3.0]