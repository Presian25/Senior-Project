"""
Tests the data part of the project. This includes data gathering, validation and feature derivation
Tests are seperated per file.
Importantly, the data is a forged dataset in order to show that everything is working properly
without the need to load a dataset. This dataset is used throughout all tests, besides
the last ones flagged by real data

Quick note, there is no need to test all other assets besides crypto. They are simple scripts that reuse the methods from their
config file, therefore, if we validate the config file, we successfully validate the data_gather_x files as well (besides crypto)
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import sys

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Data"
RAW = DATA / "Datasets" / "Raw_Datasets"
DERIVED = DATA / "Datasets" / "Derived_Datasets"

for folder in ["Data Modeling", "Data_Gathering", "Data_Validation"]:
    sys.path.insert(0, str(DATA / folder))

import derive_daily_features as daily
import derive_intraday_features as hourly
import gather_config as config
import data_gather_crypto as crypto
import data_existance as existence
import data_value_validation as values


#Method to build price series with a UTC time index
def make_prices(n, freq, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"close": close}, index=index)


#Method to build a small OHLCV table that passes every value check
def make_ohlcv(n=10):
    index = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"open": 10.0, "high": 12.0, "low": 9.0, "close": 11.0, "volume": 100.0}, index=index)




#derive_daily_features tests

#Test that the features are calculated right
def test_daily_features_calculation():
    """
    Quick note, the test checks the vals for a particular 3d horizon. 
    The chekc starts for row 10 and so on as enough data is collected to perfrom a valid test
    """
    df = make_prices(30, "D")
    out = daily.derive_features_1d(df, [3])
    r = np.log(df["close"]).diff()
    t = 10

    expected_rv = (r.iloc[t - 3:t] ** 2).sum()
    expected_bv = np.pi / 2 * abs(r.iloc[t - 1]) * abs(r.iloc[t - 2]) 
    assert out["RV_3d"].iloc[t] == pytest.approx(expected_rv)
    assert out["BV"].iloc[t] == pytest.approx(expected_bv)
    assert out["RV_pos"].iloc[t] + out["RV_neg"].iloc[t] == pytest.approx(r.iloc[t - 1] ** 2)


#Test that a feature is never calculated with a lookahead bias
def test_daily_features_info_scope():
    """
    The test works as follows: we check if a change of a feature at time t or t + x 
    affects the feature derivation in any way
    """
    df = make_prices(60, "D")
    changed = df.copy()
    changed.iloc[30, 0] *= 1.5
    first = daily.derive_features_1d(df, daily.HORIZONS_DAILY)
    second = daily.derive_features_1d(changed, daily.HORIZONS_DAILY)
    features = [col for col in first.columns if col not in ("close", "log_return")]
    assert first.loc[first.index[:31], features].fillna(-1).equals(second.loc[second.index[:31], features].fillna(-1))





#derive_intraday_features tests

#Same as the prev test, adapted for hourly, follows the same logic
def test_hourly_features_info_scope():
    df = make_prices(24 * 6, "h")
    day = df.index[24 * 3].date()
    rows = df.index[df.index.date == day]
    changed = df.copy()
    changed.loc[rows[-1], "close"] *= 1.05
    first = hourly.derive_features_1h(df, hourly.HORIZONS_HOURLY)
    second = hourly.derive_features_1h(changed, hourly.HORIZONS_HOURLY)
    assert first.loc[rows[:-1]].fillna(-1).equals(second.loc[rows[:-1]].fillna(-1))


#Test that 1) prev day data is aggregated properly, 2) bv takes data from its day only
def test_hourly_rows_aggregartion():
    df = make_prices(24 * 5, "h")
    out = hourly.derive_features_1h(df, [1])
    r = np.log(df["close"]).diff()
    previous = r[r.index.date == pd.Timestamp("2024-01-02").date()]
    rows = out[out.index.date == pd.Timestamp("2024-01-03").date()]

    assert np.allclose(rows["RV_pos"], (previous[previous > 0] ** 2).sum())
    assert np.allclose(rows["RV_neg"], (previous[previous <= 0] ** 2).sum())

    day_abs = previous.abs().to_numpy()
    expected_bv = np.pi / 2 * sum(day_abs[i] * day_abs[i - 1] for i in range(1, len(day_abs)))
    assert np.allclose(rows["BV"], expected_bv)





#gather_config tests

#Method to make a yfinance table
def make_yfinance(ticker="AAA", n=5):
    index = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    names = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    columns = pd.MultiIndex.from_product([names, [ticker]])
    return pd.DataFrame(np.ones((n, 6)), index=index, columns=columns)


#Test that a download is cleaned and saved with the right file name, folder and columns
def test_fetch_proper(monkeypatch, tmp_path):
    monkeypatch.setattr(config.yf, "download", lambda ticker, **kwargs: make_yfinance(ticker))
    folder = tmp_path / "new_folder"
    config.fetch_daily("^GSPC", folder)
    config.fetch_hourly("EURUSD=X", str(folder))

    assert sorted(path.name for path in folder.iterdir()) == ["EURUSD_X_1h.csv", "GSPC_1d.csv"]
    saved = pd.read_csv(folder / "GSPC_1d.csv")
    assert list(saved.columns) == ["datetime", "open", "high", "low", "close", "volume"]




#data_gather_crypto

#Test that all pages are collected and that the result is cleaned
def test_crypto_fetch(monkeypatch):
    bars = [[i * 1000, "1", "2", "0.5", "1.5", "10", i * 1000 + 999, "0", 1, "0", "0", "0"] for i in range(8)]

    #fake Binance: returns the bars from startTime on, at most 3 per request
    def fake_get(url, params, timeout):
        part = [bar for bar in bars if bar[0] >= params["startTime"]][:params["limit"]]
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: part)

    monkeypatch.setattr(crypto, "LIMIT", 3)
    monkeypatch.setattr(crypto.requests, "get", fake_get)
    monkeypatch.setattr(crypto.time, "sleep", lambda seconds: None)

    rows = crypto.fetch_symbol("BTCUSDT", "1h", 0, 8000)
    assert len(rows) == 8 

    df = crypto.convert_to_df(rows[::-1] + rows[:2])
    assert len(df) == 8
    assert df["open_time"].is_monotonic_increasing




#data existence

#Test that a crypto series passes when complete and is flagged when an hour is missing
def test_crypto_completeness():
    df = make_prices(48, "h")
    assert existence.data_completeness_crypto(df, "x_1h.csv", "_1h")

    missing_hour = df.drop(index=pd.Timestamp("2024-01-01 10:00", tz="UTC"))
    assert not existence.data_completeness_crypto(missing_hour, "x_1h.csv", "_1h")

#Tests that if a business day is missing it is flagged
def test_daily_missing_flagged():
    df = make_prices(15, "B")
    assert existence.data_completeness(df, "x_1d.csv", "_1d")
    assert not existence.data_completeness(df.drop(df.index[5]), "x_1d.csv", "_1d")

#Tests that an hour in a day is flagged if missing
def test_hourly_gap_flagged():
    df = make_prices(12, "h")
    assert existence.data_completeness(df, "x_1h.csv", "_1h")
    assert not existence.data_completeness(df.drop(df.index[5]), "x_1h.csv", "_1h")

#Tests if duplicate row is flagged
def test_duplicate_flagged():
    df = make_prices(10, "D")
    assert existence.ensure_unqiueness(df, "x.csv")
    assert not existence.ensure_unqiueness(pd.concat([df, df.iloc[[3]]]), "x.csv")



#data_value_validation - all five checks are working properly

def test_prices_check():
    df = make_ohlcv()
    assert values.validation_prices(df, "x")
    df.loc[df.index[2], "low"] = 0.0
    assert not values.validation_prices(df, "x")

def test_volume_check():
    df = make_ohlcv()
    assert values.validation_volume(df, "x")
    df.loc[df.index[2], "volume"] = -1.0
    assert not values.validation_volume(df, "x")

def test_ohlc_range_check():
    df = make_ohlcv()
    assert values.validation_range(df, "x")
    df.loc[df.index[4], "close"] = 13.0
    assert not values.validation_range(df, "x")

def test_returns_check():
    df = make_ohlcv()
    df.loc[df.index[5:], "close"] = 11.0 * 1.10
    assert values.validation_returns(df, "x")
    df.loc[df.index[7:], "close"] = 11.0 * 1.10 * 1.60
    assert not values.validation_returns(df, "x")

def test_price_volume_check():
    df = make_ohlcv()
    assert values.validation_price_volume(df, "x")
    df.loc[df.index[3], "volume"] = 0.0
    assert not values.validation_price_volume(df, "x")
    df["volume"] = 0.0
    assert values.validation_price_volume(df, "x")




#Test the derived datasets - sanity check for the input of the models
def test_derived_features():
    if not RAW.exists() or not DERIVED.exists():
        pytest.skip("real datasets not available")

    cases = [("1d", daily, daily.derive_features_1d, daily.HORIZONS_DAILY),
             ("1h", hourly, hourly.derive_features_1h, hourly.HORIZONS_HOURLY)]
    for tag, script, function, horizons in cases:
        files = sorted(RAW.rglob(f"*_{tag}.csv"))
        assert len(files) == 25
        for file in files:
            new = function(script.load_dataset(file), horizons)
            name = file.stem.replace(f"_{tag}", "")
            saved = pd.read_csv(DERIVED / file.relative_to(RAW).parent / f"{name}_features_{tag}.csv",
                                index_col=0, parse_dates=True)
            assert new.shape == saved.shape
            assert np.allclose(new.to_numpy(dtype=float), saved.to_numpy(dtype=float), equal_nan=True)