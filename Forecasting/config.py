"""
This file is used to unify the configuration of all forecasting models
"""

from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DERIVED_DIR = ROOT / "Data" / "Datasets" / "Derived_Datasets"
RESULTS_DIR = ROOT / "Data" / "Results"
FORECAST_DIR = RESULTS_DIR / "forecasts"
TABLE_DIR = RESULTS_DIR / "tables"
PILOT_DIR = RESULTS_DIR / "pilot"

WINDOW = 1000             

#Refitting window for every model, depoending on the data
REFIT_EVERY = {
    ("garch", "1h"): 24, ("garch", "1d"): 1,
    ("arima", "1h"): 24, ("arima", "1d"): 5,
    ("har",   "1h"): 1,  ("har",   "1d"): 1,
}
#Absolute return target of the forecasts
ABS_SCALE = float(np.sqrt(2.0 / np.pi))


def rv_col(freq):
    return "RV_1h" if freq == "1h" else "RV_1d"

#Gets all assets currently observed in the project
def get_assets(freqs=None, assets=None):
    found = []
    for class_dir in sorted(DERIVED_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        for f in sorted(class_dir.glob("*_features_*.csv")):
            asset, _, freq = f.stem.rpartition("_features_")
            if freqs and freq not in freqs:
                continue
            if assets and asset not in assets:
                continue
            found.append({"asset": asset, "cls": class_dir.name, "freq": freq, "path": f})
    return found
