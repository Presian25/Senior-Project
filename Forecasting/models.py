"""
File to initiate each model, keeps in storage the models' different specifications - args needed, targets, etc.
"""

from dataclasses import dataclass
from typing import Callable

from Models.GARCH.garch_1_1 import GARCH_1_1
from Models.GARCH.gjr_garch_1_1 import GJR_GARCH_1_1
from Models.GARCH.egarch_1_1 import EGARCH_1_1
from Models.ARIMA.arima import ARIMA
from Models.ARIMA.sarima import SARIMA
from Models.ARIMA.arfima import ARFIMA
from Models.HAR.har import HAR
from Models.HAR.log_har import LogHAR
from Models.HAR.har_cj import HAR_CJ
from Models.HAR.semi_var_har import SemiVarHAR

HAR_WINDOWS = (1, 5, 22)      #used only for har forecasting
SV_WINDOWS = (5, 22)


@dataclass
class Model:
    name: str
    family: str
    cls: type
    kwargs: Callable          # rv_col and season for the constructors of the models


MODELS = [
    #GARCH models
    Model("GARCH_1_1", "garch", GARCH_1_1, lambda c: {}),
    Model("GJR_GARCH_1_1", "garch", GJR_GARCH_1_1, lambda c: {}),
    Model("EGARCH_1_1", "garch", EGARCH_1_1, lambda c: {}),
    #ARIMA models
    Model("ARIMA", "arima", ARIMA, lambda c: dict(p=1, d=0, q=1)),
    Model("SARIMA", "arima", SARIMA, lambda c: dict(p=1, d=0, q=1, P=1, D=0, Q=1, s=c["season"])),
    Model("ARFIMA", "arima", ARFIMA, lambda c: dict(p=1, d=0.3, q=1)),
    #HAR models
    Model("HAR", "har", HAR, lambda c: dict(rv_col=c["rv_col"], windows=HAR_WINDOWS)),
    Model("LogHAR", "har", LogHAR, lambda c: dict(rv_col=c["rv_col"], windows=HAR_WINDOWS)),
    Model("HAR_CJ", "har", HAR_CJ, lambda c: dict(rv_col=c["rv_col"], bv_col="BV", windows=HAR_WINDOWS)),
    Model("SemiVarHAR", "har", SemiVarHAR, lambda c: dict(rv_col=c["rv_col"], rv_pos_col="RV_pos", rv_neg_col="RV_neg", windows=SV_WINDOWS)),
]
#map MODELS - Model
map = {m.name: m for m in MODELS}
