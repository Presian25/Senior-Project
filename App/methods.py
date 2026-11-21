"""
Contains the methods needed for the dashboard
"""

import numpy as np
import pandas as pd

from Forecasting.config import ABS_SCALE
from Forecasting.evaluate import qlike_var, dm_stat
from Blending.blend_reg import reg_labels

PREV = {"1h": 500, "1d": 60}
MIN_PER = 30
REG_NAMES = {0: "low", 1: "mid", 2: "high", -1: "unknown"}