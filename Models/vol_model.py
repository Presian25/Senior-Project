"""

This is an abstract class, which should serve as the shared base for every forecasting
model used in this thesis (GARCH-, ARIMA-, HAR-types).

The methods possessed are true for all three, based on the the way they work via forward walk validation
This includes fitting on a trailing window, forecast one step ahead from the
resulting state, then proceed forward and repeat. 
"""

from abc import ABC, abstractmethod
import numpy as np


class VolatilityModel(ABC):
    def __init__(self):
        self.params = None           #estimated parameters necessary for each model
        self.fitted_values = None    #fitted output by each model
        self.series = None           #Used as memory for the last data the model was fitted on
        self.is_fitted = False       #boolean, needed to know if the model is fit to forecast or not
        self.fit_start - None
        self.fit_end = None

    #Abstract method, which should be overridden for the unique fitting process of each models parameters
    @abstractmethod
    def fit(self, series):
        raise NotImplementedError
    #Method to forecast volatility at time t for t+1
    @abstractmethod
    def forecast(self):
        raise NotImplementedError

    #Method to ensure the model is fit to forecast
    def _check_is_fitted(self):
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fit before this operation."
            )
    #Method to output fitting process data - fitting window, params estimated
    @abstractmethod
    def fit_validation(self):
        raise NotImplementedError