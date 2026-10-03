"""
This is the base class for all HAR type models. It inherits from the basic 
Volatility model class, but adds on important abstract method to build custom input,
for each model, which will be used later on in the research
"""

import numpy as np
from abc import abstractmethod

from Models.vol_model import VolatilityModel

#Base class for all HAR models
class BaseHARModel(VolatilityModel):

    #Method to create custom HAR datasets with different horizons/inputs
    @abstractmethod
    def build_input(self, df):
        raise NotImplementedError
    
    #Method to create custom HAR datasets with different horizons/inputs
    @abstractmethod
    def forecast(self, df):
        raise NotImplementedError
        
    #Method to fit params of the HAR models
    def fit(self, df):
        rm, y = self.build_input(df)            #rm is regressor matrix, y is target volatility at time t

        weights, residuals, rank, sv = np.linalg.lstsq(rm, y, rcond=None)

        self.params = weights
        self.series = y
        self.fitted_values = rm @ weights
        self.fit_start = 0
        self.fit_end = len(y) - 1
        self.is_fitted = True
        return weights