"""
This is the base class for all garch type models. It inherits from the basic 
Volatility model class, implements the methods for fit and forecasting, but also
provides new abstract ones, necessary for each specific algorithm
"""

import numpy as np
from scipy.optimize import minimize
from abc import abstractmethod

from Models.vol_model import VolatilityModel

#Base calss for the GARCH models
class BaseGARCHModel(VolatilityModel):

    #Abstract method, which allows each GARCH-Type model to define its own variance recursion formula
    @abstractmethod
    def variance_recursion(self, params, returns):
        raise NotImplementedError

    #Abstract method, which allows each GARCH-Type model to define its own starting guess for the optimizer
    @abstractmethod
    def initial_params(self, returns):
        raise NotImplementedError

    #Abstract method, which allows each GARCH-Type model to define its own parameter bounds
    @abstractmethod
    def param_bounds(self):
        raise NotImplementedError

    #Method to produce a one-step-ahead variance forecast, shared across the family
    @abstractmethod
    def forecast(self):
        raise NotImplementedError

    #Method to compute the negative Gaussian log-likelihood over a variance path
    def neg_log_likelihood(self, params, returns):
        variance = self._variance_recursion(params, returns)
        if np.any(variance <= 0) or np.any(~np.isfinite(variance)):
            return np.inf
        log_likelihood = -0.5 * np.sum(
            np.log(2 * np.pi) + np.log(variance) + returns**2 / variance
        )
        return -log_likelihood
    
    #Method to fit the model via maximum likelihood gor all GARCG tyoe models
    def fit(self, series):
        returns = np.asarray(series)

        x0 = self._initial_params(returns)
        bounds = self._param_bounds()

        result = minimize(
            self._neg_log_likelihood, x0, args=(returns,),
            method="L-BFGS-B", bounds=bounds
        )

        self.params = result.x
        self.series = returns
        self.fitted_values = self._variance_recursion(self.params, returns)
        self.fit_start = 0
        self.fit_end = len(returns) - 1
        self.is_fitted = True
        return result

    