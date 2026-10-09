"""
This is the base class for all garch type models. It inherits from the basic 
Volatility model class, implements the methods for fit and forecasting, but also
provides new abstract ones, necessary for each specific algorithm. Uniquely for this class models,
there are shared methods for residual recursion, Gaussian log-likelihood and MLE fitting
"""

import numpy as np
from scipy.optimize import minimize
from abc import abstractmethod

from Models.vol_model import VolatilityModel

#Base calss for the ARIMA models
class BaseARIMAModel(VolatilityModel):

    #Abstract method to convert raw series into ARIMA fitting input
    @abstractmethod
    def convert(self,series):
        raise NotImplementedError

    #Abstract method to invert the conversion in order to bring a forecast back to original scale
    @abstractmethod
    def invert_conversion(self, last_vals, forecast_val):
        raise NotImplementedError


    #Abstract method to set starting params
    @abstractmethod
    def set_initial_params(self, converted_series):
        raise NotImplementedError

    #Abstract method to limit params
    @abstractmethod
    def param_bounds(self):
        raise NotImplementedError

    #Abstract method to return num of AR and MA terms the specific model has
    @abstractmethod
    def param_num(self):
        raise NotImplementedError

    #Method to compute fitted vals and residuals recursively
    def recursion(self, params, series): 
        p, q = self.param_num()
        r = params[0]
        x = params[1:1+p]
        y = params[1+p:1+p+q]

        period = len(series)
        fit = np.zeros(period)
        residuals = np.zeros(period)

        for t in range(period):
            ar = sum(x[i] * series[t-1-i] for i in range(p) if t-1-i >= 0)
            ma = sum(y[j] * residuals[t-1-j] for j in range(q) if t-1-j >= 0)
            fit[t] = r + ar + ma
            residuals[t] = series[t] - fit[t]

        return fit, residuals

    #Method to compute the negative Gaussian log-likelihood
    def neg_log_likelihood(self, params, series):
        x, residuals = self.recursion(params, series)
        residual_var = np.var(residuals)
        if residual_var <= 0 or not np.isfinite(residual_var):
            return np.inf
        period = len(residuals)
        log_likelihood = -0.5 * period * (np.log(2 * np.pi) + np.log(residual_var)) \
             - 0.5 * np.sum(residuals**2) / residual_var
        return - log_likelihood

    #Method to fit the model via MLE function
    def fit(self, returns):
        init_set = np.asarray(returns)
        series = self.convert(init_set)

        params_start = self.set_initial_params(series)
        bounds = self.param_bounds()

        result = minimize(
            self.neg_log_likelihood, params_start, args=(series,),
            method="L-BFGS-B", bounds=bounds
        )

        self.params = result.x
        self.series = init_set
        self.converted_series = series
        self.fitted_values, self.residuals = self.recursion(self.params, series)
        self.fit_start = 0
        self.fit_end = len(init_set) -1
        self.is_fitted = True
        return result


    #Method to produce a one-step-ahead forecast
    def forecast(self):
        self._check_is_fitted()
        p, q = self.param_num()
        c = self.params[0]
        x = self.params[1:1+p]
        y = self.params[1+p:1+p+q]

        dataset = self.converted_series
        residuals = self.residuals

        ar_term = sum(x[i] * dataset[-1-i] for i in range(p))
        ma_term = sum(y[j] * residuals[-1-j] for j in range(q))
        forecast_val = c + ar_term + ma_term

        return self.invert_conversion(self.series, forecast_val)
