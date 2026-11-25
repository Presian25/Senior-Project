"""
Model's formula:
GARCH(1,1): sigma_t^2 = o + a* eps_{t-1}^2 + b * sigma_{t-1}^2

This is the most common GARCH-type model, uses one lagged squared
return and variance term 
"""

import numpy as np

from Models.GARCH.base_garch_model import BaseGARCHModel


class GARCH_1_1(BaseGARCHModel):

    #Method to eval variance recursion
    def variance_recursion(self, params, returns):
        o, a, b = params           #omega, alpha, beta
        period = len(returns)
        variance = np.zeros(period)
        variance[0] = np.var(returns)

        for t in range(1, period):
            variance[t] = o + a * returns[t-1]**2 + b * variance[t-1]

        return variance

    #Method to set a the starting guess for the parameters
    def initial_params(self, returns):
        sample_var = np.var(returns)
        a_0 = 0.05
        b_0 = 0.90
        o_0 = sample_var * (1 - a_0 - b_0)
        return [o_0, a_0, b_0]

    #Method to set parameter limits
    def param_bounds(self):
        return [(1e-8, None), (0.0, 1.0), (0.0, 1.0)]

    #Method to produce a variance forecast at time t for time t+1
    def forecast(self):
        self._check_is_fitted()
        o, a, b = self.params
        prev_return = self.series[-1]
        prev_variance = self.fitted_values[-1]
        return o + a * prev_return**2 + b * prev_variance
