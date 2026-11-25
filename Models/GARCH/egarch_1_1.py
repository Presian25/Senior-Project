"""
Model's formula:
EGARCH(1,1): o + b*log(sigma_{t-1}^2)
                 + a*(|eps_{t-1}|/sigma_{t-1} - sqrt(2/pi))
                 + g*(eps_{t-1}/sigma_{t-1})

The model specializes at capturing volatility shocks
"""

import numpy as np

from Models.GARCH.base_garch_model import BaseGARCHModel
VAR_TH = 20.0
Z_CAP = 5.0 

class EGARCH_1_1(BaseGARCHModel):

    #Method to eval variance recursion
    def variance_recursion(self, params, returns):
        o, a, b, g = params           #omega, alpha, beta
        period = len(returns)
        log_var = np.zeros(period)
        log_var[0] = np.clip(np.log(np.var(returns)), -VAR_TH, VAR_TH)

        const_term = np.sqrt(2/np.pi)

        for t in range(1, period):
            prev_std = np.sqrt(np.exp(log_var[t-1]))
            stand_returns = np.clip(returns[t-1]/prev_std, -Z_CAP, Z_CAP)
            temp_log_var = (o + b*log_var[t-1] + a*(np.abs(stand_returns) - const_term) + g * stand_returns)
            log_var[t] = np.clip(temp_log_var, -VAR_TH, VAR_TH)
        return np.exp(log_var)

    #Method to set a the starting guess for the parameters
    def initial_params(self, returns):
        sample_var = np.var(returns)
        a_0 = 0.1
        b_0 = 0.90
        o_0 = np.log(sample_var) * (1-b_0)
        g_0 = 0.05
        return [o_0, a_0, b_0, g_0]

    #Method to set parameter limits
    def param_bounds(self):
        return [(-5.0, 5.0), (0.0, 5.0), (-0.999, 0.999), (-5.0, 5.0),]

    #Method to produce a variance forecast at time t for time t+1
    def forecast(self):
        self._check_is_fitted()
        o, a, b, g = self.params
        prev_return = self.series[-1]
        prev_variance = self.fitted_values[-1]
        prev_std = np.sqrt(prev_variance)
        stand = np.clip(prev_return / prev_std, -Z_CAP, Z_CAP)
        const_term = np.sqrt(2/np.pi)

        forecast = (o + b* np.log(prev_variance) + a*(np.abs(stand) - const_term) + g * stand)
        forecast = np.clip(forecast, -VAR_TH, VAR_TH)
        return np.exp(forecast)

    