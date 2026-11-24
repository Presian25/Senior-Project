"""
Class used for Validation for the HAR model
"""

import numpy as np

from Models.HAR.har import HAR
from Models.Model_Validation.base_har_validation import (
    HARValidation, target_i, t_means, prev_means)


class HARValidator(HARValidation):
    name = "HAR"
    model_cls = HAR
    roles = ("rv",)

    #Method to build the matrix
    def design(self, d):
        rv = d["rv"]
        t = target_i(len(rv))
        X = np.column_stack([np.ones(len(t))] + t_means(rv, t))
        return X, rv[t]

    #Method for next step vector
    def next_x(self, d):
        rv = d["rv"]
        return [1.0] + prev_means(rv)


if __name__ == "__main__":
    HARValidator().run()