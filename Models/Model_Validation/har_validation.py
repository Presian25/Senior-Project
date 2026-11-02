from Models.HAR.har import HAR
from Models.Model_Validation.base_har_validation import (
    HARValidation, target_i, t_means, prev_means)
import numpy as np


class HARValidator(HARValidation):
    name = "HAR"
    model_cls = HAR
    roles = ("rv",)

    def design(self, d):
        rv = d["rv"]
        t = target_i(len(rv))
        X = np.column_stack([np.ones(len(t))] + t_means(rv, t))
        return X, rv[t]

    def next_x(self, d):
        return [1.0] + prev_means(d["rv"])


if __name__ == "__main__":
    HARValidator().run()