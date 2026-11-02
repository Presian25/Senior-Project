from Models.HAR.har_cj import HAR_CJ
from Models.Model_Validation.base_har_validation import (
    HARValidation, WINDOWS,
    target_i , t_means, prev_means,
)
import numpy as np


def components(rv, bv):
    return np.minimum(rv, bv), np.maximum(rv - bv, 0.0)    # C, J


class HARCJValidator(HARValidation):
    name = "HAR_CJ"
    model_cls = HAR_CJ
    roles = ("rv", "bv")

    def design(self, d):
        rv = d["rv"]
        C, J = components(rv, d["bv"])
        t = target_i(len(rv))
        X = np.column_stack([np.ones(len(t))] + t_means(C, t) + [J[t - 1]])
        return X, rv[t]

    def next_x(self, d):
        C, J = components(d["rv"], d["bv"])
        return [1.0] + prev_means(C) + [J[-1]]

    def param_names(self):
        return ["const"] + [f"beta_C_{w}" for w in WINDOWS] + ["beta_J"]

    def model_kwargs(self, cols):
        return {"rv_col": cols["rv"], "bv_col": cols["bv"], "windows": WINDOWS}


if __name__ == "__main__":
    HARCJValidator().run()