from Models.HAR.log_har import LogHAR
from Models.Model_Validation.base_har_validation import (
    HARValidation,
    target_i as target_index,
    t_means as trailing_means,
    prev_means as last_means,
)
import numpy as np


class LogHARValidator(HARValidation):
    name = "LogHAR"
    model_cls = LogHAR
    roles = ("rv",)

    def design(self, d):
        rv = d["rv"]
        t = target_index(len(rv))
        with np.errstate(divide="ignore", invalid="ignore"):
            cols = [np.log(m) for m in trailing_means(rv, t)]
            y = np.log(rv[t])
        X = np.column_stack([np.ones(len(t))] + cols)
        keep = np.isfinite(X).all(axis=1) & np.isfinite(y)     # drop rows with log(0) as it raises problems
        return X[keep], y[keep]

    def next_x(self, d):
        rv = d["rv"]
        floor = rv[rv > 0].min()         
        return [1.0] + [np.log(max(m, floor)) for m in last_means(rv)]

    #exp the prediction
    def transform(self, pred):
        return float(np.exp(pred))        


if __name__ == "__main__":
    LogHARValidator().run()