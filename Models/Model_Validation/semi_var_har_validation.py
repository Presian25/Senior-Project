from Models.HAR.semi_var_har import SemiVarHAR
from Models.Model_Validation.base_har_validation import (
    HARValidation, COLS,
    target_i,
)
import numpy as np

SV_WINDOWS = (5, 22)     # this model has no window-1 term: the semivariances take its place

# Register the extra data roles without touching the base file.
COLS["rv_pos"] = {label: "RV_pos" for label in COLS["rv"]}
COLS["rv_neg"] = {label: "RV_neg" for label in COLS["rv"]}


def window_means(x, t):
    """mean(x[t-w:t]) for every target index t and each of this model's windows."""
    c = np.concatenate([[0.0], np.cumsum(np.asarray(x, dtype=float))])
    return [(c[t] - c[t - w]) / w for w in SV_WINDOWS]


class SemiVarHARValidator(HARValidation):
    name = "SemiVarHAR"
    model_cls = SemiVarHAR
    roles = ("rv", "rv_pos", "rv_neg")

    def design(self, d):
        rv, pos, neg = d["rv"], d["rv_pos"], d["rv_neg"]
        t = target_i(len(rv))           # first target has 22 rows of history
        X = np.column_stack([np.ones(len(t)), pos[t - 1], neg[t - 1]] + window_means(rv, t))
        return X, rv[t]

    def next_x(self, d):
        rv, pos, neg = d["rv"], d["rv_pos"], d["rv_neg"]
        return [1.0, pos[-1], neg[-1]] + [rv[-w:].mean() for w in SV_WINDOWS]

    def param_names(self):
        return ["const", "beta_pos", "beta_neg"] + [f"beta_{w}" for w in SV_WINDOWS]

    def model_kwargs(self, cols):
        return {"rv_col": cols["rv"], "rv_pos_col": cols["rv_pos"],
                "rv_neg_col": cols["rv_neg"], "windows": SV_WINDOWS}


if __name__ == "__main__":
    SemiVarHARValidator().run()