import warnings
import pandas as pd

from Blending.utilities import BLEND_DIR, BLEND_TABLE_DIR, BLEND_NAMES
from Forecasting.config import get_assets
from Forecasting.evaluate import (load_forec, add_ref_cols, prep_targets, score, qlike_var, dm_stat,
                                  FLOOR_FRACTION, REFS)

FREQS = None
ASSETS = None
OUT_DIR = BLEND_TABLE_DIR
SAMPLES = {"full": 0.0, "last30": 0.7}


#Method to read the blends a dataset
def load_b(item):
    cols = {}
    for name in BLEND_NAMES:
        path = BLEND_DIR / item["freq"] / item["asset"] / f"{name}.csv"
        if path.exists():
            cols[name] = pd.read_csv(path, index_col="date", parse_dates=["date"])["sigma_hat_pct"]
    return pd.DataFrame(cols)


#Method to compare every blend
def compare(tag, y, Sc, res, singles, blends):
    res = res.set_index("model")
    best_q = res.loc[singles, "QLIKE"].idxmin()
    best_c = res.loc[singles, "corr"].idxmax()
    rows = []
    for b in blends:
        loss_b = qlike_var(y, Sc[b])
        rows.append({**tag, "blend": b,
                     "QLIKE": res.loc[b, "QLIKE"], "best_single": best_q, "best_single_QLIKE": res.loc[best_q, "QLIKE"],
                     "ENS_MEDIAN_QLIKE": res.loc["ENS_MEDIAN", "QLIKE"], "NAIVE_QLIKE": res.loc["NAIVE", "QLIKE"],
                     "DM_vs_best": dm_stat(loss_b, qlike_var(y, Sc[best_q])),
                     "DM_vs_ENS_MEDIAN": dm_stat(loss_b, qlike_var(y, Sc["ENS_MEDIAN"])),
                     "corr": res.loc[b, "corr"], "best_single_corr": best_c, "best_single_corr_value": res.loc[best_c, "corr"]})
    return rows


def main():
    warnings.filterwarnings("ignore")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    m_rows, comp_rows = [], []

    for item in get_assets(FREQS, ASSETS):
        set = load_forec(item)
        blend = load_b(item)
        if set.empty or blend.empty:
            continue
        set, singles = add_ref_cols(set, item)
        set = set.join(blend, how="inner").dropna()
        blends = list(blend.columns)

        for sample, cut in SAMPLES.items():
            part = set.iloc[int(len(set) * cut):]
            for target, y, Sc, scale, y_hist in prep_targets(item, part):
                tag = {"sample": sample, "asset": item["asset"], "class": item["cls"], "freq": item["freq"], "target": target}
                Sc = Sc.clip(lower=FLOOR_FRACTION * Sc["NAIVE"].median())

                rows = []
                for m in Sc.columns:
                    kind = "blend" if m in blends else ("reference" if m in REFS else "model")
                    rows.append({**tag, "model": m, "kind": kind, **score(y, Sc[m], scale, y_hist)})
                res = pd.DataFrame(rows)
                naive = res[res["model"] == "NAIVE"].iloc[0]
                res["skill_MAE"] = 1 - res["MAE"] / naive["MAE"]
                res["skill_QLIKE"] = naive["QLIKE"] - res["QLIKE"]
                m_rows.append(res)
                comp_rows += compare(tag, y, Sc, res, singles, blends)
        print("done", item["asset"], item["freq"], flush=True)

    metrics = pd.concat(m_rows, ignore_index=True)
    vs = pd.DataFrame(comp_rows)
    metrics.to_csv(OUT_DIR / "blend_metrics.csv", index=False)
    vs.to_csv(OUT_DIR / "blend_vs_best.csv", index=False)

    ranked = metrics[metrics["kind"] != "reference"].copy()
    keys = ["sample", "target", "freq", "asset"]
    ranked["rank_QLIKE"] = ranked.groupby(keys)["QLIKE"].rank()
    ranked["rank_corr"] = ranked.groupby(keys)["corr"].rank(ascending=False)
    columns = ["rank_QLIKE", "rank_corr", "skill_QLIKE", "corr"]
    how = {"rank_QLIKE": "mean", "rank_corr": "mean", "skill_QLIKE": "median", "corr": "median"}
    ranking = ranked.groupby(["sample", "target", "freq", "model"])[columns].agg(how).round(3)

    vs["beats_best_QLIKE"] = vs["QLIKE"] < vs["best_single_QLIKE"]
    vs["beats_ENS_MEDIAN"] = vs["QLIKE"] < vs["ENS_MEDIAN_QLIKE"]
    vs["beats_best_corr"] = vs["corr"] > vs["best_single_corr_value"]
    vs["sig_better_than_best"] = vs["DM_vs_best"] < -1.96
    vs["sig_worse_than_best"] = vs["DM_vs_best"] > 1.96
    group = ["sample", "target", "freq", "blend"]
    wins = vs.groupby(group)[["beats_best_QLIKE", "beats_ENS_MEDIAN", "beats_best_corr",
                                  "sig_better_than_best", "sig_worse_than_best"]].sum()
    wins["assets"] = vs.groupby(group).size()
    ranking = ranking.join(wins.rename_axis(index={"blend": "model"}), how="left")
    ranking.to_csv(OUT_DIR / "blend_ranking.csv")

    by_class = ranked.groupby(["sample", "target", "freq", "class", "model"])[columns].agg(how).round(3)
    by_class.to_csv(OUT_DIR / "blend_by_class.csv")

    for sample in SAMPLES:
        print("=== sample:", sample, "(abs target, rank 1 = best)")
        print(ranking.loc[sample].loc["abs"].sort_values(["freq", "rank_QLIKE"]).to_string())
    print("tables written to", OUT_DIR)


if __name__ == "__main__":
    main()