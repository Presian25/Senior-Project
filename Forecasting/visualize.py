"""
Draws the charts from the tables and the log that evaluate.py wrote. Nothing is calculated here.
Run with: python -m Forecasting.visualize
Charts: pies (error types), rolling (MAE and corr over time), regimes (MAE and QLIKE in low/mid/high volatility).
They are saved in Data/Results/tables/plots/<chart>/, one png per asset.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Forecasting.config import TABLE_DIR

ASSETS = None                    #None = all assets, or a list like ["SPY", "BTCUSDT"]
FREQS = ["1h", "1d"]
TARGETS = ["abs"]
MODELS = None                    #None = all models, or a list like ["GARCH_1_1", "HAR"]
LOSS = "MSE"                     #loss split in the pies: MAE, MSE or QLIKE

LOG_DIR = TABLE_DIR / "log"
PLOT_DIR = TABLE_DIR / "plots"
ALL_MODELS = ["GARCH_1_1", "GJR_GARCH_1_1", "EGARCH_1_1", "ARIMA", "SARIMA", "ARFIMA", "HAR", "LogHAR", "HAR_CJ", "SemiVarHAR"]


#Method to read one csv (the log folder first, then the tables folder), every file is only read once
@lru_cache(maxsize=None)
def read(name):
    path = LOG_DIR / name
    if not path.exists():
        path = TABLE_DIR / name
    return pd.read_csv(path)


#Method to keep the rows of one asset / frequency / target
def pick(df, asset, freq, target):
    return df[(df["asset"] == asset) & (df["freq"] == freq) & (df["target"] == target)].copy()


#Method to get the models to draw, in a fixed order
def models_to_draw(df):
    wanted = MODELS if MODELS is not None else ALL_MODELS
    return [m for m in wanted if m in set(df["model"])]


#Method to give every model the same colour in every chart (NAIVE is gray and dotted)
def style(model):
    if model == "NAIVE":
        return {"color": "gray", "linestyle": ":", "linewidth": 1.5}
    return {"color": plt.get_cmap("tab10")(ALL_MODELS.index(model) % 10), "linewidth": 1.2}


#Method to save the figure in plots/<chart>/
def save(fig, chart, name):
    folder = PLOT_DIR / chart
    folder.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(folder / f"{name}.png", dpi=130)
    plt.close(fig)


#Method to draw one pie per model with the three error types (jump, calm, normal)
def pies(asset, freq, target):
    df = pick(read("shocks_all.csv"), asset, freq, target)
    df = df[df["model"].isin(models_to_draw(df))]
    cols = [f"{LOSS}_jump_pct", f"{LOSS}_calm_pct", f"{LOSS}_normal_pct"]
    df = df.dropna(subset=cols)
    if df.empty:
        return
    fig, axes = plt.subplots(2, 5, figsize=(14, 6.5))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, row) in zip(axes.ravel(), df.iterrows()):
        ax.pie(row[cols].astype(float), colors=["#d9534f", "#4a7ebb", "#cccccc"], autopct="%.0f%%", textprops={"fontsize": 8})
        ax.set_title(row["model"], fontsize=9)
    fig.legend(["jump (not reacting)", "calm (not calming down)", "normal"], loc="lower center", ncol=3)
    fig.suptitle(f"Where the {LOSS} comes from, {asset} {freq} {target}")
    save(fig, "pies", f"{asset}_{freq}_{target}")


#Method to draw the rolling MAE and corr of every model over time
def rolling(asset, freq, target):
    df = pick(read("rolling_all.csv"), asset, freq, target)
    if df.empty:
        return
    df["date"] = pd.to_datetime(df["date"], utc=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True)
    for ax, col in zip(axes, ["roll_MAE", "roll_corr"]):
        for m in models_to_draw(df) + ["NAIVE"]:
            g = df[df["model"] == m]
            ax.plot(g["date"], g[col], label=m, **style(m))
        ax.set_title(col, fontsize=10)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"Rolling scores, {asset} {freq} {target}")
    save(fig, "rolling", f"{asset}_{freq}_{target}")


#Method to draw the MAE and QLIKE of every model in the low, mid and high volatility regime (bars)
def regimes(asset, freq, target):
    df = pick(read("breakdown_all.csv"), asset, freq, target)
    df = df[df["group_type"] == "regime"]
    models = models_to_draw(df)
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    width = 0.8 / len(models)
    for ax, col in zip(axes, ["MAE", "QLIKE"]):
        for i, m in enumerate(models):
            g = df[df["model"] == m].set_index("group").reindex(["low", "mid", "high"])
            ax.bar(np.arange(3) + i * width, g[col], width=width, label=m, color=style(m)["color"])
        ax.set_xticks(np.arange(3) + 0.4 - width / 2)
        ax.set_xticklabels(["low", "mid", "high"])
        ax.set_xlabel("volatility regime (from the trailing volatility)")
        ax.set_title(col, fontsize=10)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"Performance by regime, {asset} {freq} {target}")
    save(fig, "regimes", f"{asset}_{freq}_{target}")


def main():
    for freq in FREQS:
        for target in TARGETS:
            assets = ASSETS if ASSETS is not None else sorted(pick_assets(freq, target))
            for asset in assets:
                pies(asset, freq, target)
                rolling(asset, freq, target)
                regimes(asset, freq, target)
            print("drawn", freq, target, len(assets), "assets")
    print("plots saved to", PLOT_DIR)


#Method to list the assets that have results for this frequency and target
def pick_assets(freq, target):
    df = read("rolling_all.csv")
    return df[(df["freq"] == freq) & (df["target"] == target)]["asset"].unique()


if __name__ == "__main__":
    main()
