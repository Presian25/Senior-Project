"""
The user application of the project, it uutilizes a streamlit dashboard
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from App import data_handling, methods

st.set_page_config(page_title="Volatility forecasting tool", layout="wide")


#Method to load the data of one asset once and keep it in memory
@st.cache_data(show_spinner="Loading data")
def get_data(freq, asset):
    item = data_handling.list_assets(freq)[asset]
    return data_handling.load_data(item)


#Method to average the variance forecasts of the recommended models (equal weights)
def average_top(forec, names):
    return (forec[names] ** 2).mean(axis=1) ** 0.5


#Method to draw the forecast of the chosen model against the realized |return|
def draw_chart(forec, rr, ts, top, freq, reveal, baseline):
    before = 120 if freq == "1h" else 60
    after = 0
    if reveal:
        after = 24 if freq == "1h" else 10
    forec_part, rr_part = methods.crop_periods(forec, rr, ts, before, after)
    if not reveal:
        rr_part = rr_part[rr_part.index < ts]
    model = top[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rr_part.index, y=rr_part, name="Realized |return| (%)",
                             mode="lines", line=dict(color="#9a9a9a", width=1)))
    fig.add_trace(go.Scatter(x=forec_part.index, y=forec_part[model], name=f"{model} forecast (%)",
                             mode="lines", line=dict(width=2.5)))
    if len(top) > 1:
        fig.add_trace(go.Scatter(x=forec_part.index, y=average_top(forec_part, top),
                                 name=f"Average of top {len(top)} (%)", mode="lines",
                                 line=dict(width=2, dash="dash")))
    if baseline:
        fig.add_trace(go.Scatter(x=forec_part.index, y=forec_part["GARCH_1_1"], name="GARCH_1_1 forecast (%)",
                                 mode="lines", line=dict(width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=[ts], y=[forec.loc[ts, model]], name="Chosen time", mode="markers",
                             marker=dict(size=12, symbol="diamond", color="#d62728")))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="% per period",
                      legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig)


#Method to show everything for the chosen time
def show_result(forec, rr, naive, query, name):
    ts, freq = query["ts"], query["freq"]
    st.subheader(f"{name} - forecast for {ts:%Y-%m-%d %H:%M} UTC")
    st.caption("Forecast made at the start of this period, using only data from before it.")
    try:
        rec = methods.recommend_model(forec, rr, ts, query["window"], query["top_n"])
    except methods.MinPeriodError:
        rec = None

    left, middle, right = st.columns(3)
    left.metric("Volatility regime", methods.label_reg(naive, freq, ts))
    if rec is None:
        st.warning(f"Not enough earlier periods to score the models (at least {methods.MIN_PER} needed).")
        top = ["GARCH_1_1"]
    else:
        top = rec["models"]
        middle.metric("Recommended model", top[0])
    right.metric("Its forecast (%)", f"{forec.loc[ts, top[0]]:.3f}")

    if rec is not None:
        st.write(f"**Recommended models** (lowest QLIKE over the last {rec['num_obs']} periods before this time)")
        for i in range(len(top)):
            if rec["significant"][i]:
                st.write(f"{i + 1}. **{top[i]}** - significantly better than the next model")
            else:
                st.write(f"{i + 1}. **{top[i]}** - not significantly better than the next model")
        st.write("**Scores behind the recommendation** (lower QLIKE and MAE are better, higher correlation is better)")
        scores = rec["scores"]
        st.dataframe(scores.style.format({"QLIKE": "{:.3f}", "corr": "{:.3f}", "MAE": "{:.3f}"}),
                     height=36 * len(scores) + 40)

    reveal = st.checkbox("Reveal what happened", key="reveal")
    table, realized = methods.get_forec(forec, rr, naive, ts)
    st.write("**Forecast of every model for this period** (standard deviation of the return, in %)")
    if reveal:
        st.write(f"Realized |return| in this period: **{realized:.3f}%**")
        st.dataframe(table.style.format("{:.3f}"))
    else:
        st.dataframe(table[["forec_pct"]].style.format("{:.3f}"))

    baseline = st.checkbox("Compare with GARCH_1_1 on the chart", key="baseline")
    draw_chart(forec, rr, ts, top, freq, reveal, baseline)


#Sidebar: frequency, asset, window, number of models and watchlist
st.title("Volatility forecast explorer")
st.sidebar.header("Settings")
label = st.sidebar.radio("Forecast frequency (one step ahead)", ["Hourly", "Daily"])
freq = "1h" if label == "Hourly" else "1d"
try:
    assets = sorted(data_handling.list_assets(freq))
except data_handling.LoadError as error:
    st.error(str(error))
    st.stop()

asset = st.sidebar.selectbox("Asset", assets)
top_window = 1500 if freq == "1h" else 250
window = st.sidebar.slider("Recent window for the recommendation (periods)", methods.MIN_PER, top_window,
                           methods.PREV[freq], key=f"window_{freq}")
top_n = st.sidebar.number_input("Number of models to recommend", min_value=1, max_value=5, value=1, step=1)
st.sidebar.multiselect("My assets (watchlist)", assets, key="watchlist")

explorer, watchlist = st.tabs(["Forecast explorer", "My assets"])

with explorer:
    try:
        forec, rr, naive = get_data(freq, asset)
    except methods.LoadError as error:
        st.error(str(error))
        st.stop()

    first, last = forec.index[0], forec.index[-1]
    st.write(f"Forecasts are available from {first:%Y-%m-%d} to {last:%Y-%m-%d} (UTC). "
             "The first 1000 periods of each asset are only used to fit the first model.")
    date = st.date_input("Date", value=last.date(), min_value=first.date(), max_value=last.date())
    times = methods.get_hs(forec, date)
    ts = None
    if len(times) == 0:
        nearest = methods.get_closest_per(forec, pd.Timestamp(date, tz="UTC") - pd.Timedelta(seconds=1))
        hint = ""
        if nearest is not None:
            hint = f" The closest earlier time is {nearest:%Y-%m-%d %H:%M} UTC."
        st.warning(f"There is no forecast on {date} (market closed or no data)." + hint)
    elif freq == "1h":
        ts = st.selectbox("Time (UTC)", times, format_func=lambda t: f"{t:%H:%M}")
    else:
        ts = times[0]

    if st.button("Show forecasts", disabled=ts is None):
        st.session_state["query"] = {"asset": asset, "freq": freq, "ts": ts, "window": window, "top_n": top_n}

    query = st.session_state.get("query")
    if query and query["asset"] == asset and query["freq"] == freq:
        show_result(forec, rr, naive, query, asset)
    else:
        st.info("Choose a date and press the button to see the forecasts.")

    item = data_handling.list_assets(freq)[asset]
    longrun = data_handling.load_period(item)
    if longrun is not None:
        with st.expander("Long-run ranking of this asset (whole test period, from the saved evaluation)"):
            st.dataframe(longrun.style.format({"QLIKE": "{:.3f}", "corr": "{:.3f}"}))

with watchlist:
    chosen = st.session_state.get("watchlist", [])
    if len(chosen) == 0:
        st.info("Add assets to your watchlist in the sidebar to follow them here.")
    else:
        st.write("The recommendation is computed from the most recent periods of each asset.")
        rows = []
        for name in chosen:
            try:
                forec_a, rr_a, naive_a = get_data(freq, name)
            except data_handling.LoadError:
                rows.append({"asset": name, "recommended": "no data"})
                continue
            rows.append({"asset": name, **methods.summary(forec_a, rr_a, naive_a, freq, window)})
        st.dataframe(pd.DataFrame(rows).set_index("asset"))