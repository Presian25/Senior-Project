"""
The user application of the project, it utilizes a streamlit dashboard.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from App import asset_info, data_handling, methods

st.set_page_config(page_title="Volatility forecasting tool", layout="wide")

BORDER_RED = "#ff1a1a"
st.markdown(f"<style>[data-testid='stVerticalBlockBorderWrapper'], [data-testid='stVerticalBlockBorderWrapper'], [data-testid='stVerticalBlock'], [data-testid='stExpander'] details {{border-color: {BORDER_RED} !important;}}</style>",
            unsafe_allow_html=True)

BLEND_NAMES = {"BLEND_OPT": "Blend by QLIKE optimization",
              "BLEND_REG": "Blend by regime",
              "BLEND_HIER2G": "Blend by asset class"}


#Method to map blend names
def bl_names(name):
    return BLEND_NAMES.get(name, name)


#Method to rename the models and the columns of a table before it is shown
def set_table_headers(table):
    table = table.rename(index=BLEND_NAMES)
    if "model" in table.columns:
        table["model"] = table["model"].map(bl_names)
    table = table.rename(columns={"model": "Model", "forec_pct": "Forecast (%)", "abs_err": "Absolute error (%)",
                                  "corr": "Correlation", "MAE": "MAE (%)"})
    return table.rename_axis(index={"place": "Rank"})

#Method to load the data of one asset once and keep it in memory
@st.cache_data(show_spinner="Loading data")
def get_data(freq, asset):
    item = data_handling.list_assets(freq)[asset]
    return data_handling.load_data(item)


#Method to show the welcome page before the first Apply
def welc_menu():
    st.title("Volatility Forecast Analysis")
    st.markdown("""
Welcome! This dashboard shows **volatility forecasts** for 25 assets across
equities, ETFs, crypto, FX and futures. The forecasts are produced by 10 forecasting models and 3 custom blends of them.

**How to use the dashboard**
1. Choose the forecast frequency (hourly or daily) and an asset in the **Forecast filters** on the left.
2. Pick a date and, for hourly data, a time.
3. Press **Apply**.

**What you get**
- The volatility regime at that time and the **recommended model**, chosen based on the models'
  performance on the most recent periods before it.
- The scores behind the recommendation and whether the best model is significantly better than the next one.
- A chart of the forecasts against the realized volatility, and the forecast of every model.

Every forecast is made at the start of a period using only earlier data, so you can check afterwards
what really happened with **Reveal what happened**. Forecast frequency means the length of the
forecast period (one hour or one day).
""")
    with st.expander("Asset guide: what each asset is"):
        guide = pd.DataFrame(asset_info.ASSETS, index=["Name", "Class", "Description"]).T
        guide.index.name = "Ticker"
        st.dataframe(guide.sort_values(["Class", "Name"]))


#Method to draw the chosen models against the realized |return|
def draw_chart(forec, rr, ts, models, freq, rev, base):
    bef = 120 if freq == "1h" else 60
    aft = 0
    if rev:
        aft = 24 if freq == "1h" else 10
    forec_period, rr_period = methods.crop_periods(forec, rr, ts, bef, aft)
    if not rev:
        rr_period = rr_period[rr_period.index < ts]          

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rr_period.index, y=rr_period, name="Realized |return| (%)",
                             mode="lines", line=dict(color="#9a9a9a", width=1)))
    for model in models:
        fig.add_trace(go.Scatter(x=forec_period.index, y=forec_period[model], name=f"{bl_names(model)} forecast (%)",
                                 mode="lines", line=dict(width=2.5)))
    if base and "GARCH_1_1" not in models:
        fig.add_trace(go.Scatter(x=forec_period.index, y=forec_period["GARCH_1_1"], name="GARCH_1_1 forecast (%)",
                                 mode="lines", line=dict(width=1.5, dash="dot")))
    if len(models) > 0:
        fig.add_trace(go.Scatter(x=[ts], y=[forec.loc[ts, models[0]]], name="Chosen time", mode="markers",
                                 marker=dict(size=12, symbol="diamond", color="#d62728")))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="% per period",
                      legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig)


#Method to show the top group of the dashboard
def show_header(forec, naive, ts, freq, top):
    with st.container(border=True):
        left, middle, right = st.columns(3)
        with left.container(border=True):
            st.metric("Volatility regime", methods.label_reg(naive, freq, ts).capitalize())
        with middle.container(border=True):
            st.metric("Recommended model", bl_names(top[0]))
        with right.container(border=True):
            st.metric("Recommended model's forecast (%)", f"{forec.loc[ts, top[0]]:.3f}")


#Method to show the the recommended models section
def show_recommended(rec):
    with st.container(border=True):
        st.markdown("#### Recommended models")
        st.caption(f"Lowest QLIKE over the last {rec['num_obs']} periods before this time.")
        for i in range(len(rec["models"])):
            if rec["significant"][i]:
                st.write(f"{i + 1}. **{bl_names(rec['models'][i])}** - significantly better than the next model")
            else:
                st.write(f"{i + 1}. **{bl_names(rec['models'][i])}** - not significantly better than the next model")
        scores = rec["scores"]
        with st.expander("Scores behind the recommendation (lower QLIKE and MAE are better, higher correlation is better)"):
            st.dataframe(set_table_headers(scores).style.format({"QLIKE": "{:.3f}", "Correlation": "{:.3f}", "MAE (%)": "{:.3f}"}),
                         height=36 * len(scores) + 40)


#Method to show everything for the applied filters
def show_result(forec, rr, naive, query, name):
    ts, freq = query["ts"], query["freq"]
    st.subheader(f"{name} - forecast for {ts:%Y-%m-%d %H:%M} UTC")
    st.caption("Forecast made at the start of this period, using only data from before it.")
    try:
        rec = methods.recommend_model(forec, rr, ts, query["window"], query["top_n"])
    except methods.MinPeriodError:
        rec = None

    if rec is None:
        st.warning(f"Not enough earlier periods to score the models (at least {methods.MIN_PER} needed).")
        top = ["GARCH_1_1"]
        chart_options = top
    else:
        top = rec["models"]
        chart_options = list(rec["scores"]["model"].head(3))

    show_header(forec, naive, ts, freq, top)
    if rec is not None:
        show_recommended(rec)

    with st.container(border=True):
        st.markdown("#### Forecast against realized volatility")
        key = f"{query['asset']}_{query['freq']}_{ts}_{query['window']}"
        left, middle, right = st.columns([3, 1, 1])
        chosen = left.multiselect("Models on the chart (the three best)", chart_options, format_func=bl_names,
                                  default=chart_options[:1], key=f"chart_{key}")
        base = middle.checkbox("Add GARCH_1_1", key="baseline")
        rev = right.checkbox("Reveal what happened", key="reveal")
        draw_chart(forec, rr, ts, chosen, freq, rev, base)

    table, realized = methods.get_forec(forec, rr, naive, ts)
    with st.container(border=True):
        st.markdown("#### Forecast of every model for this period")
        st.caption("Standard deviation of the return, in %.")
        if rev:
            st.write(f"Realized |return| in this period: **{realized:.3f}%**")
            shown = set_table_headers(table)
            shown.index.name = "Model"
            st.dataframe(shown.style.format("{:.3f}"))
        else:
            shown = set_table_headers(table[["forec_pct"]])
            shown.index.name = "Model"
            st.dataframe(shown.style.format("{:.3f}"))

    item = data_handling.list_assets(freq)[query["asset"]]
    longrun = data_handling.load_period(item)
    if longrun is not None:
        with st.expander("Long-run ranking of this asset (whole test period, from the saved evaluation)"):
            st.dataframe(set_table_headers(longrun).style.format({"QLIKE": "{:.3f}", "Correlation": "{:.3f}"}))


#Sidebar: the forecast filters
st.sidebar.header("Forecast filters")
label = st.sidebar.radio("Forecast frequency (one step ahead)", ["Hourly", "Daily"])
freq = "1h" if label == "Hourly" else "1d"
try:
    assets = sorted(data_handling.list_assets(freq), key=asset_info.sort_key)
except data_handling.LoadError as error:
    st.error(str(error))
    st.stop()

asset = st.sidebar.selectbox("Asset", assets, format_func=asset_info.label)
st.sidebar.caption(asset_info.describe(asset))
try:
    forec, rr, naive = get_data(freq, asset)
except data_handling.LoadError as error:
    st.error(str(error))
    st.stop()

first, last = forec.index[0], forec.index[-1]
date = st.sidebar.date_input("Date", value=last.date(), min_value=first.date(), max_value=last.date())
st.sidebar.caption(f"Forecasts available from {first:%Y-%m-%d} to {last:%Y-%m-%d} (UTC).")
times = methods.get_hs(forec, date)
ts = None
if len(times) == 0:
    nearest = methods.get_closest_per(forec, pd.Timestamp(date, tz="UTC") - pd.Timedelta(seconds=1))
    hint = ""
    if nearest is not None:
        hint = f" Closest earlier time: {nearest:%Y-%m-%d %H:%M} UTC."
    st.sidebar.warning(f"No forecast on {date} (market closed or no data)." + hint)
elif freq == "1h":
    ts = st.sidebar.selectbox("Time (UTC)", times, format_func=lambda t: f"{t:%H:%M}")
else:
    ts = times[0]

top_window = 1500 if freq == "1h" else 250
window = st.sidebar.slider("Recent window for the recommendation (periods)", methods.MIN_PER, top_window,
                           methods.PREV[freq], key=f"window_{freq}")
top_n = st.sidebar.number_input("Number of models to recommend", min_value=1, max_value=5, value=1, step=1)
current = {"asset": asset, "freq": freq, "ts": ts, "window": window, "top_n": top_n}
if st.sidebar.button("Apply", disabled=ts is None, type="primary"):
    st.session_state["query"] = current

st.sidebar.divider()
st.sidebar.multiselect("My assets (watchlist)", assets, format_func=asset_info.label, key="watchlist")

explorer, watchlist = st.tabs(["Forecast explorer", "My assets"])

with explorer:
    query = st.session_state.get("query")
    if query is None:
        welc_menu()
    else:
        if query != current:
            st.info("The filters were changed. Press Apply to update the results.")
        forec_q, rr_q, naive_q = get_data(query["freq"], query["asset"])
        show_result(forec_q, rr_q, naive_q, query, asset_info.label(query["asset"]))

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
                rows.append({"asset": asset_info.label(name), "recommended": "no data"})
                continue
            rows.append({"asset": asset_info.label(name), **methods.summary(forec_a, rr_a, naive_a, freq, window)})
        shown = pd.DataFrame(rows).set_index("asset")
        shown["recommended"] = shown["recommended"].map(bl_names)
        shown = shown.rename(columns={"recommended": "Recommended model", "recent QLIKE": "Recent QLIKE",
                                      "last forecast (%)": "Last forecast (%)", "regime": "Regime",
                                      "last period": "Last period"})
        shown.index.name = "Asset"
        st.dataframe(shown)