import streamlit as st

from src.config import Settings
from src.visualization import dashboard_components as ui

ws, settings = ui.get_ws(), ui.get_settings()
ui.setup_page("Settings", "Every setting changes a real calculation. Defaults are documented in docs/methodology.md", filters=False)
s = settings
with st.form("settings"):
    st.markdown("**Forecasting**")
    c = st.columns(4)
    horizon = c[0].select_slider("Forecast horizon (weeks)", [6, 7, 8], value=min(max(s.horizon_weeks, 6), 8))
    folds = c[1].number_input("Backtest folds", 1, 8, s.backtest_folds)
    minh = c[2].number_input("Minimum history (weeks)", 12, 104, s.min_history_weeks)
    maxs = c[3].number_input("Max SKUs forecast", 10, 5000, s.max_skus)
    lvl = st.slider("Prediction interval level", 0.5, 0.95, float(s.interval_level), 0.05)
    st.markdown("**Data processing**")
    c = st.columns(3)
    zero = c[0].checkbox("Treat weeks with no sales rows as zero demand (between a SKU's first sale and the last week)", s.missing_weeks_as_zero)
    zoo = c[1].checkbox("Treat blank on-order units as 0", s.assume_zero_on_order)
    dead = c[2].number_input("Dead-stock window (days without sales)", 14, 720, s.dead_stock_days)
    st.markdown("**Inventory policy and risk thresholds**")
    c = st.columns(4)
    sl = c[0].slider("Service level", 0.80, 0.99, float(s.service_level), 0.01)
    rv = c[1].number_input("Review period (days)", 1, 60, s.review_period_days)
    hr = c[2].number_input("High stockout if position < × lead-time demand", 0.5, 2.0, float(s.stockout_high_ratio), 0.1)
    tc = c[3].number_input("Excess-stock coverage target (days)", 7, 365, s.target_coverage_days)
    c = st.columns(3)
    om = c[0].number_input("Overstock Medium above (days of cover)", 14, 720, s.overstock_medium_days)
    oh = c[1].number_input("Overstock High above (days of cover)", 15, 1000, s.overstock_high_days)
    sd = c[2].number_input("Inventory snapshot stale after (days)", 1, 120, s.stale_days)
    st.markdown("**Display**")
    c = st.columns(2)
    rows = c[0].number_input("Rows shown per table", 20, 5000, s.table_rows, 20)
    show_i = c[1].checkbox("Show prediction intervals on forecast charts", s.show_intervals)
    saved = st.form_submit_button("Save settings", type="primary")
if saved:
    if oh <= om:
        st.error("The High overstock threshold must be greater than the Medium threshold.")
    else:
        st.session_state["settings"] = Settings(horizon, int(folds), int(minh), int(maxs), s.season_length, lvl, zero, zoo, int(dead), sl, int(rv), hr, int(om), int(oh), int(tc), int(sd), int(rows), show_i)
        st.success("Settings saved. Dependent results recompute automatically; if processing options changed, re-run Validate & process.")
        st.rerun()
if st.button("Reset to documented defaults"):
    st.session_state["settings"] = Settings()
    st.rerun()
st.caption("Settings live in this browser session only. Uploaded data is held in session memory and is discarded when the session ends; nothing is written to disk or sent to any external service.")
