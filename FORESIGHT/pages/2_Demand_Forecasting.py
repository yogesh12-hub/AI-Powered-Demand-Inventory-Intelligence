import numpy as np
import pandas as pd
import streamlit as st

from src.inventory.risk_scoring import scored
from src.reporting.report_generator import filtered
from src.utils.helpers import inr, num, pct
from src.visualization import charts
from src.visualization import dashboard_components as ui

ws, settings = ui.get_ws(), ui.get_settings()
f = ui.setup_page("Demand Forecasting", "Weekly SKU-level forecasts, seasonal-naive baseline and rolling-origin backtest")
ui.require_sales(ws, settings)
ui.missing_banner(ws)
fr = ui.ensure_forecast(ws, settings)
for w in fr.warnings:
    st.warning(w)
if fr.forecasts.empty:
    ui.empty_state("Forecasting requires sufficient historical data.", f"No SKU has {settings.min_history_weeks}+ complete weeks. Upload a longer history or lower the minimum in Settings.")
    st.stop()
opts = [x for x in (6, 7, 8) if x <= fr.horizon]
h = st.select_slider("Forecast horizon (weeks)", options=opts, value=opts[-1]) if len(opts) > 1 else opts[0]
d = filtered(ws, f, settings, horizon=h)
fc, m = d["forecasts"], d["metrics"]
if fc is None or fc.empty:
    st.info("No forecasted SKU matches the selected category/SKU filters.")
    st.stop()
total_skus = ws.weekly["sku_id"].nunique()
if m is not None and len(m):
    mi = m.set_index("model")
    mw = mi.loc[fr.model, "wape"] if fr.model in mi.index else np.nan
    bw = mi.loc["seasonal_naive", "wape"] if "seasonal_naive" in mi.index else np.nan
    imp = (bw - mw) / bw * 100 if bw and np.isfinite(bw) and bw > 0 else np.nan
    n_obs = int(mi.loc[fr.model, "n_obs"]) if fr.model in mi.index else 0
    ui.kpi_row([("Selected model", fr.model, f"lowest backtest WAPE of {len(mi)} models", "brand"), ("WAPE", pct(mw), f"over {n_obs:,} held-out SKU-weeks"),
                ("Baseline WAPE", pct(bw), fr.baseline_kind), ("Vs baseline", (f"{imp:+.1f}%" if np.isfinite(imp) else "—"), "relative WAPE improvement", "good" if np.isfinite(imp) and imp > 0 else "warn"),
                ("Evaluation period", f"{fr.eval_period[0].date()}\n→ {fr.eval_period[1].date()}" if fr.eval_period else "—", f"{fr.folds} rolling-origin fold(s)"),
                ("Forecast coverage", f"{fc.sku_id.nunique()}/{total_skus}", "SKUs forecast vs in data")])
else:
    ui.kpi_row([("Model", fr.model or "—", "not validated", "warn"), ("WAPE", "—", "no backtest possible"), ("Coverage", f"{fc.sku_id.nunique()}/{total_skus}", "SKUs forecast")])
if not fr.intervals and fr.evaluated is False:
    st.caption("Prediction intervals are shown only when residuals from a backtest support them.")

st.subheader("Historical vs forecast")
choices = ["All filtered SKUs (total)"] + sorted(fc.sku_id.unique())
sel = st.selectbox("SKU", choices)
hist_w = d["weekly"] if d["weekly"] is not None else ws.weekly
if sel == choices[0]:
    hist = hist_w[hist_w.sku_id.isin(fc.sku_id.unique())].groupby("week_start")["units"].sum()
    fsel = fc.groupby("week_start", as_index=False)[["forecast", "baseline"]].sum().assign(lower=np.nan, upper=np.nan)
    note = "Totals do not show intervals (SKU-level intervals cannot be summed)."
else:
    hist = hist_w[hist_w.sku_id == sel].set_index("week_start")["units"]
    fsel, note = fc[fc.sku_id == sel], ""
ui.plot(charts.forecast_chart(hist, fsel, show_intervals=settings.show_intervals), "fc_chart")
st.caption(note or (f"Shaded band: {int(settings.interval_level * 100)}% interval from backtest residuals (SKUs with 8+ residuals)." if fr.intervals else "No prediction interval available for this selection."))

st.subheader("Forecast comparison by SKU")
tab = fc.groupby("sku_id").agg(forecast_total=("forecast", "sum"), baseline_total=("baseline", "sum")).reset_index()
last8 = hist_w.sort_values("week_start").groupby("sku_id").tail(8).groupby("sku_id")["units"].sum().rename("last_8w_actual")
tab = tab.merge(last8.reset_index(), on="sku_id", how="left")
if len(fr.sku_metrics):
    tab = tab.merge(fr.sku_metrics[["sku_id", "wape_model", "wape_baseline"]], on="sku_id", how="left")
tab["vs_last_8w_%"] = (tab["forecast_total"] / tab["last_8w_actual"].replace(0, np.nan) - 1) * 100
ui.table(tab.sort_values("forecast_total", ascending=False), "fc_tab", "forecast_comparison.csv")

if m is not None and len(m):
    st.subheader("Model evaluation")
    ui.table(m, "metrics", "model_evaluation.csv")
    st.caption("All models are scored on identical held-out SKU-weeks. WAPE = Σ|actual − forecast| / Σ|actual|; undefined (blank) when actual demand sums to zero.")
    ui.plot(charts.backtest_chart(d["backtest"], fr.model), "bt_chart")
v = d["view"]
st.subheader("Inventory view")
if v is None:
    st.info("Inventory recommendations appear here once valid inventory snapshots (with on-order and lead time) are uploaded.")
else:
    sc = scored(v)
    ui.kpi_row([("High stockout risk", str(int((sc.stockout_risk == "High").sum())), "SKUs", "bad"), ("High overstock risk", str(int((sc.overstock_risk == "High").sum())), "SKUs", "warn"),
                ("Recommended reorder spend", inr(sc.reorder_cost.sum()), "advisory estimate")])
    top = sc[sc.reorder_qty > 0].sort_values("stockout_score", ascending=False).head(5)
    ui.recommendation_cards(top)
with st.expander(f"SKUs not forecast ({len(fr.excluded)})"):
    ui.table(fr.excluded, "excl", "skus_not_forecast.csv")
