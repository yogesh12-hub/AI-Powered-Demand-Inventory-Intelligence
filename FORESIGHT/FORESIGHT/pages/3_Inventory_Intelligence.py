import streamlit as st

from src.inventory import overstock_analysis, stockout_analysis
from src.inventory.risk_scoring import exposure_by_category, scored
from src.reporting.report_generator import filtered
from src.utils.helpers import inr
from src.visualization import charts
from src.visualization import dashboard_components as ui

ws, settings = ui.get_ws(), ui.get_settings()
f = ui.setup_page("Inventory Intelligence", "Stockout and overstock risk from current stock and forecast demand")
ui.require_sales(ws, settings)
ui.require_inventory(ws)
fr = ui.ensure_forecast(ws, settings)
d = filtered(ws, f, settings)
v = d["view"]
if v is None or v.empty:
    ui.notice("No inventory rows match the current filters", "Reset the filters, or check that inventory SKU IDs match the SKU IDs in sales.")
    st.stop()
sc = scored(v)
if sc.empty:
    ui.unscored_notice(v)
    st.stop()
if sc["stale"].any():
    st.warning(f"{int(sc['stale'].sum())} SKU snapshot(s) are older than {settings.stale_days} days relative to the latest sales date; risk may be misstated.")
if len(v) > len(sc):
    st.warning(f"{len(v) - len(sc)} SKU(s) could not be scored (missing on-order, lead time or forecast). See the table at the bottom.")
ui.kpi_row([("Scored SKUs", str(len(sc)), f"{len(v) - len(sc)} not scored"), ("High stockout risk", str(int((sc.stockout_risk == "High").sum())), "position < lead-time demand", "bad"),
            ("High overstock risk", str(int((sc.overstock_risk == "High").sum())), f"> {settings.overstock_high_days} days coverage", "warn"),
            ("Inventory value", inr(sc.inventory_value.sum()) if sc.inventory_value.notna().any() else "no unit_cost", "on-hand at cost"),
            ("Potential lost sales", inr(sc.lost_sales_value.sum()) if sc.lost_sales_value.notna().any() else "—", "during lead time; estimate", "bad"),
            ("Excess inventory value", inr(sc.excess_value.sum()) if sc.excess_value.notna().any() else "no unit_cost", f"above {settings.target_coverage_days}-day target", "warn")])
c1, c2, c3 = st.columns([1, 1, 2], gap="large")
with c1:
    ui.plot(charts.risk_donut(stockout_analysis.summary(sc), "Stockout risk"), "inv_so")
with c2:
    ui.plot(charts.risk_donut(overstock_analysis.summary(sc), "Overstock risk"), "inv_ov")
with c3:
    if sc["category"].notna().any():
        ui.plot(charts.risk_heatmap(sc), "inv_heat")
    else:
        ui.plot(charts.coverage_hist(sc), "inv_cov0")
if sc["category"].notna().any():
    ui.plot(charts.coverage_hist(sc), "inv_cov")
st.subheader("SKU risk table")
a, b = st.columns(2)
risk = a.selectbox("Overall risk", ["All", "High", "Medium", "Low"])
kind = b.selectbox("Risk type", ["All", "Stockout", "Overstock", "Stockout + overstock", "None"])
t = sc if risk == "All" else sc[sc.risk == risk]
t = t if kind == "All" else t[t.risk_type == kind]
cols = ["sku_id", "category", "on_hand", "on_order", "lead_time_days", "lead_time_demand", "coverage_days", "stockout_risk", "overstock_risk", "risk",
        "financial_exposure", "stockout_reason", "overstock_reason"]
ui.table(t.sort_values("financial_exposure", ascending=False)[cols].replace(float("inf"), 9999), "inv_tab", "inventory_risk.csv")
st.caption("coverage_days = (on-hand + on-order) ÷ forecast daily demand; 9999 means no forecast demand. Exposure = lost-sales + excess value (estimates).")
if sc["category"].notna().any():
    st.subheader("Exposure by category")
    ui.table(exposure_by_category(v), "inv_cat", "exposure_by_category.csv")
with st.expander("SKUs not scored"):
    ui.table(v[v.status != "scored"][["sku_id", "status"]], "inv_ns", "not_scored.csv")
