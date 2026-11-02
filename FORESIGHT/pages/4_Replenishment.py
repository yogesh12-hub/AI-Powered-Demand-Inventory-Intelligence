import streamlit as st

from src.inventory.risk_scoring import scored
from src.reporting.report_generator import filtered
from src.utils.helpers import inr, num
from src.visualization import dashboard_components as ui

ws, settings = ui.get_ws(), ui.get_settings()
f = ui.setup_page("Replenishment", "Explainable, advisory reorder quantities. No purchase orders are placed.")
ui.require_sales(ws, settings)
ui.require_inventory(ws)
ui.ensure_forecast(ws, settings)

with st.expander("Safety-stock policy (applies app-wide)", expanded=True):
    c = st.columns(3)
    settings.service_level = c[0].slider("Service level", 0.80, 0.99, float(settings.service_level), 0.01, format="%.2f")
    settings.review_period_days = c[1].number_input("Review period (days)", 1, 60, int(settings.review_period_days))
    settings.target_coverage_days = c[2].number_input("Excess-stock coverage target (days)", 7, 365, int(settings.target_coverage_days))
    st.caption("target stock = demand over (lead time + review period) + safety stock;  safety stock = z(service level) × weekly forecast-error σ × √(lead time / 7);  "
               "reorder qty = max(0, target − (on-hand + on-order)), rounded up and raised to MOQ if provided.")
st.subheader("What-if analysis")
w = st.columns(2)
dm = w[0].slider("Demand change vs forecast (%)", -50, 100, 0, 5)
lm = w[1].slider("Lead-time change (%)", -50, 100, 0, 5)
wi = {"demand_mult": 1 + dm / 100, "lead_time_mult": 1 + lm / 100} if (dm or lm) else None
base = filtered(ws, f, settings)["view"]
d = filtered(ws, f, settings, what_if=wi)
v = d["view"]
if v is None or v.empty:
    ui.notice("Nothing to recommend", "No inventory SKUs match the filters or none could be scored.")
    st.stop()
sc, sb = scored(v), scored(base)
if sc.empty:
    ui.unscored_notice(v)
    st.stop()
rec = sc[sc.reorder_qty > 0]
delta = lambda a, b: f"{'+' if a - b >= 0 else '−'}{abs(a - b):,.0f} vs base" if wi else "base scenario"
ui.kpi_row([("SKUs to reorder", str(len(rec)), delta(len(rec), (sb.reorder_qty > 0).sum()), "brand"),
            ("Units to order", num(rec.reorder_qty.sum()), delta(rec.reorder_qty.sum(), sb.reorder_qty.sum())),
            ("Estimated reorder cost", inr(rec.reorder_cost.sum()) if rec.reorder_cost.notna().any() else "no unit_cost", delta(rec.reorder_cost.sum(), sb.reorder_cost.sum())),
            ("Urgent", str(int((rec.priority == "Urgent").sum())), "high stockout risk", "bad"),
            ("High stockout risk", str(int((sc.stockout_risk == "High").sum())), delta((sc.stockout_risk == "High").sum(), (sb.stockout_risk == "High").sum()), "bad")])
pr = st.multiselect("Priority", ["Urgent", "Soon", "Routine"], default=["Urgent", "Soon", "Routine"])
plan = rec[rec.priority.isin(pr)].copy()
plan["_o"] = plan.priority.map({"Urgent": 0, "Soon": 1, "Routine": 2})
plan = plan.sort_values(["_o", "financial_exposure"], ascending=[True, False])
cols = ["sku_id", "category", "priority", "stockout_risk", "overstock_risk", "on_hand", "on_order", "position", "forecast_weekly_avg", "coverage_days",
        "lead_time_days", "safety_stock", "target_stock", "reorder_qty", "reorder_cost", "recommendation"]
ui.table(plan[cols].replace(float("inf"), 9999), "plan", "replenishment_action_plan.csv")
if wi:
    st.info(f"Scenario applied: demand {dm:+d}%, lead time {lm:+d}%. Table and downloads reflect this scenario.")
