import streamlit as st

from src.analytics import eda, sales_analysis as sa
from src.inventory.risk_scoring import scored
from src.reporting.report_generator import filtered
from src.utils.helpers import inr, num, pct
from src.visualization import charts
from src.visualization import dashboard_components as ui

ws, settings = ui.get_ws(), ui.get_settings()
f = ui.setup_page("Overview", "Calculated KPIs from your uploaded data")
ui.require_sales(ws, settings)
d = filtered(ws, f, settings)
s = d["sales"]
if s is None or s.empty:
    ui.empty_state("No sales match the current filters", "Widen the date range or reset the filters.")
    st.stop()
k = sa.kpis(s)
fr, v = ws.forecast, d["view"]
sc = scored(v) if v is not None else None
has_sc = sc is not None and len(sc) > 0
ui.missing_banner(ws)
if v is not None and not has_sc:
    ui.unscored_notice(v)
wape = d["metrics"].set_index("model") if d["metrics"] is not None and len(d["metrics"]) else None
model_w = wape.loc[fr.model, "wape"] if wape is not None and fr.model in wape.index else None
items = [("Revenue", inr(k["revenue"]) if k["revenue"] is not None else "n/a", "no revenue/price columns" if k["revenue"] is None else f"{k['start'].date()} → {k['end'].date()}", "brand"),
         ("Units sold", num(k["units"]), f"{k['skus']} SKUs"),
         ("Forecast WAPE", pct(model_w) if model_w is not None else "—", f"{fr.model}, backtested" if model_w is not None else "run the forecast", "good" if model_w is not None else ""),
         ("High stockout-risk SKUs", str(int((sc.stockout_risk == 'High').sum())) if has_sc else "—", "" if has_sc else "needs inventory + forecast", "bad" if has_sc and (sc.stockout_risk == 'High').any() else ""),
         ("Inventory value (at cost)", inr(sc.inventory_value.sum()) if has_sc and sc.inventory_value.notna().any() else "—", "on-hand × unit cost"),
         ("Estimated exposure", inr(sc.financial_exposure.sum()) if has_sc and sc.financial_exposure.notna().any() else "—", "lost sales + excess; an estimate", "warn")]
ui.kpi_row(items)
if fr is None and "inventory" in ws.clean:
    if st.button("Run forecast & inventory analysis", type="primary"):
        with st.spinner("Running backtest and scoring inventory…"):
            ws.run_forecast(settings)
        st.rerun()
elif fr is None:
    st.info("Forecasting has not been run yet. Open Demand Forecasting to run it.")
c1, c2 = st.columns([3, 2], gap="large")
with c1:
    ui.plot(charts.sales_trend(sa.trend(s, "Weekly"), "Weekly units and revenue"), "ov_trend")
with c2:
    if has_sc:
        ui.plot(charts.risk_donut(sc["stockout_risk"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0), "Stockout risk (SKUs)"), "ov_donut")
    else:
        ui.notice("Inventory risk summary", "Appears once inventory is uploaded and the forecast has run.")
st.subheader("Business insights")
ins, notes = eda.generate_insights(s, d["weekly"], ws.clean.get("calendar"), ws.inventory_latest(), settings.dead_stock_days)
if ins:
    ui.insight_cards(ins)
if len(ins) < 3:
    st.caption("Fewer than three insights are supported by this data. " + " ".join(notes))
elif notes:
    st.caption("Not shown (data does not support them): " + " ".join(notes))
st.subheader("Data sources")
rows = [{"dataset": n, "file": m["filename"], "rows uploaded": m["rows"], "rows after cleaning": len(ws.clean[n]) if n in ws.clean else None,
         "status": "ready" if n in ws.reports and ws.reports[n].ok else "needs fixes"} for n, m in ws.meta.items()]
st.dataframe(rows, hide_index=True, width="stretch")
