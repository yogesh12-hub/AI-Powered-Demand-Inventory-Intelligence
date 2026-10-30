import streamlit as st

from src.analytics.data_quality import build_alerts
from src.visualization import dashboard_components as ui

ws, settings = ui.get_ws(), ui.get_settings()
ui.setup_page("Alerts", "Issues detected in your data, forecasts and inventory", filters=False)
if not ws.raw:
    ui.welcome()
    st.stop()
if not ws.processed:
    ui.notice("No alerts yet", "Alerts are generated from validated data. Open Data Quality and process your datasets first.")
    st.stop()
ui.missing_banner(ws)
if ws.forecast is not None and "inventory" in ws.clean:
    ws.inventory_view(settings)
a = build_alerts(ws, settings)
if a.empty:
    st.success("No alerts: validation, cross-checks and (where run) forecast and inventory analysis found nothing to flag.")
    st.stop()
if ws.forecast is None:
    st.caption("Forecast- and inventory-based alerts appear after the forecast has been run.")
c = st.columns(3)
sev = c[0].multiselect("Severity", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
typ = c[1].multiselect("Type", sorted(a["type"].unique()), default=sorted(a["type"].unique()))
q = c[2].text_input("SKU contains")
t = a[a.severity.isin(sev) & a.type.isin(typ)]
if q:
    t = t[t.sku_id.str.contains(q, case=False, na=False)]
ui.kpi_row([("High", str((a.severity == "High").sum()), "", "bad"), ("Medium", str((a.severity == "Medium").sum()), "", "warn"), ("Low", str((a.severity == "Low").sum()), "")])
ui.table(t, "alerts", "alerts.csv")
