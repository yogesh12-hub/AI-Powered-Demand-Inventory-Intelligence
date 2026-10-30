import streamlit as st

from src.reporting.export_csv import to_csv_bytes, to_zip_bytes
from src.reporting.report_generator import build_tables, filter_describe, make_pdf
from src.visualization import dashboard_components as ui

ws, settings = ui.get_ws(), ui.get_settings()
f = ui.setup_page("Reports & Exports", "Everything below reflects the active datasets and the filters above")
ui.require_sales(ws, settings)
ui.missing_banner(ws)
st.caption(f"Scope: {filter_describe(f)}")
if ws.forecast is None:
    st.info("Forecast and inventory exports appear after you run the forecast (Demand Forecasting page).")
tables, insights, lines = build_tables(ws, settings, f)
LABELS = {"data_quality_report": "Data-quality report (with cleaning log)", "eda_top_skus": "EDA: top SKUs", "eda_low_skus": "EDA: lowest SKUs",
          "eda_category_performance": "EDA: category performance", "business_insights": "Business insights", "forecast_outputs": "Forecast outputs",
          "model_evaluation": "Model evaluation results", "model_evaluation_backtest_rows": "Backtest rows (actual vs predicted)",
          "inventory_risk_report": "Inventory risk report", "replenishment_recommendations": "Replenishment recommendations", "executive_summary": "Executive summary"}
for name, df in tables.items():
    a, b, c = st.columns([4, 1.2, 1.5])
    a.markdown(f"**{LABELS.get(name, name)}**")
    b.caption(f"{len(df):,} rows")
    c.download_button("Download CSV", to_csv_bytes(df), file_name=f"{name}.csv", mime="text/csv", key=f"rep_{name}")
st.divider()
c1, c2 = st.columns(2)
c1.download_button("Download all CSVs (ZIP)", to_zip_bytes(tables), file_name="foresight_exports.zip", mime="application/zip")
with c2:
    if st.button("Build consolidated PDF report", type="primary"):
        with st.spinner("Building PDF…"):
            st.session_state["pdf_bytes"] = (ws.version, make_pdf(ws, settings, f))
    p = st.session_state.get("pdf_bytes")
    if p and p[0] == ws.version:
        st.download_button("Download PDF report", p[1], file_name="foresight_report.pdf", mime="application/pdf")
st.caption("Charts: use the camera icon in any chart's toolbar to download it as a PNG.")
