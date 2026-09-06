import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime
from ask_your_data import render_ask_your_data_tab

st.set_page_config(page_title="RetailPulse", layout="wide")
st.title("📊 RetailPulse — Customer Analytics & Demand Forecasting")

st.markdown("""
This dashboard is generated from the same cleaned data and models built in
the companion Colab notebook. Re-run the notebook's export cell to refresh
`rfm.csv` / `forecast.csv` before restarting this app.
""")


@st.cache_data
def load():
    rfm = pd.read_csv("rfm.csv")
    forecast = pd.read_csv("forecast.csv", parse_dates=["ds"])
    return rfm, forecast


try:
    rfm, forecast = load()
except FileNotFoundError:
    st.warning("Run the notebook's export cell first to create rfm.csv and forecast.csv.")
    st.stop()


# ------------------------------------------------------------------
# Shared helpers for the Export Reports tab
# ------------------------------------------------------------------
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def build_pdf_summary(rfm: pd.DataFrame, forecast: pd.DataFrame) -> bytes:
    """Builds a short, print-ready PDF summary of the headline numbers and
    the segment/forecast tables -- no external image dependency needed,
    so this works on Streamlit Community Cloud without extra system
    packages (fpdf2 is pure Python)."""
    from fpdf import FPDF

    total_recommended = forecast["yhat"].tail(30).sum()
    segment_summary = rfm.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean().round(1)
    segment_counts = rfm["Segment"].value_counts()

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(91, 63, 217)  # accent purple
    pdf.cell(0, 12, "RetailPulse - Summary Report", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Key Metrics", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"Customers segmented: {len(rfm):,}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Segments identified: {rfm['Segment'].nunique()}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Recommended 30-day stock value: Rs {total_recommended:,.0f}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Customer Segments", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 10)
    col_w = [45, 30, 30, 35, 30]
    headers = ["Segment", "Count", "Recency", "Frequency", "Monetary"]
    for w, h in zip(col_w, headers):
        pdf.cell(w, 8, h, border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 10)
    for seg in segment_summary.index:
        row = segment_summary.loc[seg]
        vals = [str(seg), str(int(segment_counts.get(seg, 0))),
                f"{row['Recency']:.1f}", f"{row['Frequency']:.1f}", f"{row['Monetary']:.0f}"]
        for w, v in zip(col_w, vals):
            pdf.cell(w, 8, v, border=1)
        pdf.ln()
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "30-Day Forecast (tail)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 10)
    for w, h in zip([50, 60, 60], ["Date", "Forecast (yhat)", "Range"]):
        pdf.cell(w, 8, h, border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 10)
    for _, r in forecast.tail(10).iterrows():
        pdf.cell(50, 8, str(r["ds"].date()), border=1)
        pdf.cell(60, 8, f"{r['yhat']:,.0f}", border=1)
        pdf.cell(60, 8, f"{r['yhat_lower']:,.0f} - {r['yhat_upper']:,.0f}", border=1)
        pdf.ln()

    return bytes(pdf.output())


# ------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Customer Segments", "Demand Forecast", "Reorder Guidance", "📤 Export Reports", "💬 Ask Your Data"]
)

with tab1:
    st.subheader("Customer Segments (RFM + KMeans)")
    st.dataframe(rfm.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean().round(1))
    st.bar_chart(rfm["Segment"].value_counts())

with tab2:
    st.subheader("30-Day Revenue Forecast")
    st.line_chart(forecast.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]].tail(60))

with tab3:
    st.subheader("Recommended Stock Value (next 30 days)")
    total = forecast["yhat"].tail(30).sum()
    st.metric("Total recommended stock value", f"Rs {total:,.0f}")

with tab4:
    st.subheader("Export Reports")
    st.caption("Download the underlying data or a print-ready PDF summary for stakeholders.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "⬇️ Customer Segments (CSV)",
            data=df_to_csv_bytes(rfm.reset_index() if rfm.index.name else rfm),
            file_name="retailpulse_customer_segments.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "⬇️ Demand Forecast (CSV)",
            data=df_to_csv_bytes(forecast),
            file_name="retailpulse_demand_forecast.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col3:
        top_products_csv = forecast.tail(30)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        st.download_button(
            "⬇️ Reorder Guidance (CSV)",
            data=df_to_csv_bytes(top_products_csv),
            file_name="retailpulse_reorder_guidance.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.divider()
    st.markdown("**PDF Summary Report** — key metrics, segment table, and forecast tail in one page.")
    try:
        pdf_bytes = build_pdf_summary(rfm, forecast)
        st.download_button(
            "📄 Download PDF Summary",
            data=pdf_bytes,
            file_name=f"retailpulse_summary_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
        )
    except ImportError:
        st.error("PDF export requires `fpdf2` — add it to requirements.txt (pip install fpdf2).")

with tab5:
    render_ask_your_data_tab()