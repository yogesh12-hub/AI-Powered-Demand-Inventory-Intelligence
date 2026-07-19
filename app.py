import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="RetailPulse", layout="wide")
st.title("📊 RetailPulse — Customer Analytics & Demand Forecasting")

st.markdown("""
This dashboard is generated from the same cleaned data and models built in
the companion Colab notebook. Re-run the notebook's export cell to refresh
`rfm.csv` / `forecast.csv` / `churn_scores.csv` before restarting this app.
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

tab1, tab2, tab3 = st.tabs(["Customer Segments", "Demand Forecast", "Reorder Guidance"])

with tab1:
    st.subheader("Customer Segments (RFM + KMeans)")
    st.dataframe(rfm.groupby("Segment")[["Recency","Frequency","Monetary"]].mean().round(1))
    st.bar_chart(rfm["Segment"].value_counts())

with tab2:
    st.subheader("30-Day Revenue Forecast")
    st.line_chart(forecast.set_index("ds")[["yhat","yhat_lower","yhat_upper"]].tail(60))

with tab3:
    st.subheader("Recommended Stock Value (next 30 days)")
    total = forecast["yhat"].tail(30).sum()
    st.metric("Total recommended stock value", f"Rs {total:,.0f}")
