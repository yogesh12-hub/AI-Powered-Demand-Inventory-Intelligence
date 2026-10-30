"""FORESIGHT entry point: page config, global styling, navigation and session state."""
import streamlit as st

st.set_page_config(page_title="FORESIGHT", page_icon="🔭", layout="wide", initial_sidebar_state="expanded")

from src.visualization.dashboard_components import get_settings, get_ws, inject_css, sidebar_status  # noqa: E402

get_ws()
get_settings()
inject_css()
st.logo("assets/logo.svg")

PAGES = [
    st.Page("pages/1_Overview.py", title="Overview", icon=":material/dashboard:", default=True),
    st.Page("pages/2_Demand_Forecasting.py", title="Demand Forecasting", icon=":material/show_chart:"),
    st.Page("pages/3_Inventory_Intelligence.py", title="Inventory Intelligence", icon=":material/inventory_2:"),
    st.Page("pages/4_Replenishment.py", title="Replenishment", icon=":material/local_shipping:"),
    st.Page("pages/5_Sales_Analytics.py", title="Sales Analytics", icon=":material/bar_chart:"),
    st.Page("pages/6_Reports_Exports.py", title="Reports & Exports", icon=":material/download:"),
    st.Page("pages/7_Data_Quality.py", title="Data Quality", icon=":material/fact_check:"),
    st.Page("pages/8_Alerts.py", title="Alerts", icon=":material/notifications:"),
    st.Page("pages/9_Settings.py", title="Settings", icon=":material/settings:"),
]
nav = st.navigation(PAGES)
sidebar_status()
try:
    nav.run()
except Exception:  # noqa: BLE001 - never show a raw traceback to the user
    import traceback

    from src.visualization.dashboard_components import notice

    notice("This page could not be displayed with the current data",
           "Part of your data may be missing or in an unexpected format. Check the <b>Data Quality</b> page to see what is missing, then upload a file that includes it.")
    with st.expander("Technical details (for support)"):
        st.code(traceback.format_exc())
