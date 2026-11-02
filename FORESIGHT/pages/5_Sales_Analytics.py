import streamlit as st

from src.analytics import sales_analysis as sa
from src.analytics import seasonality as se
from src.reporting.report_generator import filtered
from src.utils.helpers import inr, num, pct
from src.visualization import charts
from src.visualization import dashboard_components as ui
from src.utils.helpers import Filters, apply_filters

ws, settings = ui.get_ws(), ui.get_settings()
f = ui.setup_page("Sales Analytics", "Trends, seasonality, top and low movers, promotions and dead stock")
ui.require_sales(ws, settings)
ui.missing_banner(ws)
d = filtered(ws, f, settings)
s = d["sales"]
if s is None or s.empty:
    ui.empty_state("No sales match the current filters", "Widen the date range or reset the filters.")
    st.stop()
k = sa.kpis(s)
ui.kpi_row([("Units sold", num(k["units"]), f"{k['days']} days"), ("Revenue", inr(k["revenue"]) if k["revenue"] is not None else "n/a", "from revenue or price × units"), ("Active SKUs", str(k["skus"]), "")])
freq = st.radio("Granularity", ["Daily", "Weekly", "Monthly"], index=1, horizontal=True)
ui.plot(charts.sales_trend(sa.trend(s, freq), f"{freq} sales"), "sa_trend")

c1, c2 = st.columns(2, gap="large")
by = "revenue" if k["revenue"] is not None else "units"
top, low = sa.top_low(s, 10, by)
with c1:
    ui.plot(charts.bar(top, "sku_id", by, f"Top 10 SKUs by {by}", horizontal=True, height=360), "sa_top")
with c2:
    ui.plot(charts.bar(low, "sku_id", by, f"Lowest 10 SKUs by {by} (among SKUs with sales)", color="#5b9dff", horizontal=True, height=360), "sa_low")
cp = sa.category_performance(s)
if cp is not None:
    st.subheader("Category performance")
    ui.plot(charts.bar(cp, "category", "revenue" if cp.revenue.notna().any() else "units", "Category performance"), "sa_cat")
    ui.table(cp, "sa_cat_t", "category_performance.csv")
else:
    st.caption("Category performance needs a product dataset with a category column.")

st.subheader("Seasonality")
mi = se.monthly_index(s)
if mi is not None:
    ui.plot(charts.seasonality_bar(mi), "sa_season")
else:
    st.info("Seasonality needs at least 12 full months of sales covering every calendar month in the selected range.")
ss = se.seasonal_strength(d["weekly"]) if d["weekly"] is not None else None
if ss is not None:
    st.caption(f"Year-over-year correlation of weekly total units: {ss:.2f} (closer to 1 = stronger repeating pattern).")

st.subheader("Promotions and calendar events")
pe = sa.promo_effect(s)
cal = ws.clean.get("calendar")
shown = False
if pe:
    shown = True
    st.markdown(f"**Promo flag:** {pe['promo_avg']:.2f} vs {pe['base_avg']:.2f} units per SKU-day ({pe['uplift_pct']:+.1f}%). Descriptive only; promotions are not randomly assigned.")
for col, label in (("promo_event_flag", "Calendar promo events"), ("holiday_flag", "Holidays")):
    e = sa.event_effect(s, cal, col)
    if e:
        shown = True
        st.markdown(f"**{label}:** {e['event_avg']:,.0f} vs {e['other_avg']:,.0f} total units per day ({e['uplift_pct']:+.1f}%) across {e['event_days']} event days.")
if not shown:
    st.caption("Not enough promotion/holiday data (needs promo_flag or a calendar with promo_event/holiday and enough event days).")

st.subheader("Dead stock")
full = apply_filters(ws.sales, Filters(categories=f.categories, subcategories=f.subcategories, skus=f.skus), None)
ds = sa.dead_stock(full, ws.sales["date"].max(), settings.dead_stock_days, ws.inventory_latest())
st.caption(f"SKUs with no sales in the last {settings.dead_stock_days} days (configurable in Settings), relative to the latest sales date.")
ui.table(ds, "sa_dead", "dead_stock.csv")
