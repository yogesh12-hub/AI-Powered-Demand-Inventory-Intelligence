"""Streamlit UI helpers shared by every page (no business logic here)."""
from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import APP_NAME, DATASETS, WELCOME, Settings
from src.analytics.data_quality import missing_data_report
from src.data.upload_manager import Workspace
from src.reporting.export_csv import to_csv_bytes
from src.utils.helpers import Filters
from src.visualization.charts import PLOT_CONFIG

ROOT = Path(__file__).resolve().parents[2]


def get_ws() -> Workspace:
    if "ws" not in st.session_state:
        st.session_state["ws"] = Workspace()
    return st.session_state["ws"]


def get_settings() -> Settings:
    if "settings" not in st.session_state:
        st.session_state["settings"] = Settings()
    return st.session_state["settings"]


def inject_css():
    st.markdown(f"<style>{(ROOT / 'assets' / 'styles.css').read_text()}</style>", unsafe_allow_html=True)


def sidebar_status():
    ws, s = get_ws(), get_settings()
    with st.sidebar:
        st.markdown("<div class='fs-tag'>Demand & inventory intelligence</div>", unsafe_allow_html=True)
        rows = []
        for k, d in DATASETS.items():
            if k in ws.raw:
                ok = ws.reports.get(k).ok if k in ws.reports else None
                col = "#22c55e" if ok else "#ef4444" if ok is False else "#f59e0b"
                txt = "ready" if ok else "needs fixes" if ok is False else "uploaded"
            else:
                col, txt = "#4b5878", "not uploaded"
            rows.append(f"<div><span class='fs-dot' style='background:{col}'></span>{d['label'].split(' (')[0]}: <b>{txt}</b></div>")
        proc = "processed" if ws.processed else ("waiting" if ws.raw else "no data")
        fr = ws.forecast
        model = ("evaluated: " + str(fr.model)) if fr is not None and fr.evaluated else ("run, not validated" if fr is not None else "not run")
        st.markdown(f"<div class='fs-status'><span class='h'>Active datasets</span>{''.join(rows)}</div>"
                    f"<div class='fs-status'><span class='h'>Status</span>Processing: <b>{proc}</b><br>Forecast model: <b>{html.escape(str(model))}</b></div>", unsafe_allow_html=True)


# ---- layout ------------------------------------------------------------------------------------
def header(title: str, subtitle: str = ""):
    ws = get_ws()
    chip = (f"{len(ws.loaded)}/4 datasets · " + ("processed" if ws.processed else "not processed")) if ws.raw else "no dataset"
    st.markdown(f"<div class='fs-title'>{title}</div><div class='fs-sub'>{subtitle}<span class='fs-chip'>{chip}</span></div>",
                unsafe_allow_html=True)


def empty_state(title: str, message: str, action: str = ""):
    st.markdown(f"<div class='fs-empty'><h3>{title}</h3><p>{message}</p><p>{action}</p></div>", unsafe_allow_html=True)


def welcome():
    empty_state(WELCOME, "No dataset uploaded yet. Nothing is shown until you provide your own data — FORESIGHT never loads sample or synthetic records.",
                "Open <b>Data Quality</b> in the sidebar to upload sales, product, calendar and inventory CSVs.")


def _kpi_html(label: str, value: str, sub: str = "", tone: str = "") -> str:
    v = html.escape(str(value)).replace("\n", "<br>")
    long = " long" if len(str(value)) > 14 else ""
    return (f"<div class='fs-kpi {tone}'><div><div class='l'>{html.escape(str(label))}</div><div class='v{long}'>{v}</div></div>"
            f"<div class='s'>{html.escape(str(sub))}</div></div>")


def kpi(label: str, value: str, sub: str = "", tone: str = ""):
    st.markdown(f"<div class='fs-kpis'>{_kpi_html(label, value, sub, tone)}</div>", unsafe_allow_html=True)


def kpi_row(items: list[tuple]):
    """Responsive CSS grid (wraps by available width; stacks on mobile)."""
    st.markdown(f"<div class='fs-kpis n{len(items)}'>" + "".join(_kpi_html(*it) for it in items) + "</div>", unsafe_allow_html=True)


def notice(title: str, message: str):
    """Compact message for missing data (preferred over a large empty state on analysis pages)."""
    st.markdown(f"<div class='fs-notice'><b>{title}</b><p>{message}</p></div>", unsafe_allow_html=True)


def insight_cards(insights: list[dict]):
    cards = "".join(f"<div class='fs-card'><h4>{html.escape(i['title'])}</h4><p>{html.escape(i['evidence'])}</p>"
                    f"<span class='lbl'>Business implication</span><p>{html.escape(i['relevance'])}</p></div>" for i in insights)
    st.markdown(f"<div class='fs-insights'>{cards}</div>", unsafe_allow_html=True)


def recommendation_cards(rows):
    for _, r in rows.iterrows():
        st.markdown(f"<div class='fs-card'><h4>{html.escape(str(r.sku_id))} <span class='fs-pill {r.stockout_risk}'>Stockout: {r.stockout_risk}</span></h4>"
                    f"<p>{html.escape(str(r.recommendation))}</p></div>", unsafe_allow_html=True)


def plot(fig, key: str):
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG, key=key)


def table(df: pd.DataFrame | None, key: str, filename: str, rows: int | None = None, height: int | None = None):
    if df is None or len(df) == 0:
        st.caption("No rows to show for the current selection.")
        return
    n = rows or get_settings().table_rows
    show = df.head(n).copy()
    for c in show.columns:
        if pd.api.types.is_float_dtype(show[c]):
            show[c] = show[c].round(2)
    kw = {"height": height} if height else {}
    st.dataframe(show, hide_index=True, width="stretch", **kw)
    left = f" (showing first {n:,} of {len(df):,})" if len(df) > n else ""
    st.download_button(f"Download CSV{left}", to_csv_bytes(df), file_name=filename, mime="text/csv", key=f"dl_{key}")


# ---- gates -------------------------------------------------------------------------------------
def missing_banner(ws: Workspace):
    """Compact notice listing missing/partial datasets and what they limit. Analysis continues on the data that exists."""
    if not ws.raw:
        return
    rep = missing_data_report(ws)
    bad = rep[rep.status.isin(["Missing", "Partial", "Needs fixes"])]
    if bad.empty:
        return
    items = "".join(f"<li><b>{r.dataset}</b> ({r.status.lower()}): needs <b>{html.escape(r.missing)}</b> &rarr; {html.escape(r.limits)}</li>" for r in bad.itertuples())
    notice("Some data is missing - analysis runs on what you uploaded",
           f"<ul style='margin:6px 0 0 18px'>{items}</ul>Upload a file that contains these columns on the <b>Data Quality</b> page to unlock the rest.")


def unscored_notice(view):
    """Explain why no SKU could be scored (instead of crashing)."""
    why = view[view["status"] != "scored"]["status"].str.replace("Not scored: ", "", regex=False).value_counts().head(4) if view is not None and len(view) else []
    lines = "".join(f"<li>{html.escape(str(k))} - {int(v)} SKU(s)</li>" for k, v in (why.items() if len(why) else []))
    notice("Inventory risk cannot be scored yet", f"No inventory SKU could be scored.<ul style='margin:6px 0 0 18px'>{lines}</ul>"
           "Make sure the inventory data has <b>sku_id, on_hand_units, on_order_units, lead_time_days</b> and that its SKU IDs match the SKU IDs in the sales data.")


def require_sales(ws: Workspace, settings: Settings):
    if not ws.raw:
        welcome()
        st.stop()
    if not ws.processed:
        notice("Your data is not processed yet", "Open <b>Data Quality</b> and click <b>Validate &amp; process</b>.")
        st.stop()
    if ws.sales is None:
        notice("Sales data is required for this page", "The uploaded file(s) do not contain a valid sales table (needs <b>date, sku_id, units_sold</b>).")
        missing_banner(ws)
        st.stop()
    r = ws.stale_reason(settings)
    if r:
        st.warning(r)


def require_inventory(ws: Workspace):
    if "inventory" not in ws.clean:
        notice("Inventory analysis requires valid inventory data",
               "Upload inventory snapshots with <b>date, sku_id, on_hand_units, on_order_units, lead_time_days</b> on the Data Quality page, then process them. "
               "Nothing is estimated without real stock levels.")
        st.stop()
    if not {"on_order_units", "lead_time_days"} <= set(ws.clean["inventory"].columns):
        notice("Inventory file is missing required fields",
               "Risk scoring needs <b>on_order_units</b> and <b>lead_time_days</b>; they are never assumed. Map them on the Data Quality page or upload a file that includes them.")
        st.stop()


def ensure_forecast(ws: Workspace, settings: Settings):
    """Return the current ForecastResult, or prompt to run it (then stop)."""
    key = f"{ws.version}|{settings.forecast_key()}"
    if ws.forecast is None or ws.forecast_key != key:
        notice("Forecasts have not been run for this data", "Forecasting backtests several models with rolling-origin validation, so it runs on demand.")
        if st.button("Run forecast", type="primary", key=f"runfc_{st.session_state.get('_page', '')}"):
            with st.spinner("Backtesting models and generating forecasts…"):
                ws.run_forecast(settings)
            st.rerun()
        st.stop()
    return ws.forecast


# ---- filters -----------------------------------------------------------------------------------
def _reset_filters():
    for k in ("f_date", "f_cat", "f_sub", "f_sku"):
        st.session_state.pop(k, None)
    st.session_state["_flt"] = {}


def render_filters(ws: Workspace) -> Filters:
    s = ws.sales
    store = st.session_state.setdefault("_flt", {})
    if store.get("_v") != ws.version:  # dataset changed: forget old selections
        store.clear()
        store["_v"] = ws.version
        for k in ("f_date", "f_cat", "f_sub", "f_sku"):
            st.session_state.pop(k, None)
    dmin, dmax = s["date"].min().date(), s["date"].max().date()
    cats = sorted(s["category"].dropna().unique()) if "category" in s.columns else []
    subs = sorted(s["subcategory"].dropna().unique()) if "subcategory" in s.columns else []
    skus = sorted(s["sku_id"].unique())
    for k, default in (("f_date", (dmin, dmax)), ("f_cat", []), ("f_sub", []), ("f_sku", [])):
        if k not in st.session_state:
            st.session_state[k] = store.get(k, default)
    c = st.columns([2.2, 1.5, 1.5, 2, 0.8])
    with c[0]:
        st.date_input("Date range", min_value=dmin, max_value=dmax, key="f_date")
    with c[1]:
        st.multiselect("Category", cats, key="f_cat", disabled=not cats, placeholder="All" if cats else "No category data")
    with c[2]:
        st.multiselect("Subcategory", subs, key="f_sub", disabled=not subs, placeholder="All" if subs else "No subcategory data")
    with c[3]:
        st.multiselect("SKU", skus, key="f_sku", placeholder="All SKUs")
    with c[4]:
        st.write("")
        st.button("Reset", on_click=_reset_filters, key=f"reset_{id(st.session_state)}_{st.session_state.get('_page', '')}", width="stretch")
    for k in ("f_date", "f_cat", "f_sub", "f_sku"):
        store[k] = st.session_state[k]
    d = st.session_state["f_date"]
    d0 = pd.Timestamp(d[0]) if len(d) else pd.Timestamp(dmin)
    d1 = pd.Timestamp(d[1]) if len(d) > 1 else pd.Timestamp(dmax)
    return Filters(d0, d1, list(st.session_state["f_cat"]), list(st.session_state["f_sub"]), list(st.session_state["f_sku"]))


def setup_page(title: str, subtitle: str = "", filters: bool = True) -> Filters | None:
    st.session_state["_page"] = title
    header(title, subtitle)
    ws, settings = get_ws(), get_settings()
    if not filters:
        return None
    if ws.sales is None:
        return None
    return render_filters(ws)
