import pandas as pd
import streamlit as st

from src.config import DATASETS
from src.data.column_mapping import mapping_problems
from src.analytics.data_quality import missing_data_report
from src.data.combined_upload import merge_files
from src.reporting.export_csv import to_csv_bytes
from src.utils.helpers import num
from src.visualization import dashboard_components as ui

ROOT = ui.ROOT
ws, settings = ui.get_ws(), ui.get_settings()
ui.setup_page("Data Quality", "One upload for all your data - we check what is there and tell you what is missing", filters=False)
if not ws.raw:
    ui.welcome()
nonce = st.session_state.setdefault("_nonce", {"up": 0})

st.subheader("1. Upload your data")
st.caption("Ek hi upload option: ek file jisme sab data ho (CSV ya Excel). Required columns: **date, sku_id, units_sold** (sales). "
           "Inventory risk ke liye **on_hand_units, on_order_units, lead_time_days** bhi chahiye. Jo data nahi hoga uska analysis band rahega, baaki chalega.")
files = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"], accept_multiple_files=True, key=f"up_{nonce['up']}")
sig = tuple(sorted(f.file_id for f in files)) if files else ()
if sig != st.session_state.get("_files_sig", ()):
    st.session_state["_files_sig"] = sig
    with st.spinner("Reading and splitting your file(s)…"):
        parts, notes = merge_files([(f.name, f.getvalue()) for f in files]) if files else ({}, [])
        for ds in list(ws.loaded):
            ws.remove_dataset(ds)
        for ds, (df, src, size) in parts.items():
            ws.set_dataset(ds, df, src, size, f"{sig}:{ds}")
        st.session_state["_combined_notes"] = notes
        st.session_state.pop("pdf_bytes", None)
        if parts:
            ws.process(settings)
    st.rerun()
if ws.raw and st.button("Clear all uploaded data"):
    nonce["up"] += 1
    st.rerun()
for n in st.session_state.get("_combined_notes", []):
    st.markdown(f"- {n}")

if ws.raw:
    st.subheader("What we found")
    grid = st.columns(2, gap="large") + st.columns(2, gap="large")
    for col, (name, spec) in zip(grid, DATASETS.items()):
        with col:
            if name in ws.meta:
                m = ws.meta[name]
                body = f"<p>From <b>{m['filename']}</b> · {m['rows']:,} rows · {len(m['columns'])} columns</p>"
                dot = "#34d27b"
            else:
                body = f"<p>Not found. Needed: <b>{', '.join(spec['required'])}</b></p>"
                dot = "#ff5d5d"
            st.markdown(f"<div class='fs-card'><h4><span class='fs-dot' style='background:{dot}'></span>{spec['label']}</h4>{body}</div>", unsafe_allow_html=True)

if ws.processed:
    st.subheader("Missing data check")
    rep_df = missing_data_report(ws)
    if (rep_df.status == "OK").all():
        st.success("Everything needed is present.")
    else:
        ui.missing_banner(ws)
    st.dataframe(rep_df, hide_index=True, width="stretch")

with st.expander("Template and field definitions"):
    tpl = ROOT / "templates" / "foresight_combined_template.csv"
    st.download_button("Download combined header-only template (all columns in one file)", tpl.read_bytes(), file_name=tpl.name, mime="text/csv", key="tpl_all")
    st.caption("Put sales rows and inventory rows in the same table (leave unrelated cells blank), or use an Excel workbook with sheets named sales, sku_master, calendar, inventory.")
    st.dataframe(pd.DataFrame([(DATASETS[d]["label"].split(" (")[0], k, "required" if k in DATASETS[d]["required"] else "optional", ty, desc)
                               for d in DATASETS for k, (ty, desc) in DATASETS[d]["fields"].items()],
                              columns=["dataset", "field", "need", "type", "description"]), hide_index=True, width="stretch")

if ws.raw:
    st.subheader("2. Preview and map columns")
    for name in ws.loaded:
        spec = DATASETS[name]
        with st.expander(f"{spec['label']} — preview and mapping", expanded=False):
            st.dataframe(ws.raw[name].head(15), hide_index=True, width="stretch")
            src_opts = ["(not mapped)"] + list(ws.raw[name].columns)
            new = {}
            mc = st.columns(3)
            for i, tgt in enumerate(spec["required"] + spec["optional"]):
                cur = ws.mapping[name].get(tgt)
                with mc[i % 3]:
                    pick = st.selectbox(f"{tgt}{' *' if tgt in spec['required'] else ''}", src_opts, index=src_opts.index(cur) if cur in src_opts else 0, key=f"map_{name}_{tgt}_{ws.meta[name]['file_id']}")
                new[tgt] = None if pick == "(not mapped)" else pick
            df_first = st.checkbox("Dates are day-first (e.g. 31/12/2024)", value=ws.dayfirst.get(name, False), key=f"df_{name}_{ws.meta[name]['file_id']}") if spec["dates"] else False
            ws.set_mapping(name, new, df_first)
            for p in mapping_problems(ws.mapping[name], name):
                st.error(p)

    st.subheader("3. Validate and process")
    if st.button("Validate & process", type="primary"):
        with st.spinner("Validating and cleaning…"):
            ws.process(settings)
        st.session_state.pop("pdf_bytes", None)
        st.rerun()
    if ws.stale_reason(settings):
        st.warning(ws.stale_reason(settings))
    if not ws.processed:
        st.info("Uploaded files have not been validated yet. Click **Validate & process**; nothing downstream runs on unvalidated data.")

if ws.processed:
    st.subheader("4. Results")
    for name, rep in ws.reports.items():
        tag = "✅ ready" if rep.ok else "❌ needs fixes"
        with st.expander(f"{DATASETS[name]['label']} — {tag} · {len(rep.errors)} error(s), {len(rep.warnings)} warning(s)", expanded=not rep.ok):
            if not rep.ok:
                st.error("Fix the errors below, then process again. This dataset is excluded from analysis until valid.")
            st.dataframe(rep.to_frame(), hide_index=True, width="stretch")
            if rep.missing is not None:
                st.markdown("**Missing values by column**")
                st.dataframe(rep.missing, hide_index=True, width="stretch")
            if name in ws.logs:
                st.markdown("**Cleaning log** (every transformation applied)")
                st.dataframe(pd.DataFrame(ws.logs[name]), hide_index=True, width="stretch")
    if ws.weekly_log:
        with st.expander("Weekly aggregation log"):
            st.dataframe(pd.DataFrame(ws.weekly_log), hide_index=True, width="stretch")
    if ws.cross:
        st.markdown("**Cross-dataset checks**")
        st.dataframe(pd.DataFrame(ws.cross), hide_index=True, width="stretch")
    st.markdown("**Feature availability**")
    st.dataframe(pd.DataFrame([(k, "Available" if ok else "Unavailable", "" if ok else why) for k, (ok, why) in ws.features(settings).items()],
                              columns=["feature", "status", "what is needed"]), hide_index=True, width="stretch")
    if ws.weekly is not None and len(ws.weekly):
        st.download_button("Download processed weekly dataset", to_csv_bytes(ws.weekly), file_name="processed_weekly_demand.csv", mime="text/csv")
