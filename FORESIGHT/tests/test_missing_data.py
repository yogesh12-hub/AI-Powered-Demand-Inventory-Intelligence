"""Regression: missing / mismatched data must produce notices and partial analysis - never a crash."""
import io

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.analytics.data_quality import build_alerts, missing_data_report
from src.config import Settings
from src.data.combined_upload import merge_files
from src.data.upload_manager import Workspace
from src.inventory.risk_scoring import VIEW_COLUMNS, scored

APP = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "app.py")
PAGES = ["1_Overview", "2_Demand_Forecasting", "3_Inventory_Intelligence", "4_Replenishment", "5_Sales_Analytics",
         "6_Reports_Exports", "7_Data_Quality", "8_Alerts", "9_Settings"]


def make_ws(raw, st, drop=(), mismatch=False):
    w = Workspace()
    for k, df in raw.items():
        if k in drop:
            continue
        df = df.copy()
        if mismatch and k in ("inventory", "sku_master"):
            df["sku_id"] = ["ZZ%02d" % i for i in range(len(df))]
        w.set_dataset(k, df, f"{k}.csv", 1)
    w.process(st)
    return w


def test_unmatched_inventory_skus_keep_full_schema(raw, settings):
    w = make_ws(raw, settings, mismatch=True)
    w.run_forecast(settings)
    v = w.inventory_view(settings)
    assert set(VIEW_COLUMNS) <= set(v.columns) and scored(v).empty
    assert v.status.str.startswith("Not scored").all()
    assert not build_alerts(w, settings).empty                      # alerts still build (explain, not crash)
    assert any("None of the inventory SKU" in c["message"] for c in w.cross)


def test_missing_data_report_lists_what_is_missing(raw, settings):
    w = make_ws(raw, settings, drop=("inventory", "calendar"))
    r = missing_data_report(w).set_index("dataset")
    assert r.loc["Inventory", "status"] == "Missing" and "on_order_units" in r.loc["Inventory", "missing"]
    assert "reorder" in r.loc["Inventory", "limits"].lower() or "risk" in r.loc["Inventory", "limits"].lower()
    assert r.loc["Calendar", "status"] == "Missing" and r.loc["Sales", "status"] == "OK"


def test_partial_columns_reported(raw, settings):
    inv = raw["inventory"].drop(columns=["lead_time_days"])
    w = Workspace()
    w.set_dataset("sales", raw["sales"], "s", 1)
    w.set_dataset("inventory", inv, "i", 1)
    w.process(settings)
    r = missing_data_report(w).set_index("dataset")
    assert r.loc["Inventory", "status"] == "Partial" and "lead_time_days" in r.loc["Inventory", "missing"]


def test_merge_files_multiple_sources(raw):
    csv = lambda df: df.to_csv(index=False).encode()
    parts, notes = merge_files([("sales_daily.csv", csv(raw["sales"])), ("stock.csv", csv(raw["inventory"])), ("broken.xlsx", b"not an excel file")])
    assert {"sales", "inventory"} <= set(parts) and parts["sales"][1] == "sales_daily.csv"
    assert any("broken.xlsx" in n and "could not be read" in n for n in notes)
    assert not any("sales: NOT created" in n for n in notes)         # created elsewhere, so no noise


def _app(w, st):
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["ws"], at.session_state["settings"] = w, st
    return at.run()


@pytest.mark.parametrize("case", ["mismatch", "sales_only"])
@pytest.mark.parametrize("page", PAGES)
def test_every_page_survives_missing_or_mismatched_data(raw, settings, case, page):
    w = make_ws(raw, settings, drop=("inventory", "sku_master", "calendar") if case == "sales_only" else (), mismatch=(case == "mismatch"))
    w.run_forecast(settings)
    at = _app(w, settings)
    at.switch_page(f"pages/{page}.py").run()
    assert not at.exception, [e.value for e in at.exception]
    text = " ".join(m.value for m in at.markdown)
    assert "Traceback" not in text and "AttributeError" not in text


def test_sales_only_still_analyses_and_names_missing_data(raw, settings):
    w = make_ws(raw, settings, drop=("inventory", "sku_master", "calendar"))
    at = _app(w, settings)
    at.switch_page("pages/1_Overview.py").run()
    text = " ".join(m.value for m in at.markdown)
    assert "Units sold" in text and "Some data is missing" in text and "Inventory" in text
    at.switch_page("pages/3_Inventory_Intelligence.py").run()
    text = " ".join(m.value for m in at.markdown)
    assert "Inventory analysis requires valid inventory data" in text and not at.exception


def test_no_sales_dataset_gives_notice_not_error(raw, settings):
    w = make_ws(raw, settings, drop=("sales",))
    at = _app(w, settings)
    for p in ("1_Overview", "2_Demand_Forecasting", "5_Sales_Analytics"):
        at.switch_page(f"pages/{p}.py").run()
        assert not at.exception
        assert "Sales data is required" in " ".join(m.value for m in at.markdown)


def test_unfiltered_view_keeps_unmatched_inventory_rows_so_reason_is_shown(raw, settings):
    from src.reporting.report_generator import filtered
    from src.utils.helpers import Filters
    w = make_ws(raw, settings, mismatch=True)
    w.run_forecast(settings)
    d = filtered(w, Filters(), settings)
    assert len(d["view"]) == 10 and d["view"].status.str.startswith("Not scored").all()


def test_filters_still_restrict_when_selected(ws, settings):
    from src.reporting.report_generator import filtered
    from src.utils.helpers import Filters
    ws.run_forecast(settings)
    d = filtered(ws, Filters(skus=["SKU01"]), settings)
    assert set(d["view"].sku_id) == {"SKU01"} and set(d["forecasts"].sku_id) == {"SKU01"}
