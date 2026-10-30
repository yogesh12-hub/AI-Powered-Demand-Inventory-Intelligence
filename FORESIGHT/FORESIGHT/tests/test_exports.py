import io
import zipfile

import pandas as pd

from src.data.upload_manager import Workspace, read_csv_bytes
from src.reporting.export_csv import to_csv_bytes, to_zip_bytes
from src.reporting.report_generator import build_tables, filtered, make_pdf
from src.utils.helpers import Filters, apply_filters, inr


def test_inr_formatting_lakh_crore():
    assert inr(1234567) == "₹12,34,567" and inr(999) == "₹999" and inr(None) == "—" and inr(-1500) == "-₹1,500"


def test_filtered_exports_contain_only_selected_records(ws, settings):
    ws.run_forecast(settings)
    f = Filters(skus=["SKU02", "SKU03"])
    t, _, _ = build_tables(ws, settings, f)
    assert set(t["forecast_outputs"].sku_id) == {"SKU02", "SKU03"}
    assert set(t["inventory_risk_report"].sku_id) == {"SKU02", "SKU03"}
    assert set(t["model_evaluation_backtest_rows"].sku_id) <= {"SKU02", "SKU03"}
    cat = Filters(categories=["A"])
    d = filtered(ws, cat, settings)
    assert set(d["sales"].category.dropna()) == {"A"} and set(d["forecasts"].sku_id) == {f"SKU{i:02d}" for i in range(0, 10, 2)}


def test_date_filter_applies_to_sales_exports(ws, settings):
    f = Filters(date_from=pd.Timestamp("2022-06-01"), date_to=pd.Timestamp("2022-06-30"))
    d = filtered(ws, f, settings)
    assert d["sales"].date.min() >= f.date_from and d["sales"].date.max() <= f.date_to


def test_csv_zip_and_pdf_outputs(ws, settings):
    ws.run_forecast(settings)
    t, _, lines = build_tables(ws, settings, None)
    assert {"data_quality_report", "forecast_outputs", "model_evaluation", "inventory_risk_report", "replenishment_recommendations", "executive_summary"} <= set(t)
    z = zipfile.ZipFile(io.BytesIO(to_zip_bytes(t)))
    assert "forecast_outputs.csv" in z.namelist()
    back = pd.read_csv(io.BytesIO(to_csv_bytes(t["forecast_outputs"])))
    assert len(back) == len(t["forecast_outputs"])
    pdf = make_pdf(ws, settings, None)
    assert pdf[:4] == b"%PDF" and len(pdf) > 2000


def test_dataset_replacement_invalidates_everything(ws, settings, raw):
    ws.run_forecast(settings)
    assert ws.forecast is not None and ws.inventory_view(settings) is not None
    v0 = ws.version
    sales2 = raw["sales"].copy()
    sales2["units_sold"] = (sales2["units_sold"].astype(int) * 2).astype(str)
    ws.set_dataset("sales", sales2, "new_sales.csv", 10)
    assert ws.version > v0 and ws.forecast is None and ws.weekly is None and not ws.processed and ws.reports == {}
    assert ws.inventory_view(settings) is None           # no stale inventory view served
    ws.process(settings)
    new_total = ws.sales.units_sold.sum()
    assert new_total == 2 * int(raw["sales"]["units_sold"].astype(int).sum())
    assert ws.run_forecast(settings) is not None


def test_mapping_change_invalidates_and_remove_dataset(ws, settings):
    ws.run_forecast(settings)
    m = dict(ws.mapping["sales"])
    ws.set_mapping("sales", m)                           # unchanged mapping keeps results
    assert ws.forecast is not None
    m["promo_flag"] = None
    ws.set_mapping("sales", m)
    assert ws.forecast is None and not ws.processed
    ws.remove_dataset("inventory")
    assert "inventory" not in ws.raw


def test_invalid_dataset_blocks_processing_but_others_continue(raw, settings):
    w = Workspace()
    w.set_dataset("sales", raw["sales"], "s.csv", 1)
    w.set_dataset("inventory", raw["inventory"].drop(columns=["on_hand_units"]), "i.csv", 1)
    w.process(settings)
    assert w.reports["sales"].ok and not w.reports["inventory"].ok
    assert "inventory" not in w.clean and w.weekly is not None


def test_read_csv_bytes_keeps_text():
    df = read_csv_bytes(b"date,sku_id,units_sold\n2024-01-01,00123,5\n")
    assert df.loc[0, "sku_id"] == "00123"               # leading zeros preserved
