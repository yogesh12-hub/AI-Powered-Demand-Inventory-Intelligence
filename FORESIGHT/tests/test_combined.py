import io

import pandas as pd

from src.config import Settings
from src.data.combined_upload import read_combined, split_flat, split_workbook
from src.data.upload_manager import Workspace


def flat(raw):
    s, m, i = raw["sales"], raw["sku_master"].copy(), raw["inventory"]
    m["sku_id"] = m["sku_id"].str.lower()
    f = s.merge(m, on="sku_id", how="left")
    inv = i.assign(sku_id=i.sku_id.str.lower())
    return pd.concat([f, inv], ignore_index=True)       # inventory rows have no units_sold


def test_flat_file_splits_into_datasets(raw):
    parts, notes = split_flat(flat(raw))
    assert {"sales", "sku_master", "inventory"} <= set(parts)
    assert parts["sales"].units_sold.notna().all() and len(parts["sales"]) == len(raw["sales"])
    assert len(parts["sku_master"]) == 10 and parts["sku_master"].sku_id.is_unique
    assert len(parts["inventory"]) == 10 and "calendar" not in parts and notes


def test_flat_file_runs_end_to_end(raw):
    parts, _ = split_flat(flat(raw))
    ws = Workspace()
    for k, df in parts.items():
        ws.set_dataset(k, df, "one.csv", 1)
    st = Settings(backtest_folds=2)
    ws.process(st)
    assert all(r.ok for r in ws.reports.values())
    ws.run_forecast(st)
    assert (ws.inventory_view(st).status == "scored").all()


def test_missing_required_columns_do_not_create_dataset():
    parts, notes = split_flat(pd.DataFrame({"date": ["2024-01-01"], "category": ["x"]}))
    assert "sales" not in parts and any("NOT created" in n for n in notes)


def test_excel_workbook_sheets_and_csv_path(raw, tmp_path):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as w:
        raw["sales"].to_excel(w, sheet_name="Sales", index=False)
        raw["sku_master"].to_excel(w, sheet_name="Products", index=False)
        raw["inventory"].to_excel(w, sheet_name="Inventory", index=False)
        raw["sales"].head(1).to_excel(w, sheet_name="Notes", index=False)
    parts, notes = read_combined(buf.getvalue(), "all.xlsx")
    assert {"sales", "sku_master", "inventory"} == set(parts) and any("Ignored sheet 'Notes'" in n for n in notes)
    csv = flat(raw).to_csv(index=False).encode()
    assert "sales" in read_combined(csv, "all.csv")[0]
