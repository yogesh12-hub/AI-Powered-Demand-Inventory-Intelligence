"""Split ONE uploaded file into the four datasets.

Two layouts are supported:
* Flat CSV / single sheet: one table whose columns can mix sales, product, calendar and inventory fields.
* Excel workbook with several sheets: sheets are matched to datasets by name (sales / sku_master|product / calendar / inventory|stock).

Nothing is invented: a dataset is created only when its required columns are present, and every split decision is returned as a note.
"""
from __future__ import annotations

import io

import pandas as pd

from src.config import DATASETS
from src.data.column_mapping import _norm, apply_mapping, auto_map
from src.data.schema_validation import parse_dates

SHEET_HINTS = {"sales": ("sales", "sale", "transaction", "order"), "sku_master": ("master", "product", "sku", "item"),
               "calendar": ("calendar", "cal", "event", "holiday"), "inventory": ("inventory", "stock", "snapshot")}


def split_flat(df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], list[str]]:
    out: dict[str, pd.DataFrame] = {}
    notes: list[str] = []
    cols = list(df.columns)
    maps = {n: auto_map(cols, n) for n in DATASETS}
    claimed = {s for s in maps["sales"].values() if s}
    for n in ("sku_master", "calendar", "inventory"):  # avoid one ambiguous column (e.g. "price") feeding two datasets
        for t, s in list(maps[n].items()):
            if s and s in claimed and t not in ("date", "sku_id") and _norm(s) != _norm(t):
                maps[n][t] = None
    date_src = maps["sales"].get("date") or maps["inventory"].get("date")

    def has(n, *need):
        return all(maps[n].get(c) for c in need)

    if has("sales", "date", "sku_id", "units_sold"):
        s = apply_mapping(df, maps["sales"])
        k = s["units_sold"].notna()
        out["sales"] = s[k].reset_index(drop=True)
        notes.append(f"sales: {int(k.sum()):,} rows with units_sold" + (f" ({int((~k).sum()):,} rows without units_sold skipped)." if (~k).any() else "."))
    else:
        notes.append("sales: NOT created - needs date, sku_id and units_sold columns.")

    attrs = [c for c in DATASETS["sku_master"]["optional"] if maps["sku_master"].get(c)]
    if maps["sku_master"].get("sku_id") and attrs:
        m = apply_mapping(df, {**maps["sku_master"], **({"date": date_src} if date_src else {})})
        if "date" in m.columns:
            m = m.assign(_d=parse_dates(m["date"])).sort_values("_d", kind="stable").drop(columns=["_d", "date"])
        m = m.groupby("sku_id", sort=False).last().reset_index()  # latest non-blank value per SKU
        out["sku_master"] = m
        notes.append(f"sku_master: {len(m):,} SKUs (product columns: {', '.join(attrs)}); if a value changes over time the latest non-blank value is used.")
    else:
        notes.append("sku_master: not created - no product columns (category, unit_cost, ...) found.")

    if has("inventory", "date", "sku_id", "on_hand_units"):
        i = apply_mapping(df, maps["inventory"])
        k = i["on_hand_units"].notna()
        out["inventory"] = i[k].reset_index(drop=True)
        notes.append(f"inventory: {int(k.sum()):,} snapshot rows with on_hand_units" + ("" if "on_order_units" in i and "lead_time_days" in i else
                     " - WARNING: on_order_units and/or lead_time_days missing, so risk scoring will be unavailable."))
    else:
        notes.append("inventory: not created - needs date, sku_id and on_hand_units.")

    cal = [c for c in DATASETS["calendar"]["optional"] if maps["calendar"].get(c)]
    if maps["calendar"].get("date") and cal:
        c_ = apply_mapping(df, maps["calendar"])
        c_ = c_.groupby("date", sort=False).first().reset_index()  # first non-blank value per date
        out["calendar"] = c_
        notes.append(f"calendar: {len(c_):,} dates (columns: {', '.join(cal)}).")
    else:
        notes.append("calendar: not created - no week/month/season/holiday/promo_event columns found.")
    return out, notes


def split_workbook(sheets: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], list[str]]:
    if len(sheets) == 1:
        return split_flat(next(iter(sheets.values())))
    out, notes, used = {}, [], set()
    for ds, hints in SHEET_HINTS.items():
        for name, df in sheets.items():
            if name not in used and any(h in name.lower() for h in hints):
                out[ds] = df
                used.add(name)
                notes.append(f"{ds}: sheet '{name}' ({len(df):,} rows).")
                break
        else:
            notes.append(f"{ds}: no sheet with a matching name.")
    notes += [f"Ignored sheet '{n}' (name did not match any dataset)." for n in sheets if n not in used]
    return out, notes


def read_combined(data: bytes, filename: str) -> tuple[dict[str, pd.DataFrame], list[str]]:
    from src.data.upload_manager import read_csv_bytes
    if filename.lower().endswith((".xlsx", ".xls")):
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, dtype=str)
        return split_workbook({k: v for k, v in sheets.items() if len(v)})
    return split_flat(read_csv_bytes(data))


def merge_files(files: list[tuple[str, bytes]]) -> tuple[dict[str, tuple[pd.DataFrame, str, int]], list[str]]:
    """The single upload entry point. Each file (CSV or Excel; flat or multi-sheet) is split into datasets; datasets are merged.
    If two files provide the same dataset, the one with more rows is kept and the user is told.
    Returns ({dataset: (frame, source_file, size_bytes)}, notes)."""
    best: dict[str, tuple[pd.DataFrame, str, int]] = {}
    notes: list[str] = []
    not_created: dict[str, str] = {}
    for name, data in files:
        try:
            parts, n = read_combined(data, name)
        except Exception as e:  # noqa: BLE001 - report unreadable files instead of crashing
            notes.append(f"{name}: could not be read ({e}).")
            continue
        for line in n:
            ds = line.split(":", 1)[0]
            if "not created" in line.lower() or "no sheet" in line.lower():
                not_created.setdefault(ds, f"{name} -> {line}")
            else:
                notes.append(f"{name} -> {line}")
        for ds, df in parts.items():
            if ds in best:
                keep_new = len(df) > len(best[ds][0])
                notes.append(f"{ds}: provided by both {best[ds][1]} and {name}; kept the one with more rows ({name if keep_new else best[ds][1]}).")
                if keep_new:
                    best[ds] = (df, name, len(data))
            else:
                best[ds] = (df, name, len(data))
    notes += [msg for ds, msg in not_created.items() if ds not in best and ds in DATASETS]
    return best, notes
