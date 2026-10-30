"""Data-quality summaries, cross-dataset checks, feature availability and alerts."""
from __future__ import annotations

import numpy as np
import pandas as pd


def dataset_summary(raw_mapped: pd.DataFrame, clean: pd.DataFrame | None) -> dict:
    return {"rows_raw": len(raw_mapped), "rows_clean": None if clean is None else len(clean),
            "columns": list(raw_mapped.columns),
            "exact_duplicates": int(raw_mapped.duplicated().sum()),
            "missing_cells": int(raw_mapped.isna().sum().sum())}


def cross_dataset_checks(clean: dict[str, pd.DataFrame]) -> list[dict]:
    out = []

    def add(level, msg, count=0):
        out.append({"level": level, "message": msg, "count": int(count)})

    sales, master, inv, cal = (clean.get(k) for k in ("sales", "sku_master", "inventory", "calendar"))
    if sales is not None and master is not None:
        miss = set(sales.sku_id) - set(master.sku_id)
        if miss:
            add("warning", f"{len(miss)} SKU(s) in sales are missing from the product dataset (category/cost unavailable for them).", len(miss))
        else:
            add("info", "Every sales SKU exists in the product dataset.")
    if sales is not None and inv is not None:
        a, b = set(inv.sku_id) - set(sales.sku_id), set(sales.sku_id) - set(inv.sku_id)
        if not (set(inv.sku_id) & set(sales.sku_id)):
            add("warning", "None of the inventory SKU IDs match any SKU ID in the sales data, so inventory risk cannot be scored. "
                "Check that both use the same SKU IDs.", len(set(inv.sku_id)))
        if a:
            add("warning", f"{len(a)} inventory SKU(s) have no sales history and cannot be forecast.", len(a))
        if b:
            add("info", f"{len(b)} sold SKU(s) have no inventory snapshot and are not scored for risk.", len(b))
        gap = (sales.date.max() - inv.date.max()).days
        if abs(gap) > 14:
            add("warning", f"Latest inventory snapshot ({inv.date.max().date()}) and latest sales ({sales.date.max().date()}) differ by {abs(gap)} days.", abs(gap))
    if sales is not None and cal is not None:
        outside = (~sales.date.isin(cal.date)).sum()
        if outside:
            add("warning", f"{int(outside)} sales row(s) fall on dates missing from the calendar dataset.", outside)
    if sales is not None:
        full = pd.date_range(sales.date.min(), sales.date.max())
        miss_days = len(full) - sales.date.nunique()
        if miss_days:
            add("warning", f"{miss_days} calendar day(s) between the first and last sale have no sales rows at all (possible data gaps).", miss_days)
    return out


def feature_availability(clean: dict, weekly: pd.DataFrame | None, min_weeks: int) -> dict[str, tuple[bool, str]]:
    s, m, c, i = (clean.get(k) for k in ("sales", "sku_master", "calendar", "inventory"))
    weeks = weekly["week_start"].nunique() if weekly is not None and len(weekly) else 0
    return {
        "Sales analytics": (s is not None, "Upload and process the sales dataset."),
        "Category / subcategory filters": (m is not None and "category" in m.columns, "Needs a product dataset with a category column."),
        "Seasonality": (s is not None and (s.date.max() - s.date.min()).days >= 365, "Needs at least 12 months of sales."),
        "Promotion analysis": ((s is not None and "promo_flag" in s.columns) or (c is not None and "promo_event_flag" in c.columns),
                               "Needs promo_flag in sales or promo_event in the calendar."),
        "Demand forecasting": (weeks >= min_weeks, f"Needs {min_weeks}+ complete weeks of sales (found {weeks})."),
        "Inventory risk & replenishment": (i is not None and s is not None and {"on_order_units", "lead_time_days"} <= set(i.columns),
                                           "Needs sales plus inventory snapshots with on_order_units and lead_time_days."),
        "Cost-based financial exposure": (m is not None and "unit_cost" in m.columns, "Needs unit_cost in the product dataset."),
    }


def build_alerts(ws, settings) -> pd.DataFrame:
    """Alerts derived only from processed uploads / computed results."""
    rows = []

    def add(level, kind, msg, sku="", action=""):
        rows.append({"severity": level, "type": kind, "sku_id": sku, "message": msg, "action": action})

    for name, rep in ws.reports.items():
        for i in rep.issues:
            if i.level in ("error", "warning"):
                add("High" if i.level == "error" else "Medium", "Data quality", f"[{name}] {i.message}", action=i.fix or "Review on the Data Quality page.")
    for c in ws.cross:
        if c["level"] == "warning":
            add("Medium", "Data quality", c["message"], action="Check that all datasets cover the same products and period.")
    fr = ws.forecast
    if fr is not None:
        for w in fr.warnings:
            add("Medium", "Model evaluation", w)
        if len(fr.excluded):
            for _, r in fr.excluded.head(200).iterrows():
                add("Low", "Insufficient forecast history", r["reason"], r["sku_id"], "Collect more history or relax the minimum in Settings.")
    v = ws.last_inventory_view
    if v is not None and len(v):
        sc = v[v.status == "scored"]
        for _, r in sc[sc.stockout_risk == "High"].iterrows():
            add("High", "Stockout risk", r["stockout_reason"], r["sku_id"], f"Order about {r['reorder_qty']:,.0f} units.")
        for _, r in sc[sc.overstock_risk == "High"].iterrows():
            add("Medium", "Excess inventory", r["overstock_reason"], r["sku_id"], "Pause reorders; consider markdown or promotion.")
        for _, r in v[v.status != "scored"].head(200).iterrows():
            add("Low", "Not scored", r["status"], r["sku_id"], "Complete the missing inventory fields.")
        st = sc[sc["stale"]]
        if len(st):
            add("Medium", "Stale inventory", f"{len(st)} SKU snapshot(s) are older than {settings.stale_days} days relative to the latest sales date.",
                action="Upload a fresher inventory snapshot.")
    order = {"High": 0, "Medium": 1, "Low": 2}
    df = pd.DataFrame(rows, columns=["severity", "type", "sku_id", "message", "action"])
    return df.sort_values("severity", key=lambda s: s.map(order)).reset_index(drop=True) if len(df) else df


# ---------------------------------------------------------------------------------------------
# Missing-data report: what is absent, and exactly which features it affects
# ---------------------------------------------------------------------------------------------
IMPACT = {
    "sales": [("date", "all analysis"), ("sku_id", "all analysis"), ("units_sold", "all analysis"),
              ("revenue", "revenue KPIs and lost-sales value (unit_price x units is used if available)"),
              ("promo_flag", "promotion analysis")],
    "sku_master": [("sku_id", "joining product details"), ("category", "category / subcategory filters and category charts"),
                   ("unit_cost", "inventory value, excess value and reorder cost in INR"), ("list_price", "fallback selling price")],
    "calendar": [("date", "calendar analysis"), ("holiday", "holiday effect analysis"), ("promo_event", "promo-event effect analysis")],
    "inventory": [("date", "inventory risk"), ("sku_id", "inventory risk"), ("on_hand_units", "inventory risk"),
                  ("on_order_units", "risk scoring and reorder quantities (SKUs without it are not scored)"),
                  ("lead_time_days", "risk scoring and reorder quantities (SKUs without it are not scored)")],
}
DATASET_NAMES = {"sales": "Sales", "sku_master": "Products", "calendar": "Calendar", "inventory": "Inventory"}
UNLOCKS = {"sales": "Sales analytics, forecasting and everything else", "sku_master": "Category filters and cost-based INR values",
           "calendar": "Holiday and promo-event analysis", "inventory": "Inventory risk, replenishment and stock alerts"}


def missing_data_report(ws) -> pd.DataFrame:
    """One row per dataset: status, what is missing, and what that limits. Analysis still runs on whatever is available."""
    rows = []
    for ds in DATASET_NAMES:
        ok = ds in ws.clean
        cols = set(ws.clean[ds].columns) if ok else set()
        if ds not in ws.raw:
            status, miss = "Missing", [f for f, _ in IMPACT[ds]]
        elif ds in ws.reports and not ws.reports[ds].ok:
            status, miss = "Needs fixes", [i.message for i in ws.reports[ds].errors][:3]
        elif not ok:
            status, miss = "Not processed", []
        else:
            miss = [f for f, _ in IMPACT[ds] if f not in cols and not (f == "holiday" and "holiday_flag" in cols) and not (f == "promo_event" and "promo_event_flag" in cols)]
            status = "Partial" if miss else "OK"
        affects = sorted({imp for f, imp in IMPACT[ds] if f in miss}) if status in ("Missing", "Partial") else ([UNLOCKS[ds]] if status != "OK" else [])
        rows.append({"dataset": DATASET_NAMES[ds], "status": status, "missing": ", ".join(miss) if miss else "-",
                     "limits": "; ".join(affects) if affects else "-"})
    return pd.DataFrame(rows)
