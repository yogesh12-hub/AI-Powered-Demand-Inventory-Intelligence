"""Builds the SKU-level inventory view shared by the dashboard, reports and API."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Settings
from src.inventory import overstock_analysis as oa
from src.inventory import replenishment as rp
from src.inventory import stockout_analysis as sa

RANK = {"Low": 0, "Medium": 1, "High": 2}

# Every column the dashboard reads. The view ALWAYS has all of them (blank when a SKU is not scored),
# so pages never crash when no SKU could be scored.
VIEW_COLUMNS = ["sku_id", "snapshot_date", "on_hand", "on_order", "lead_time_days", "reorder_point", "moq", "category", "subcategory",
                "unit_cost", "unit_price", "status", "forecast_weekly_avg", "daily_demand", "position", "lead_time_demand", "safety_stock",
                "target_stock", "coverage_days", "on_hand_coverage_days", "stockout_risk", "stockout_reason", "stockout_score", "overstock_risk",
                "overstock_reason", "overstock_score", "excess_units", "reorder_qty", "inventory_value", "lost_sales_value", "excess_value",
                "reorder_cost", "risk", "risk_type", "financial_exposure", "priority", "position_vs_target", "recommendation",
                "snapshot_age_days", "stale"]


def demand_over_days(weekly_fc: np.ndarray, days: float) -> float:
    """Cumulative forecast over `days`: weekly rates applied piece-wise; beyond the horizon the horizon mean is extended."""
    weekly_fc = np.asarray(weekly_fc, float)
    if len(weekly_fc) == 0 or days <= 0:
        return 0.0
    total, left, i = 0.0, float(days), 0
    mean = weekly_fc.mean()
    while left > 0:
        rate = weekly_fc[i] if i < len(weekly_fc) else mean
        take = min(7.0, left)
        total += rate * take / 7.0
        left -= take
        i += 1
    return float(total)


def build_inventory_view(inv: pd.DataFrame, forecasts: pd.DataFrame, info: pd.DataFrame, sigma: pd.Series,
                         settings: Settings, as_of: pd.Timestamp | None = None, what_if: dict | None = None) -> pd.DataFrame:
    """inv: latest snapshot per SKU. forecasts: sku_id, step, forecast. info: sku_info(). Returns one row per SKU
    with a `status` of 'scored' or the reason it could not be scored (never silently imputed)."""
    wi = {"demand_mult": 1.0, "lead_time_mult": 1.0, **(what_if or {})}
    fc = {k: g.sort_values("step")["forecast"].values * wi["demand_mult"] for k, g in forecasts.groupby("sku_id")}
    info = info.set_index("sku_id") if "sku_id" in info.columns else info
    rows = []
    for _, r in inv.iterrows():
        sku = r["sku_id"]
        row = {"sku_id": sku, "snapshot_date": r.get("date"), "on_hand": r["on_hand_units"],
               "on_order": r.get("on_order_units", np.nan), "lead_time_days": r.get("lead_time_days", np.nan),
               "reorder_point": r.get("reorder_point", np.nan), "moq": r.get("moq", np.nan)}
        inf = info.loc[sku] if sku in info.index else None
        row["category"] = inf["category"] if inf is not None else np.nan
        row["subcategory"] = inf["subcategory"] if inf is not None else np.nan
        row["unit_cost"] = inf["unit_cost"] if inf is not None else np.nan
        row["unit_price"] = inf["unit_price"] if inf is not None else np.nan
        reason = None
        if pd.isna(row["on_order"]):
            if settings.assume_zero_on_order:
                row["on_order"] = 0.0
            else:
                reason = "on_order_units missing (enable 'treat blank on-order as 0' in Settings only if that is true)"
        if reason is None and (pd.isna(row["lead_time_days"]) or row["lead_time_days"] <= 0):
            reason = "lead_time_days missing or not positive"
        if reason is None and sku not in fc:
            reason = "No forecast for this SKU (its SKU ID is not in the sales data, or it has too little history)"
        if reason:
            row["status"] = f"Not scored: {reason}"
            rows.append(row)
            continue

        lt = float(row["lead_time_days"]) * wi["lead_time_mult"]
        row["lead_time_days"] = lt
        f = fc[sku]
        daily = float(f.mean() / 7.0)
        position = float(row["on_hand"] + row["on_order"])
        ltd = demand_over_days(f, lt)
        ss = rp.safety_stock(settings.z, float(sigma.get(sku, 0.0)), lt)
        so, so_reason = sa.classify_stockout(position, ltd, ss, float(row["on_hand"]), row["reorder_point"], settings.stockout_high_ratio)
        ov, ov_reason, cov = oa.classify_overstock(position, daily, settings.overstock_medium_days, settings.overstock_high_days)
        target = demand_over_days(f, lt + settings.review_period_days) + ss
        qty = rp.reorder_quantity(target, position, row["moq"])
        exc = oa.excess_units(position, daily, settings.target_coverage_days)
        price, cost = row["unit_price"], row["unit_cost"]
        row.update({
            "status": "scored", "forecast_weekly_avg": daily * 7, "daily_demand": daily, "position": position,
            "lead_time_demand": ltd, "safety_stock": ss, "target_stock": target,
            "coverage_days": cov, "on_hand_coverage_days": row["on_hand"] / daily if daily > 0 else float("inf"),
            "stockout_risk": so, "stockout_reason": so_reason, "stockout_score": sa.stockout_score(position, ltd, ss),
            "overstock_risk": ov, "overstock_reason": ov_reason, "overstock_score": oa.overstock_score(position, exc),
            "excess_units": exc, "reorder_qty": qty,
            "inventory_value": row["on_hand"] * cost if pd.notna(cost) else np.nan,
            "lost_sales_value": max(0.0, ltd - position) * price if pd.notna(price) else np.nan,
            "excess_value": exc * cost if pd.notna(cost) else np.nan,
            "reorder_cost": qty * cost if pd.notna(cost) else np.nan,
        })
        row["risk"] = max(so, ov, key=RANK.get)
        row["risk_type"] = ("Stockout + overstock" if so != "Low" and ov != "Low" else "Stockout" if so != "Low"
                            else "Overstock" if ov != "Low" else "None")
        row["financial_exposure"] = np.nansum([row["lost_sales_value"], row["excess_value"]]) if not (
            pd.isna(row["lost_sales_value"]) and pd.isna(row["excess_value"])) else np.nan
        row["priority"] = rp.priority(so, qty)
        row["position_vs_target"] = position - target
        row["recommendation"] = rp.explain({**row, "position": position, "on_hand": row["on_hand"], "on_order": row["on_order"]},
                                           settings.service_level * 100, settings.review_period_days)
        if as_of is not None and pd.notna(row["snapshot_date"]):
            row["snapshot_age_days"] = (as_of - row["snapshot_date"]).days
            row["stale"] = row["snapshot_age_days"] > settings.stale_days
        rows.append(row)
    out = pd.DataFrame(rows)
    out = out.reindex(columns=list(dict.fromkeys(list(out.columns) + VIEW_COLUMNS)))
    out["stale"] = out["stale"].fillna(False).astype(bool)
    return out


def scored(view: pd.DataFrame) -> pd.DataFrame:
    if view is None or "status" not in view.columns:
        return pd.DataFrame(columns=VIEW_COLUMNS)
    return view[view["status"] == "scored"].copy()


def exposure_by_category(view: pd.DataFrame) -> pd.DataFrame:
    v = scored(view)
    if v.empty:
        return pd.DataFrame(columns=["category", "skus", "lost_sales_value", "excess_value", "reorder_cost", "financial_exposure"])
    v["category"] = v["category"].fillna("(uncategorised)")
    return v.groupby("category").agg(skus=("sku_id", "count"), lost_sales_value=("lost_sales_value", "sum"),
                                     excess_value=("excess_value", "sum"), reorder_cost=("reorder_cost", "sum"),
                                     financial_exposure=("financial_exposure", "sum")).reset_index()
