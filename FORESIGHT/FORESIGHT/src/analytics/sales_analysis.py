"""Sales analytics on cleaned daily data (already filtered by the caller)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def kpis(sales: pd.DataFrame) -> dict:
    rev = sales["revenue"]
    return {"units": float(sales["units_sold"].sum()),
            "revenue": float(rev.sum()) if rev.notna().any() else None,
            "skus": int(sales["sku_id"].nunique()), "days": int(sales["date"].nunique()),
            "start": sales["date"].min(), "end": sales["date"].max()}


def trend(sales: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    rule = {"Daily": "D", "Weekly": "W-SUN", "Monthly": "MS"}.get(freq, freq)
    g = sales.set_index("date").resample(rule).agg({"units_sold": "sum", "revenue": lambda x: x.sum(min_count=1)})
    return g.reset_index()


def sku_summary(sales: pd.DataFrame) -> pd.DataFrame:
    g = sales.groupby("sku_id").agg(units=("units_sold", "sum"), revenue=("revenue", lambda x: x.sum(min_count=1)),
                                    last_sale=("date", "max"), active_days=("date", "nunique"))
    if "category" in sales.columns:
        g["category"] = sales.groupby("sku_id")["category"].first()
    return g.reset_index()


def top_low(sales: pd.DataFrame, n: int = 10, by: str = "units"):
    s = sku_summary(sales)
    if by == "revenue" and s["revenue"].isna().all():
        by = "units"
    s = s.sort_values(by, ascending=False)
    return s.head(n), s.tail(n).sort_values(by)


def category_performance(sales: pd.DataFrame) -> pd.DataFrame | None:
    if "category" not in sales.columns or sales["category"].isna().all():
        return None
    return (sales.assign(category=sales["category"].fillna("(uncategorised)")).groupby("category")
            .agg(units=("units_sold", "sum"), revenue=("revenue", lambda x: x.sum(min_count=1)), skus=("sku_id", "nunique"))
            .reset_index().sort_values("units", ascending=False))


def dead_stock(sales_all: pd.DataFrame, as_of: pd.Timestamp, days: int, inventory_latest: pd.DataFrame | None = None) -> pd.DataFrame:
    """SKUs with no units sold in the last `days` days (relative to the latest sales date).
    If inventory is available, shows units held and tied-up value."""
    last = sales_all.groupby("sku_id")["date"].max()
    dead = last[last < as_of - pd.Timedelta(days=days)].rename("last_sale").reset_index()
    dead["days_since_sale"] = (as_of - dead["last_sale"]).dt.days
    if inventory_latest is not None and len(inventory_latest):
        cols = [c for c in ("sku_id", "on_hand_units") if c in inventory_latest.columns]
        dead = dead.merge(inventory_latest[cols], on="sku_id", how="left")
    return dead.sort_values("days_since_sale", ascending=False)


def demand_variability(weekly: pd.DataFrame) -> pd.DataFrame:
    g = weekly.groupby("sku_id")["units"].agg(["mean", "std", "count"])
    g["cv"] = g["std"] / g["mean"].replace(0, np.nan)
    return g.reset_index()


def promo_effect(sales: pd.DataFrame) -> dict | None:
    """Average units per SKU-day on promo vs non-promo days (needs both groups). Descriptive, not causal."""
    if "promo_flag" not in sales.columns or sales["promo_flag"].isna().all():
        return None
    p = sales[sales["promo_flag"].astype("boolean").fillna(False)]["units_sold"]
    n = sales[~sales["promo_flag"].astype("boolean").fillna(False)]["units_sold"]
    if len(p) < 30 or len(n) < 30 or n.mean() == 0:
        return None
    return {"promo_avg": float(p.mean()), "base_avg": float(n.mean()), "uplift_pct": float((p.mean() / n.mean() - 1) * 100),
            "promo_rows": int(len(p)), "base_rows": int(len(n))}


def event_effect(sales: pd.DataFrame, calendar: pd.DataFrame | None, col: str) -> dict | None:
    """Average daily units on calendar event days (holiday_flag / promo_event_flag) vs other days."""
    if calendar is None or col not in calendar.columns:
        return None
    day = sales.groupby("date")["units_sold"].sum().reset_index().merge(calendar[["date", col]], on="date", how="inner", validate="1:1")
    ev, no = day[day[col]]["units_sold"], day[~day[col]]["units_sold"]
    if len(ev) < 3 or len(no) < 3 or no.mean() == 0:
        return None
    return {"event_avg": float(ev.mean()), "other_avg": float(no.mean()), "uplift_pct": float((ev.mean() / no.mean() - 1) * 100),
            "event_days": int(len(ev)), "other_days": int(len(no))}
