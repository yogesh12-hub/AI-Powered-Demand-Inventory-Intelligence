"""Preprocessing: joins, daily→weekly aggregation, snapshot selection."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Settings


def week_start(d: pd.Series) -> pd.Series:
    return (d - pd.to_timedelta(d.dt.dayofweek, unit="D")).dt.normalize()


def enrich_sales(sales: pd.DataFrame, master: pd.DataFrame | None, log: list | None = None) -> pd.DataFrame:
    """Left-join product attributes. Validates master is unique per SKU so rows are never duplicated."""
    out = sales.copy()
    if "revenue" not in out.columns:
        out["revenue"] = np.nan
    if "unit_price" not in out.columns:
        out["unit_price"] = np.nan
    if "promo_flag" not in out.columns:
        out["promo_flag"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    if master is not None:
        assert master["sku_id"].is_unique, "sku_master must be unique per sku_id before joining"
        cols = [c for c in ("sku_id", "category", "subcategory", "unit_cost", "list_price", "launch_date") if c in master.columns]
        n = len(out)
        out = out.merge(master[cols], on="sku_id", how="left", validate="m:1")
        assert len(out) == n, "join changed row count"
    return out


def build_weekly(sales: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, list[dict]]:
    """Aggregate to complete Monday-start weeks per SKU.

    * Only complete weeks (all 7 days inside the data range) are kept - partial edge weeks would look like demand drops.
    * Weeks without any sales row are zero ONLY when settings.missing_weeks_as_zero is True (documented assumption);
      otherwise they stay NaN and forecasting excludes those SKUs.
    * Revenue stays NaN when neither revenue nor price x units is available.
    """
    log: list[dict] = []
    s = sales.copy()
    miss_rev = s["revenue"].isna() & s["unit_price"].notna()
    s.loc[miss_rev, "revenue"] = s.loc[miss_rev, "units_sold"] * s.loc[miss_rev, "unit_price"]
    if miss_rev.any():
        log.append({"step": "revenue_from_price", "rows_affected": int(miss_rev.sum()), "action": "revenue = units_sold x unit_price",
                    "justification": "Revenue was blank but both factors exist."})
    s["week_start"] = week_start(s["date"])
    dmin, dmax = s["date"].min(), s["date"].max()
    first_w = week_start(pd.Series([dmin])).iloc[0]
    if first_w < dmin:
        first_w += pd.Timedelta(weeks=1)
    last_w = week_start(pd.Series([dmax])).iloc[0]
    if last_w + pd.Timedelta(days=6) > dmax:
        last_w -= pd.Timedelta(weeks=1)
    dropped = s[(s["week_start"] < first_w) | (s["week_start"] > last_w)]
    s = s[(s["week_start"] >= first_w) & (s["week_start"] <= last_w)]
    log.append({"step": "complete_weeks", "rows_affected": int(len(dropped)), "action": f"Kept weeks {first_w.date()} to {last_w.date()}",
                "justification": "Partial first/last weeks are excluded so they are not mistaken for low demand."})
    if s.empty:
        return pd.DataFrame(columns=["sku_id", "week_start", "units", "revenue", "promo_share", "observed"]), log

    g = s.groupby(["sku_id", "week_start"]).agg(
        units=("units_sold", "sum"), revenue=("revenue", lambda x: x.sum(min_count=1)),
        promo_share=("promo_flag", lambda x: x.astype("float").mean() if x.notna().any() else np.nan),
        days=("date", "nunique")).reset_index()
    g["observed"] = True
    all_weeks = pd.date_range(first_w, last_w, freq="7D")
    frames, filled = [], 0
    for sku, grp in g.groupby("sku_id"):
        start = grp["week_start"].min()
        end = last_w if settings.missing_weeks_as_zero else grp["week_start"].max()
        idx = all_weeks[(all_weeks >= start) & (all_weeks <= end)]
        grp = grp.set_index("week_start").reindex(idx)
        miss = grp["units"].isna()
        if settings.missing_weeks_as_zero:
            filled += int(miss.sum())
            grp.loc[miss, "units"] = 0.0
            grp.loc[miss, "revenue"] = np.where(grp["revenue"].notna().any(), 0.0, np.nan)
        grp["observed"] = grp["observed"].fillna(False).astype(bool)
        grp["sku_id"] = sku
        grp.index.name = "week_start"
        frames.append(grp.reset_index())
    w = pd.concat(frames, ignore_index=True)[["sku_id", "week_start", "units", "revenue", "promo_share", "observed"]]
    log.append({"step": "missing_weeks", "rows_affected": filled,
                "action": "Filled with 0 units" if settings.missing_weeks_as_zero else "Left blank (NaN)",
                "justification": ("Setting 'missing weeks as zero' is ON: a week with no sales rows between a SKU's first sale and the last data week is treated as zero demand."
                                  if settings.missing_weeks_as_zero else "Setting is OFF: gaps stay blank and those SKUs are excluded from forecasting.")})
    return w, log


def attach_sku_attrs(weekly: pd.DataFrame, master: pd.DataFrame | None) -> pd.DataFrame:
    if master is None:
        return weekly
    cols = [c for c in ("sku_id", "category", "subcategory") if c in master.columns]
    n = len(weekly)
    out = weekly.merge(master[cols], on="sku_id", how="left", validate="m:1")
    assert len(out) == n
    return out


def latest_snapshot(inv: pd.DataFrame) -> pd.DataFrame:
    """Latest inventory row per SKU (the 'current' position)."""
    return inv.sort_values("date").groupby("sku_id", as_index=False).tail(1).reset_index(drop=True)


def sku_info(sales: pd.DataFrame, master: pd.DataFrame | None, recent_weeks: int = 8) -> pd.DataFrame:
    """Per-SKU category, cost and recent realised selling price (revenue/units over last N weeks)."""
    cut = sales["date"].max() - pd.Timedelta(weeks=recent_weeks)
    r = sales[sales["date"] > cut]
    r = r[(r["units_sold"] > 0) & r["revenue"].notna()]
    price = (r.groupby("sku_id")["revenue"].sum() / r.groupby("sku_id")["units_sold"].sum()).rename("recent_price")
    info = pd.DataFrame({"sku_id": sales["sku_id"].unique()}).merge(price.reset_index(), on="sku_id", how="left")
    if master is not None:
        info = info.merge(master, on="sku_id", how="outer", validate="1:1")
    else:
        for c in ("category", "subcategory", "unit_cost", "list_price"):
            info[c] = np.nan
    for c in ("category", "subcategory", "unit_cost", "list_price"):
        if c not in info.columns:
            info[c] = np.nan
    info["unit_price"] = info["recent_price"].fillna(info["list_price"])
    return info[["sku_id", "category", "subcategory", "unit_cost", "list_price", "recent_price", "unit_price"]]
