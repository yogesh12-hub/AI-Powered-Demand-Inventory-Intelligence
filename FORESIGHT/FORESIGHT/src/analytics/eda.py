"""Business insights computed from the uploaded data. An insight appears only if the data supports it."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics import sales_analysis as sa
from src.analytics import seasonality as se
from src.utils.helpers import inr, pct


def generate_insights(sales: pd.DataFrame, weekly: pd.DataFrame | None, calendar: pd.DataFrame | None = None,
                      inventory_latest: pd.DataFrame | None = None, dead_days: int = 90) -> tuple[list[dict], list[str]]:
    """Returns (insights, unsupported_notes). Each insight: title, evidence, relevance."""
    ins, notes = [], []
    if sales.empty:
        return ins, ["No sales rows match the current filters."]
    # 1. concentration
    s = sa.sku_summary(sales)
    key = "revenue" if s["revenue"].notna().any() else "units"
    if len(s) >= 10 and s[key].sum() > 0:
        s = s.sort_values(key, ascending=False)
        top_n = max(1, int(np.ceil(len(s) * 0.2)))
        share = s[key].head(top_n).sum() / s[key].sum() * 100
        ins.append({"title": f"Top {top_n} SKUs (20%) generate {pct(share)} of {key}",
                    "evidence": f"{top_n} of {len(s)} SKUs account for {pct(share)} of total {key} in the selected period.",
                    "relevance": "Stock-outs on these SKUs cost the most; prioritise their safety stock and replenishment cadence."})
    else:
        notes.append("Sales concentration needs at least 10 SKUs.")
    # 2. seasonality
    mi = se.monthly_index(sales)
    if mi is not None:
        hi, lo = mi.loc[mi["index"].idxmax()], mi.loc[mi["index"].idxmin()]
        ins.append({"title": f"Demand peaks in {hi.month_name} ({hi['index']:.2f}x average) and dips in {lo.month_name} ({lo['index']:.2f}x)",
                    "evidence": f"Average monthly units by calendar month, full months only; index 1.00 = typical month.",
                    "relevance": f"Build inventory ahead of {hi.month_name} and avoid large orders landing before {lo.month_name}."})
    else:
        notes.append("Monthly seasonality needs at least 12 full months covering every calendar month.")
    # 3. trend
    if weekly is not None and weekly["week_start"].nunique() >= 26:
        t = se.weekly_total(weekly)
        last13, prev13 = t.iloc[-13:].sum(), t.iloc[-26:-13].sum()
        if prev13 > 0:
            g = (last13 / prev13 - 1) * 100
            ins.append({"title": f"Units over the last 13 weeks are {pct(abs(g))} {'above' if g >= 0 else 'below'} the previous 13",
                        "evidence": f"{last13:,.0f} vs {prev13:,.0f} units (complete weeks only).",
                        "relevance": "A sustained shift changes the lead-time demand used for reorder quantities; confirm it is not a one-off before ordering."})
    else:
        notes.append("Trend comparison needs at least 26 complete weeks.")
    # 4. dead stock
    as_of = sales["date"].max()
    d = sa.dead_stock(sales, as_of, dead_days, inventory_latest)
    if len(d):
        txt = f"{len(d)} SKU(s) had no sales in the last {dead_days} days."
        if "on_hand_units" in d.columns and d["on_hand_units"].notna().any():
            txt += f" They still hold {d['on_hand_units'].sum():,.0f} units on hand."
        ins.append({"title": f"{len(d)} inactive SKU(s) with no sales for {dead_days}+ days", "evidence": txt,
                    "relevance": "Inactive items tie up cash and warehouse space; review for markdown, return-to-vendor or delisting, and stop reordering."})
    # 5. promo
    pe = sa.promo_effect(sales)
    if pe:
        ins.append({"title": f"Promo days average {pct(abs(pe['uplift_pct']))} {'more' if pe['uplift_pct'] >= 0 else 'fewer'} units per SKU-day than non-promo days",
                    "evidence": f"{pe['promo_avg']:.2f} vs {pe['base_avg']:.2f} units ({pe['promo_rows']:,} promo vs {pe['base_rows']:,} regular SKU-days). Descriptive, not causal.",
                    "relevance": "Planned promotions should lift the demand forecast; reserve stock before the event."})
    else:
        notes.append("Promotion effect needs a promo_flag with enough promo and non-promo rows.")
    # 6. variability
    if weekly is not None and len(weekly):
        v = sa.demand_variability(weekly)
        v = v[v["count"] >= 8]
        if len(v):
            share = (v["cv"] > 1).mean() * 100
            ins.append({"title": f"{pct(share)} of SKUs have highly variable weekly demand (CV > 1)",
                        "evidence": f"Coefficient of variation of weekly units across {len(v)} SKUs with 8+ weeks of history.",
                        "relevance": "High variability needs larger safety stock or more frequent review; forecasts for these SKUs are less reliable."})
    return ins, notes
