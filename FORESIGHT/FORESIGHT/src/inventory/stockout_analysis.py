"""Stockout risk rules (documented in docs/methodology.md)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def classify_stockout(position: float, lt_demand: float, safety_stock: float, on_hand: float,
                      reorder_point: float | None, high_ratio: float = 1.0) -> tuple[str, str]:
    """High   : inventory position < high_ratio x lead-time demand (cannot cover demand until a new order lands)
       Medium : position < lead-time demand + safety stock, or on-hand <= existing reorder point
       Low    : otherwise (also when forecast demand is zero)"""
    if lt_demand <= 0:
        return "Low", "No forecast demand during lead time."
    if position < high_ratio * lt_demand:
        return "High", (f"Position {position:,.0f} (on-hand + on-order) is below expected lead-time demand {lt_demand:,.0f}.")
    if position < lt_demand + safety_stock:
        return "Medium", (f"Position {position:,.0f} covers lead-time demand {lt_demand:,.0f} but not the safety buffer {safety_stock:,.0f}.")
    if reorder_point is not None and np.isfinite(reorder_point) and reorder_point > 0 and on_hand <= reorder_point:
        return "Medium", f"On-hand {on_hand:,.0f} is at or below the existing reorder point {reorder_point:,.0f}."
    return "Low", "Position covers lead-time demand plus safety stock."


def stockout_score(position, lt_demand, safety_stock) -> float:
    need = lt_demand + safety_stock
    return float(100 * np.clip(1 - position / need, 0, 1)) if need > 0 else 0.0


def summary(view: pd.DataFrame) -> pd.Series:
    return view["stockout_risk"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
