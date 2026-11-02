"""Overstock risk rules and excess-inventory valuation."""
from __future__ import annotations

import numpy as np
import pandas as pd


def classify_overstock(position: float, daily_demand: float, medium_days: float, high_days: float) -> tuple[str, str, float]:
    """Coverage = position / forecast daily demand. Returns (level, reason, coverage_days; inf when no demand)."""
    if position <= 0:
        return "Low", "No stock on hand or on order.", 0.0
    if daily_demand <= 0:
        return "High", f"{position:,.0f} units in stock/on order but zero forecast demand.", float("inf")
    cov = position / daily_demand
    if cov > high_days:
        return "High", f"Coverage {cov:,.0f} days exceeds the high threshold ({high_days:g} d).", cov
    if cov > medium_days:
        return "Medium", f"Coverage {cov:,.0f} days exceeds the medium threshold ({medium_days:g} d).", cov
    return "Low", f"Coverage {cov:,.0f} days is within policy.", cov


def excess_units(position: float, daily_demand: float, target_days: float) -> float:
    """Units above the target-coverage policy (everything is excess when demand is zero)."""
    return float(max(0.0, position - daily_demand * target_days))


def overstock_score(position, excess) -> float:
    return float(100 * np.clip(excess / position, 0, 1)) if position > 0 else 0.0


def summary(view: pd.DataFrame) -> pd.Series:
    return view["overstock_risk"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
