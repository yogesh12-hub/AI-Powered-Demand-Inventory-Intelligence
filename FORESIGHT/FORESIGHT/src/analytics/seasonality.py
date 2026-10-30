"""Seasonality: only reported when the data supports it."""
from __future__ import annotations

import numpy as np
import pandas as pd


def monthly_index(sales: pd.DataFrame, min_months: int = 12) -> pd.DataFrame | None:
    """Seasonal index per calendar month = mean monthly units / overall mean monthly units. Needs >= min_months distinct months
    and every calendar month present; the first/last (possibly partial) months are excluded."""
    if sales.empty:
        return None
    m = sales.set_index("date")["units_sold"].resample("MS").sum()
    if len(m) > 2:
        m = m.iloc[1:-1]
    if len(m) < min_months or m.index.month.nunique() < 12:
        return None
    by = m.groupby(m.index.month).mean()
    out = (by / by.mean()).rename("index").reset_index().rename(columns={"date": "month"})
    out["month_name"] = pd.to_datetime(out["month"], format="%m").dt.strftime("%b")
    return out


def weekly_total(weekly: pd.DataFrame) -> pd.Series:
    return weekly.groupby("week_start")["units"].sum()


def seasonal_strength(weekly: pd.DataFrame, period: int = 52) -> float | None:
    """Correlation of total weekly units with itself one year earlier (None if < period+8 weeks)."""
    s = weekly_total(weekly)
    if len(s) < period + 8:
        return None
    v = s.values
    c = np.corrcoef(v[period:], v[:-period])[0, 1]
    return float(c) if np.isfinite(c) else None
