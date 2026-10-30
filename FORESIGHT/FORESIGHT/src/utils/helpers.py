"""Small shared helpers (formatting, filtering)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import CURRENCY


def inr(x, decimals: int = 0) -> str:
    """Format a number as Indian rupees with lakh/crore grouping (e.g. ₹12,34,567)."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    neg = x < 0
    x = abs(float(x))
    whole, frac = f"{x:.{decimals}f}".split(".") if decimals else (f"{x:.0f}", "")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        whole = ",".join(parts + [tail])
    out = f"{CURRENCY}{whole}" + (f".{frac}" if frac else "")
    return f"-{out}" if neg else out


def num(x, decimals: int = 0) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:,.{decimals}f}"


def pct(x, decimals: int = 1) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{decimals}f}%"


@dataclass
class Filters:
    date_from: pd.Timestamp | None = None
    date_to: pd.Timestamp | None = None
    categories: list = field(default_factory=list)
    subcategories: list = field(default_factory=list)
    skus: list = field(default_factory=list)


def apply_filters(df: pd.DataFrame | None, f: Filters | None, date_col: str | None = "date") -> pd.DataFrame | None:
    """Apply date/category/subcategory/SKU filters. Columns that don't exist are skipped."""
    if df is None or f is None:
        return df
    out = df
    if date_col and date_col in out.columns:
        if f.date_from is not None:
            out = out[out[date_col] >= f.date_from]
        if f.date_to is not None:
            out = out[out[date_col] <= f.date_to]
    if f.categories and "category" in out.columns:
        out = out[out["category"].isin(f.categories)]
    if f.subcategories and "subcategory" in out.columns:
        out = out[out["subcategory"].isin(f.subcategories)]
    if f.skus and "sku_id" in out.columns:
        out = out[out["sku_id"].isin(f.skus)]
    return out


def normalize_sku(s: pd.Series) -> pd.Series:
    """Single SKU normalisation rule used by every dataset: trim, collapse spaces, upper-case."""
    return s.astype("string").str.strip().str.replace(r"\s+", " ", regex=True).str.upper()
