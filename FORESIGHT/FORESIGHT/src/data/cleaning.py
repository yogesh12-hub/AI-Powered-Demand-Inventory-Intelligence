"""Cleaning. Every decision is appended to a log; nothing is dropped or filled silently."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DATASETS
from src.data.schema_validation import TRUE_SET, parse_dates, to_numeric
from src.utils.helpers import normalize_sku


def _log(log: list, step: str, rows: int, action: str, why: str):
    log.append({"step": step, "rows_affected": int(rows), "action": action, "justification": why})


def clean_dataset(df: pd.DataFrame, dataset: str, dayfirst: bool = False) -> tuple[pd.DataFrame, list[dict]]:
    spec = DATASETS[dataset]
    log: list[dict] = []
    d = df.copy()
    n0 = len(d)
    _log(log, "load", n0, "Rows received", "Starting row count after column mapping.")

    if "sku_id" in d.columns:
        d["sku_id"] = normalize_sku(d["sku_id"])
        blank = d["sku_id"].isna() | (d["sku_id"] == "")
        if blank.any():
            d = d[~blank]
            _log(log, "sku_blank", blank.sum(), "Dropped", "A row without an SKU cannot be attributed to a product.")
        _log(log, "sku_normalise", len(d), "Trimmed, collapsed spaces, upper-cased SKU ids", "One rule across all datasets so joins match.")

    for c in spec["dates"]:
        if c in d.columns:
            d[c] = parse_dates(d[c], dayfirst)
            if c in spec["required"]:
                bad = d[c].isna()
                if bad.any():
                    d = d[~bad]
                    _log(log, f"date_{c}", bad.sum(), "Dropped", "Rows with an unparseable required date cannot be placed on the timeline.")
            d[c] = d[c].dt.normalize() if hasattr(d[c], "dt") else d[c]

    for c in spec["numeric"]:
        if c in d.columns:
            d[c] = to_numeric(d[c]).astype("float64")
            if c in spec["required"]:
                bad = d[c].isna()
                if bad.any():
                    d = d[~bad]
                    _log(log, f"missing_{c}", bad.sum(), "Dropped", "Required numeric value missing or invalid; not imputed and not set to 0.")
            neg = d[c] < 0
            if neg.any():
                if c in spec["required"]:
                    d = d[~neg]
                    _log(log, f"negative_{c}", neg.sum(), "Dropped", "Negative quantity is invalid (returns are not modelled as demand).")
                else:
                    d.loc[neg, c] = np.nan
                    _log(log, f"negative_{c}", neg.sum(), "Set to blank", "Negative price/cost is invalid; blanked rather than guessed.")

    if "promo_flag" in d.columns:
        d["promo_flag"] = d["promo_flag"].astype("string").str.strip().str.lower().isin(TRUE_SET)
        _log(log, "promo_flag", len(d), "Converted to boolean", "yes/true/1 => promotion; anything else => no promotion.")
    for c in ("holiday", "promo_event"):
        if c in d.columns and dataset == "calendar":
            v = d[c].astype("string").str.strip().str.lower().fillna("")
            d[f"{c}_flag"] = ~v.isin({"", "0", "false", "no", "none", "nan", "n"})
            _log(log, c, len(d), f"Added {c}_flag", "Non-empty and not 0/no/false => event day.")

    ex = d.duplicated().sum()
    if ex:
        d = d.drop_duplicates()
        _log(log, "exact_duplicates", ex, "Dropped", "Identical repeated rows are treated as double-loaded data.")

    keys = [k for k in spec["key"] if k in d.columns]
    if keys and d.duplicated(keys).any():
        k = d.duplicated(keys, keep=False).sum()
        if dataset == "sales":
            agg = {c: "sum" for c in ("units_sold", "revenue") if c in d.columns}
            for c in d.columns:
                if c in ("unit_price",):
                    agg[c] = "mean"
                elif c == "promo_flag":
                    agg[c] = "max"
                elif c not in keys and c not in agg:
                    agg[c] = "first"
            # min_count=1 keeps revenue blank when every duplicate is blank (not 0)
            d = d.groupby(keys, as_index=False, dropna=False).agg(
                {c: (lambda s: s.sum(min_count=1)) if f == "sum" else f for c, f in agg.items()})
            _log(log, "key_duplicates", k, "Aggregated per SKU-day (sum of units/revenue, mean price)", "Multiple lines for one SKU-day are summed to a daily total.")
        else:
            keep = "first" if dataset == "calendar" else "last"
            d = d.sort_values(keys[0] if dataset != "inventory" else "date").drop_duplicates(keys, keep=keep)
            _log(log, "key_duplicates", k, f"Kept {keep} row per key", "Keys must be unique for safe joins.")
    d = d.reset_index(drop=True)
    _log(log, "final", len(d), "Rows after cleaning", f"{n0 - len(d):,} row(s) removed or merged in total; all counted above.")
    return d, log
