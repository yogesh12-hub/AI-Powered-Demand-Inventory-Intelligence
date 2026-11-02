"""Schema validation. Reports problems; never alters data."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import DATASETS
from src.utils.helpers import normalize_sku

TRUE_SET = {"1", "true", "t", "yes", "y"}
FALSE_SET = {"0", "false", "f", "no", "n", ""}


@dataclass
class Issue:
    level: str  # error | warning | info
    code: str
    message: str
    count: int = 0
    fix: str = ""


@dataclass
class ValidationReport:
    dataset: str
    rows: int = 0
    issues: list[Issue] = field(default_factory=list)
    missing: pd.DataFrame | None = None

    @property
    def errors(self):
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self):
        return [i for i in self.issues if i.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(self, level, code, message, count=0, fix=""):
        self.issues.append(Issue(level, code, message, int(count), fix))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([vars(i) for i in self.issues], columns=["level", "code", "message", "count", "fix"])


def parse_dates(s: pd.Series, dayfirst: bool = False) -> pd.Series:
    return pd.to_datetime(s.astype("string").str.strip(), errors="coerce", dayfirst=dayfirst, format="mixed")


def to_numeric(s: pd.Series) -> pd.Series:
    cleaned = s.astype("string").str.replace(r"[₹$,\s]|INR|Rs\.?", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def validate(df: pd.DataFrame, dataset: str, dayfirst: bool = False) -> ValidationReport:
    """Validate an already column-mapped frame (strings or typed values)."""
    spec = DATASETS[dataset]
    rep = ValidationReport(dataset=dataset, rows=len(df))
    if df is None or len(df) == 0:
        rep.add("error", "empty", "The dataset has no rows.", fix="Upload a file that contains data rows.")
        return rep
    missing_cols = [c for c in spec["required"] if c not in df.columns]
    if missing_cols:
        rep.add("error", "missing_columns", f"Missing required columns: {', '.join(missing_cols)}.",
                len(missing_cols), "Map these fields to columns in your file.")
        return rep
    for c in spec["optional"]:
        if c not in df.columns:
            rep.add("info", "optional_missing", f"Optional column '{c}' is not mapped; related features stay unavailable.")

    n = len(df)
    miss = df.isna().sum().rename("missing").to_frame()
    miss["pct"] = (miss["missing"] / n * 100).round(2)
    rep.missing = miss.reset_index(names="column")

    # SKU identifiers
    if "sku_id" in df.columns:
        raw = df["sku_id"].astype("string")
        blank = raw.isna() | (raw.str.strip() == "")
        if blank.sum():
            rep.add("warning", "sku_blank", "Rows with a blank sku_id will be excluded.", blank.sum(), "Fill in the SKU identifiers.")
        norm = normalize_sku(raw[~blank])
        if raw[~blank].nunique() > norm.nunique():
            rep.add("warning", "sku_variants", "SKU identifiers differ only by case/whitespace; they will be merged.",
                    raw[~blank].nunique() - norm.nunique())

    # dates
    for c in spec["dates"]:
        if c in df.columns:
            parsed = parse_dates(df[c], dayfirst)
            bad = parsed.isna() & df[c].notna()
            blank = df[c].isna()
            if bad.sum() + blank.sum() == n:
                rep.add("error", "date_parse", f"No value in '{c}' could be parsed as a date.", n,
                        "Check the date column mapping or the 'day first' option.")
            elif bad.sum() + blank.sum():
                level = "warning" if c in ("date",) or c in spec["required"] else "info"
                rep.add(level, "date_parse", f"{int(bad.sum() + blank.sum())} value(s) in '{c}' are blank or not valid dates; those rows are excluded.",
                        bad.sum() + blank.sum(), "Correct the dates or confirm the day-first setting.")

    # numerics
    for c in spec["numeric"]:
        if c not in df.columns:
            continue
        num = to_numeric(df[c])
        notnum = num.isna() & df[c].notna()
        if notnum.sum():
            lvl = "error" if notnum.sum() == df[c].notna().sum() else "warning"
            rep.add(lvl, "non_numeric", f"'{c}' has {int(notnum.sum())} non-numeric value(s).", notnum.sum(), "Remove text from numeric columns.")
        neg = (num < 0).sum()
        if neg:
            rep.add("warning", "negative", f"'{c}' has {int(neg)} negative value(s); they will be excluded/blanked (see cleaning log).", neg)
        if c == "lead_time_days" and (num == 0).sum():
            rep.add("warning", "zero_lead_time", "Some lead times are 0 days.", (num == 0).sum())
        if c in spec["required"] and df[c].isna().sum():
            rep.add("warning", "required_missing", f"Required column '{c}' has {int(df[c].isna().sum())} blank value(s); rows are excluded, never set to 0.", df[c].isna().sum())
        elif c not in spec["required"] and df[c].isna().sum():
            rep.add("info", "optional_blank", f"'{c}' has {int(df[c].isna().sum())} blank value(s); left blank (not zero).", df[c].isna().sum())

    if "promo_flag" in df.columns:
        v = df["promo_flag"].astype("string").str.strip().str.lower().fillna("")
        bad = ~v.isin(TRUE_SET | FALSE_SET)
        if bad.sum():
            rep.add("warning", "promo_values", "promo_flag has values other than 0/1/yes/no/true/false.", bad.sum())

    # duplicates
    exact = df.duplicated().sum()
    if exact:
        rep.add("warning", "exact_duplicates", f"{int(exact)} fully duplicated row(s) will be dropped.", exact)
    keys = [k for k in spec["key"] if k in df.columns]
    if keys and len(keys) == len(spec["key"]):
        tmp = df[keys].copy()
        if "sku_id" in tmp:
            tmp["sku_id"] = normalize_sku(tmp["sku_id"])
        if "date" in tmp:
            tmp["date"] = parse_dates(tmp["date"], dayfirst)
        kd = tmp.duplicated().sum() - exact
        if kd > 0:
            how = {"sales": "summed per SKU-day", "inventory": "latest row kept", "sku_master": "last row kept", "calendar": "first row kept"}[dataset]
            rep.add("warning", "key_duplicates", f"{int(kd)} row(s) repeat an existing key {tuple(keys)}; {how}.", kd)

    # suspicious values
    if dataset == "sales" and "units_sold" in df.columns:
        u = to_numeric(df["units_sold"]).dropna()
        u = u[u >= 0]
        if len(u) > 20:
            q1, q3 = u.quantile([0.25, 0.75])
            hi = q3 + 5 * (q3 - q1)
            out = (u > hi).sum() if q3 > q1 else 0
            if out:
                rep.add("info", "outliers", f"{int(out)} unusually large units_sold value(s) (> Q3 + 5×IQR). They are kept; no outlier removal is applied.", out)
    if dataset == "sales" and {"revenue", "units_sold", "unit_price"} <= set(df.columns):
        r, uq, p = (to_numeric(df[c]) for c in ("revenue", "units_sold", "unit_price"))
        m = (uq > 0) & (p > 0) & r.notna()
        if m.sum():
            off = ((r[m] - uq[m] * p[m]).abs() > 0.05 * (uq[m] * p[m])).sum()
            if off:
                rep.add("info", "revenue_mismatch", f"{int(off)} row(s) where revenue differs from units × price by more than 5%.", off)
    return rep
