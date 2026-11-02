"""Column mapping: suggest a mapping from uploaded columns to the expected schema and apply it."""
from __future__ import annotations

import re

import pandas as pd

from src.config import DATASETS

ALIASES = {
    "date": ["date", "orderdate", "saledate", "day", "invoicedate", "dt", "timestamp", "snapshotdate", "datetime"],
    "sku_id": ["skuid", "sku", "itemid", "productid", "item", "product", "stockcode", "article", "itemcode", "productcode"],
    "units_sold": ["unitssold", "units", "quantity", "qty", "quantitysold", "sold", "unitsales", "salesunits", "sales"],
    "revenue": ["revenue", "salesamount", "totalsales", "amount", "netsales", "totalprice", "sales_value"],
    "unit_price": ["unitprice", "price", "sellingprice", "priceperunit", "saleprice"],
    "promo_flag": ["promoflag", "promo", "onpromotion", "promotion", "ispromo", "discountflag"],
    "category": ["category", "productcategory", "dept", "department", "cat"],
    "subcategory": ["subcategory", "subcat", "productsubcategory", "class"],
    "launch_date": ["launchdate", "introdate", "launched", "startdate"],
    "unit_cost": ["unitcost", "cost", "costprice", "purchasecost", "cogs"],
    "list_price": ["listprice", "mrp", "retailprice", "msrp", "price"],
    "week": ["week", "weeknumber", "weekno"],
    "month": ["month", "monthnumber"],
    "season": ["season"],
    "holiday": ["holiday", "isholiday", "holidayname"],
    "promo_event": ["promoevent", "promotionevent", "event", "campaign"],
    "on_hand_units": ["onhandunits", "onhand", "stock", "stockonhand", "inventory", "qtyonhand", "closingstock"],
    "on_order_units": ["onorderunits", "onorder", "incoming", "inbound", "openorders", "qtyonorder"],
    "lead_time_days": ["leadtimedays", "leadtime", "supplierleadtime", "ltdays"],
    "reorder_point": ["reorderpoint", "rop", "reorderlevel", "reorderlvl"],
    "moq": ["moq", "minorderqty", "minimumorderquantity", "minorder"],
}


def _norm(c: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(c).lower())


def auto_map(columns: list[str], dataset: str) -> dict[str, str | None]:
    """Return {target_column: source_column or None}. Exact name matches win over aliases."""
    spec = DATASETS[dataset]
    targets = spec["required"] + spec["optional"]
    norm_src = {c: _norm(c) for c in columns}
    used: set[str] = set()
    mapping: dict[str, str | None] = {t: None for t in targets}
    for t in targets:  # pass 1: exact
        for c, n in norm_src.items():
            if c not in used and n == _norm(t):
                mapping[t] = c
                used.add(c)
                break
    for t in targets:  # pass 2: aliases
        if mapping[t] is not None:
            continue
        for alias in ALIASES.get(t, []):
            hit = next((c for c, n in norm_src.items() if c not in used and n == _norm(alias)), None)
            if hit:
                mapping[t] = hit
                used.add(hit)
                break
    return mapping


def apply_mapping(df: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    """Rename mapped source columns to the target schema. Unmapped targets are simply absent."""
    pick = {src: tgt for tgt, src in mapping.items() if src and src in df.columns}
    return df[list(pick)].rename(columns=pick).copy()


def mapping_problems(mapping: dict[str, str | None], dataset: str) -> list[str]:
    spec = DATASETS[dataset]
    probs = [f"Required column '{c}' is not mapped." for c in spec["required"] if not mapping.get(c)]
    srcs = [s for s in mapping.values() if s]
    dup = {s for s in srcs if srcs.count(s) > 1}
    probs += [f"Source column '{s}' is mapped to more than one field." for s in sorted(dup)]
    return probs
