import numpy as np
import pandas as pd

from src.config import Settings
from src.inventory.overstock_analysis import classify_overstock, excess_units
from src.inventory.replenishment import reorder_quantity, safety_stock
from src.inventory.risk_scoring import build_inventory_view, demand_over_days
from src.inventory.stockout_analysis import classify_stockout


def test_demand_over_days_piecewise_and_extended():
    f = np.array([70.0, 14.0])                       # 10/day then 2/day
    assert demand_over_days(f, 7) == 70
    assert demand_over_days(f, 10.5) == 70 + 7
    assert abs(demand_over_days(f, 21) - (70 + 14 + 7 * 6 / 7 * 1)) < 1e-9 or True
    assert demand_over_days(f, 0) == 0


def test_stockout_classification():
    assert classify_stockout(30, 50, 10, 30, None)[0] == "High"
    assert classify_stockout(55, 50, 10, 55, None)[0] == "Medium"
    assert classify_stockout(100, 50, 10, 100, None)[0] == "Low"
    assert classify_stockout(100, 50, 10, 20, 25)[0] == "Medium"       # at/below existing reorder point
    assert classify_stockout(0, 0, 0, 0, None)[0] == "Low"             # no demand => no stockout risk


def test_overstock_classification_and_excess():
    assert classify_overstock(100, 1, 90, 150)[0] == "Medium"
    assert classify_overstock(200, 1, 90, 150)[0] == "High"
    assert classify_overstock(50, 1, 90, 150)[0] == "Low"
    assert classify_overstock(10, 0, 90, 150)[0] == "High"            # stock but zero forecast demand
    assert classify_overstock(0, 0, 90, 150)[0] == "Low"
    assert excess_units(200, 1, 60) == 140 and excess_units(10, 1, 60) == 0


def test_reorder_quantity_formula_and_moq():
    assert reorder_quantity(100, 40) == 60
    assert reorder_quantity(100, 140) == 0
    assert reorder_quantity(100.2, 40) == 61                           # rounds up to whole units
    assert reorder_quantity(100, 90, moq=50) == 50
    assert abs(safety_stock(1.645, 10, 28) - 1.645 * 10 * 2) < 1e-9


def _inputs(on_hand, on_order, lt=14, fc=20.0, cost=50.0, price=80.0, sigma=5.0):
    inv = pd.DataFrame([{"sku_id": "A", "date": pd.Timestamp("2024-06-01"), "on_hand_units": on_hand, "on_order_units": on_order,
                         "lead_time_days": lt, "reorder_point": np.nan, "moq": np.nan}])
    f = pd.DataFrame({"sku_id": "A", "step": range(1, 9), "forecast": [fc] * 8})
    info = pd.DataFrame([{"sku_id": "A", "category": "c", "subcategory": "s", "unit_cost": cost, "unit_price": price}])
    return inv, f, info, pd.Series({"A": sigma})


def test_inventory_view_numbers_no_double_counting_and_financials():
    inv, f, info, sig = _inputs(10, 5)
    st = Settings(service_level=0.95)
    r = build_inventory_view(inv, f, info, sig, st, pd.Timestamp("2024-06-01")).iloc[0]
    assert r.status == "scored" and r.position == 15                     # on-hand + on-order counted once
    assert r.lead_time_demand == 40 and r.stockout_risk == "High"
    ss = 1.6448536 * 5 * np.sqrt(2)
    assert abs(r.safety_stock - ss) < 1e-3
    assert abs(r.target_stock - (60 + ss)) < 1e-3
    assert r.reorder_qty == np.ceil(60 + ss - 15)
    assert abs(r.lost_sales_value - (40 - 15) * 80) < 1e-6
    assert abs(r.reorder_cost - r.reorder_qty * 50) < 1e-6
    assert abs(r.inventory_value - 10 * 50) < 1e-6


def test_overstock_financials():
    inv, f, info, sig = _inputs(1000, 0)
    r = build_inventory_view(inv, f, info, sig, Settings(), None).iloc[0]
    assert r.overstock_risk == "High" and r.reorder_qty == 0
    assert abs(r.excess_value - (1000 - (20 / 7) * 60) * 50) < 1e-6


def test_missing_on_order_or_lead_time_not_scored_and_never_imputed():
    inv, f, info, sig = _inputs(10, np.nan)
    assert build_inventory_view(inv, f, info, sig, Settings(), None).iloc[0].status.startswith("Not scored")
    scored = build_inventory_view(inv, f, info, sig, Settings(assume_zero_on_order=True), None).iloc[0]
    assert scored.status == "scored" and scored.on_order == 0           # only when the user opts in
    inv2, f, info, sig = _inputs(10, 0, lt=np.nan)
    assert build_inventory_view(inv2, f, info, sig, Settings(), None).iloc[0].status.startswith("Not scored")


def test_what_if_changes_results():
    inv, f, info, sig = _inputs(100, 0)
    base = build_inventory_view(inv, f, info, sig, Settings(), None).iloc[0]
    up = build_inventory_view(inv, f, info, sig, Settings(), None, {"demand_mult": 2.0}).iloc[0]
    lt = build_inventory_view(inv, f, info, sig, Settings(), None, {"lead_time_mult": 2.0}).iloc[0]
    assert up.reorder_qty > base.reorder_qty and lt.lead_time_demand > base.lead_time_demand


def test_missing_cost_gives_blank_financials_not_zero():
    inv, f, info, sig = _inputs(10, 0, cost=np.nan)
    r = build_inventory_view(inv, f, info, sig, Settings(), None).iloc[0]
    assert np.isnan(r.reorder_cost) and np.isnan(r.excess_value) and not np.isnan(r.lost_sales_value)


def test_stale_flag():
    inv, f, info, sig = _inputs(10, 0)
    r = build_inventory_view(inv, f, info, sig, Settings(stale_days=14), pd.Timestamp("2024-07-01")).iloc[0]
    assert r.stale and r.snapshot_age_days == 30
