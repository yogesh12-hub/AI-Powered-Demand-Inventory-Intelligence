"""Transparent replenishment calculation.

inventory_position = on_hand + on_order            (on-order counted once; assumed to arrive within the lead time)
safety_stock       = z(service_level) x sigma_weekly x sqrt(lead_time_days / 7)
target_stock       = forecast demand over (lead_time + review_period) days + safety_stock
reorder_qty        = max(0, target_stock - inventory_position), rounded up to whole units and to MOQ if provided
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def safety_stock(z: float, sigma_weekly: float, lead_time_days: float) -> float:
    return float(z * max(sigma_weekly, 0.0) * np.sqrt(max(lead_time_days, 1.0) / 7.0))


def reorder_quantity(target: float, position: float, moq: float | None = None) -> float:
    q = max(0.0, target - position)
    if q <= 0:
        return 0.0
    q = float(np.ceil(q))
    if moq is not None and np.isfinite(moq) and moq > 0:
        q = max(q, float(moq))
    return q


def priority(stockout: str, qty: float) -> str:
    if qty <= 0:
        return "None"
    return {"High": "Urgent", "Medium": "Soon"}.get(stockout, "Routine")


def explain(r: dict, z_pct: float, review_days: float) -> str:
    if r["reorder_qty"] > 0:
        return (f"Order {r['reorder_qty']:,.0f} units: inventory position {r['position']:,.0f} (on-hand {r['on_hand']:,.0f} + on-order {r['on_order']:,.0f}) "
                f"is below target {r['target_stock']:,.0f} = demand over {r['lead_time_days']:g}-day lead time + {review_days:g}-day review "
                f"({r['target_stock'] - r['safety_stock']:,.0f}) + safety stock {r['safety_stock']:,.0f} at {z_pct:.0f}% service. "
                f"Stockout risk: {r['stockout_risk']}.")
    if r["overstock_risk"] in ("High", "Medium"):
        return f"No order: {r['overstock_reason']} Consider pausing purchases or promoting."
    return f"No order needed: position {r['position']:,.0f} already meets target {r['target_stock']:,.0f}."
