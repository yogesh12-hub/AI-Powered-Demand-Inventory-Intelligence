import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas import InventoryRequest, InventoryResponse, InventoryScore
from src.config import Settings
from src.inventory.risk_scoring import build_inventory_view

router = APIRouter(prefix="/inventory", tags=["inventory"])


def _f(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) or pd.isna(x) else float(x)


@router.post("/score", response_model=InventoryResponse, summary="Stockout/overstock risk and reorder quantity per SKU")
def score(req: InventoryRequest):
    """Same calculation as the dashboard (src.inventory.risk_scoring.build_inventory_view)."""
    if req.overstock_high_days <= req.overstock_medium_days:
        raise HTTPException(422, "overstock_high_days must be greater than overstock_medium_days")
    ids = [i.sku_id.strip().upper() for i in req.items]
    if len(set(ids)) != len(ids):
        raise HTTPException(422, "Duplicate sku_id values in request.")
    inv = pd.DataFrame([{"sku_id": k, "date": pd.NaT, "on_hand_units": i.on_hand_units, "on_order_units": i.on_order_units,
                         "lead_time_days": i.lead_time_days, "reorder_point": i.reorder_point if i.reorder_point is not None else np.nan,
                         "moq": i.moq if i.moq is not None else np.nan} for k, i in zip(ids, req.items)])
    fc = pd.DataFrame([(k, n + 1, v) for k, i in zip(ids, req.items) for n, v in enumerate(i.weekly_forecast)], columns=["sku_id", "step", "forecast"])
    info = pd.DataFrame([{"sku_id": k, "category": np.nan, "subcategory": np.nan,
                          "unit_cost": i.unit_cost if i.unit_cost is not None else np.nan,
                          "unit_price": i.unit_price if i.unit_price is not None else np.nan} for k, i in zip(ids, req.items)])
    sigma = pd.Series({k: i.weekly_sigma for k, i in zip(ids, req.items)})
    st = Settings(service_level=req.service_level, review_period_days=req.review_period_days, overstock_medium_days=req.overstock_medium_days,
                  overstock_high_days=req.overstock_high_days, target_coverage_days=req.target_coverage_days)
    v = build_inventory_view(inv, fc, info, sigma, st, None, {"demand_mult": req.demand_multiplier, "lead_time_mult": req.lead_time_multiplier})
    out = []
    for r in v.to_dict("records"):
        ok = r["status"] == "scored"
        out.append(InventoryScore(sku_id=r["sku_id"], status=r["status"], stockout_risk=r.get("stockout_risk") if ok else None,
                                  overstock_risk=r.get("overstock_risk") if ok else None, risk=r.get("risk") if ok else None,
                                  inventory_position=_f(r.get("position")), lead_time_demand=_f(r.get("lead_time_demand")),
                                  safety_stock=_f(r.get("safety_stock")), target_stock=_f(r.get("target_stock")),
                                  coverage_days=_f(r.get("coverage_days")), reorder_qty=_f(r.get("reorder_qty")),
                                  reorder_cost=_f(r.get("reorder_cost")), financial_exposure=_f(r.get("financial_exposure")),
                                  recommendation=r.get("recommendation") if ok else None))
    return InventoryResponse(scores=out)
