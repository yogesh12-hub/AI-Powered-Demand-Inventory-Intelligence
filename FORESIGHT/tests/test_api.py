import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from api.main import app

c = TestClient(app)


def _hist(n=90, seed=0, start="2022-01-03"):
    rng = np.random.default_rng(seed)
    wk = pd.date_range(start, periods=n, freq="7D")
    return [{"week_start": d.strftime("%Y-%m-%d"), "units": float(u)} for d, u in zip(wk, rng.poisson(30, n))]


def test_health():
    assert c.get("/health").json()["status"] == "ok"


def test_forecast_endpoint_runs_shared_pipeline():
    r = c.post("/forecasts/score", json={"skus": [{"sku_id": "a", "history": _hist(seed=1)}, {"sku_id": "b", "history": _hist(seed=2)}], "horizon_weeks": 6})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["evaluated"] and j["wape"] is not None and len(j["forecasts"]) == 2 and len(j["forecasts"][0]["points"]) == 6


def test_forecast_rejects_gaps_duplicates_and_short_history():
    h = _hist()
    gap = h[:30] + h[32:]
    assert c.post("/forecasts/score", json={"skus": [{"sku_id": "a", "history": gap}]}).status_code == 422
    assert c.post("/forecasts/score", json={"skus": [{"sku_id": "a", "history": h + [h[-1]]}]}).status_code == 422
    assert c.post("/forecasts/score", json={"skus": [{"sku_id": "a", "history": h[:10]}]}).status_code == 422


def test_inventory_endpoint_matches_manual_calculation():
    item = {"sku_id": "a", "on_hand_units": 10, "on_order_units": 0, "lead_time_days": 14, "weekly_forecast": [20] * 8, "weekly_sigma": 5, "unit_cost": 50, "unit_price": 80}
    r = c.post("/inventory/score", json={"items": [item]})
    s = r.json()["scores"][0]
    assert r.status_code == 200 and s["stockout_risk"] == "High" and s["reorder_qty"] == 62 and s["inventory_position"] == 10


def test_inventory_validation_errors():
    bad = {"sku_id": "a", "on_hand_units": -1, "on_order_units": 0, "lead_time_days": 14, "weekly_forecast": [1]}
    assert c.post("/inventory/score", json={"items": [bad]}).status_code == 422
    ok = {**bad, "on_hand_units": 1}
    assert c.post("/inventory/score", json={"items": [ok, ok]}).status_code == 422
    assert c.post("/inventory/score", json={"items": []}).status_code == 422
