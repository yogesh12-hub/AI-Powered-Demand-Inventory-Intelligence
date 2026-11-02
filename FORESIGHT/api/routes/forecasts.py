import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas import ForecastPoint, ForecastRequest, ForecastResponse, SkuForecast
from src.config import Settings
from src.forecasting.forecast_pipeline import run_forecast

router = APIRouter(prefix="/forecasts", tags=["forecasts"])


def _f(x):
    return None if x is None or not np.isfinite(x) else float(x)


@router.post("/score", response_model=ForecastResponse, summary="Weekly SKU forecasts with backtest evaluation")
def score(req: ForecastRequest):
    """Runs the same pipeline as the dashboard (seasonal-naive baseline, candidate models, rolling-origin backtest).
    Histories must be gap-free weekly series that end in the same week; gaps are rejected, never filled."""
    rows = []
    for s in req.skus:
        wk = pd.to_datetime([p.week_start for p in s.history])
        wk = wk - pd.to_timedelta(wk.dayofweek, unit="D")
        if len(set(wk)) != len(wk):
            raise HTTPException(422, f"SKU {s.sku_id}: more than one point in the same week.")
        order = np.argsort(wk.values)
        wk = wk[order]
        if any((wk[1:] - wk[:-1]).days != 7):
            raise HTTPException(422, f"SKU {s.sku_id}: history has missing weeks; supply a contiguous weekly series.")
        u = np.array([p.units for p in s.history])[order]
        rows += [(s.sku_id.strip().upper(), w, v, np.nan, np.nan, True) for w, v in zip(wk, u)]
    weekly = pd.DataFrame(rows, columns=["sku_id", "week_start", "units", "revenue", "promo_share", "observed"])
    st = Settings(horizon_weeks=req.horizon_weeks, backtest_folds=req.backtest_folds, min_history_weeks=req.min_history_weeks)
    fr = run_forecast(weekly, st)
    if fr.forecasts.empty:
        raise HTTPException(422, {"message": "No SKU has enough history to forecast.", "excluded": fr.excluded.to_dict("records"), "warnings": fr.warnings})
    out = [SkuForecast(sku_id=k, points=[ForecastPoint(week_start=r.week_start.date(), forecast=r.forecast, lower=_f(r.lower), upper=_f(r.upper), baseline=r.baseline)
                                         for r in g.itertuples()]) for k, g in fr.forecasts.groupby("sku_id")]
    return ForecastResponse(model=fr.model, evaluated=fr.evaluated, wape=_f(fr.model_wape) if fr.evaluated else None,
                            baseline_wape=_f(fr.baseline_wape) if fr.evaluated else None, folds=fr.folds, forecasts=out,
                            excluded=fr.excluded.to_dict("records"), warnings=fr.warnings)
