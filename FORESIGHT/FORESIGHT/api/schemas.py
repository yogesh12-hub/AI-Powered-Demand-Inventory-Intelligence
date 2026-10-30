from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class HistoryPoint(BaseModel):
    week_start: date = Field(description="Any date inside the week; normalised to the Monday of that week.")
    units: float = Field(ge=0)


class SkuHistory(BaseModel):
    sku_id: str = Field(min_length=1)
    history: list[HistoryPoint] = Field(min_length=8)


class ForecastRequest(BaseModel):
    skus: list[SkuHistory] = Field(min_length=1, max_length=2000)
    horizon_weeks: int = Field(8, ge=6, le=8)
    backtest_folds: int = Field(3, ge=1, le=8)
    min_history_weeks: int = Field(26, ge=8, le=156)


class ForecastPoint(BaseModel):
    week_start: date
    forecast: float
    lower: Optional[float] = None
    upper: Optional[float] = None
    baseline: float


class SkuForecast(BaseModel):
    sku_id: str
    points: list[ForecastPoint]


class ForecastResponse(BaseModel):
    model: Optional[str]
    evaluated: bool
    wape: Optional[float] = Field(None, description="Backtest WAPE %, null if not evaluated")
    baseline_wape: Optional[float] = None
    folds: int
    forecasts: list[SkuForecast]
    excluded: list[dict]
    warnings: list[str]


class InventoryItem(BaseModel):
    sku_id: str = Field(min_length=1)
    on_hand_units: float = Field(ge=0)
    on_order_units: float = Field(ge=0)
    lead_time_days: float = Field(gt=0)
    weekly_forecast: list[float] = Field(min_length=1, max_length=12, description="Forecast units per week, week 1 first")
    weekly_sigma: float = Field(0, ge=0, description="Std-dev of weekly forecast error (units)")
    reorder_point: Optional[float] = Field(None, ge=0)
    moq: Optional[float] = Field(None, ge=0)
    unit_cost: Optional[float] = Field(None, ge=0)
    unit_price: Optional[float] = Field(None, ge=0)

    @field_validator("weekly_forecast")
    @classmethod
    def non_negative(cls, v):
        if any(x < 0 for x in v):
            raise ValueError("weekly_forecast values must be >= 0")
        return v


class InventoryRequest(BaseModel):
    items: list[InventoryItem] = Field(min_length=1, max_length=5000)
    service_level: float = Field(0.95, ge=0.8, le=0.999)
    review_period_days: int = Field(7, ge=1, le=90)
    overstock_medium_days: int = Field(90, ge=7)
    overstock_high_days: int = Field(150, ge=8)
    target_coverage_days: int = Field(60, ge=7)
    demand_multiplier: float = Field(1.0, gt=0, le=5)
    lead_time_multiplier: float = Field(1.0, gt=0, le=5)


class InventoryScore(BaseModel):
    sku_id: str
    status: str
    stockout_risk: Optional[str] = None
    overstock_risk: Optional[str] = None
    risk: Optional[str] = None
    inventory_position: Optional[float] = None
    lead_time_demand: Optional[float] = None
    safety_stock: Optional[float] = None
    target_stock: Optional[float] = None
    coverage_days: Optional[float] = None
    reorder_qty: Optional[float] = None
    reorder_cost: Optional[float] = None
    financial_exposure: Optional[float] = None
    recommendation: Optional[str] = None


class InventoryResponse(BaseModel):
    scores: list[InventoryScore]
