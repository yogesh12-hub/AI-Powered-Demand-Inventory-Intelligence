"""Central configuration: dataset schemas, field documentation and user-tunable settings.

No credentials live here. Anything secret must come from environment variables (see .env.example).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from statistics import NormalDist

APP_NAME = "FORESIGHT"
CURRENCY = "₹"
WELCOME = "Welcome to FORESIGHT — Upload your datasets to begin."

# ---------------------------------------------------------------------------------------------
# Dataset schemas. "required" = analysis cannot start without it. "optional" columns unlock
# extra features (documented in FEATURES below). Field docs are shown in the UI and docs/.
# ---------------------------------------------------------------------------------------------
DATASETS: dict[str, dict] = {
    "sales": {
        "label": "Sales (sales_daily.csv)",
        "filename": "sales_daily.csv",
        "required": ["date", "sku_id", "units_sold"],
        "optional": ["revenue", "unit_price", "promo_flag"],
        "key": ["date", "sku_id"],
        "numeric": ["units_sold", "revenue", "unit_price"],
        "dates": ["date"],
        "fields": {
            "date": ("date", "Calendar date of the sales record."),
            "sku_id": ("text", "Product identifier; must match the other datasets."),
            "units_sold": ("number >= 0", "Units sold on that date."),
            "revenue": ("number >= 0", "Revenue in INR for the row (optional)."),
            "unit_price": ("number >= 0", "Selling price per unit in INR (optional)."),
            "promo_flag": ("0/1, yes/no", "Whether a promotion applied (optional)."),
        },
    },
    "sku_master": {
        "label": "Products (sku_master.csv)",
        "filename": "sku_master.csv",
        "required": ["sku_id"],
        "optional": ["category", "subcategory", "launch_date", "unit_cost", "list_price"],
        "key": ["sku_id"],
        "numeric": ["unit_cost", "list_price"],
        "dates": ["launch_date"],
        "fields": {
            "sku_id": ("text", "Unique product identifier."),
            "category": ("text", "Product category (enables category filters)."),
            "subcategory": ("text", "Product subcategory."),
            "launch_date": ("date", "Date the product launched."),
            "unit_cost": ("number >= 0", "Purchase cost per unit in INR (enables cost-based exposure)."),
            "list_price": ("number >= 0", "List price per unit in INR."),
        },
    },
    "calendar": {
        "label": "Calendar (calendar.csv)",
        "filename": "calendar.csv",
        "required": ["date"],
        "optional": ["week", "month", "season", "holiday", "promo_event"],
        "key": ["date"],
        "numeric": [],
        "dates": ["date"],
        "fields": {
            "date": ("date", "Calendar date."),
            "week": ("number", "Week number (informational)."),
            "month": ("number", "Month number (informational)."),
            "season": ("text", "Season label."),
            "holiday": ("text or 0/1", "Holiday name or flag; blank/0 means no holiday."),
            "promo_event": ("text or 0/1", "Promotion event name or flag; blank/0 means none."),
        },
    },
    "inventory": {
        "label": "Inventory (inventory_snapshots.csv)",
        "filename": "inventory_snapshots.csv",
        "required": ["date", "sku_id", "on_hand_units"],
        "optional": ["on_order_units", "lead_time_days", "reorder_point", "moq"],
        "key": ["date", "sku_id"],
        "numeric": ["on_hand_units", "on_order_units", "lead_time_days", "reorder_point", "moq"],
        "dates": ["date"],
        "fields": {
            "date": ("date", "Snapshot date."),
            "sku_id": ("text", "Product identifier."),
            "on_hand_units": ("number >= 0", "Units physically in stock."),
            "on_order_units": ("number >= 0", "Units ordered but not yet received (needed for risk scoring)."),
            "lead_time_days": ("number > 0", "Supplier lead time in days (needed for risk scoring)."),
            "reorder_point": ("number >= 0", "Existing reorder point (optional cross-check)."),
            "moq": ("number >= 0", "Minimum order quantity (optional, applied to recommendations)."),
        },
    },
}

PROCESSING_FIELDS = ("missing_weeks_as_zero", "assume_zero_on_order")
FORECAST_FIELDS = ("horizon_weeks", "backtest_folds", "min_history_weeks", "max_skus", "season_length",
                   "interval_level", "missing_weeks_as_zero")


@dataclass
class Settings:
    """User-tunable assumptions. Every field has a real effect (see docs/methodology.md)."""

    # forecasting
    horizon_weeks: int = 8
    backtest_folds: int = 4
    min_history_weeks: int = 26
    max_skus: int = 500
    season_length: int = 52
    interval_level: float = 0.80
    # data processing
    missing_weeks_as_zero: bool = True
    assume_zero_on_order: bool = False
    dead_stock_days: int = 90
    # inventory policy
    service_level: float = 0.95
    review_period_days: int = 7
    stockout_high_ratio: float = 1.0
    overstock_medium_days: int = 90
    overstock_high_days: int = 150
    target_coverage_days: int = 60
    stale_days: int = 14
    # display
    table_rows: int = 200
    show_intervals: bool = True

    @property
    def z(self) -> float:
        return NormalDist().inv_cdf(min(max(self.service_level, 0.5), 0.9999))

    def to_dict(self) -> dict:
        return asdict(self)

    def key(self, names: tuple[str, ...] | None = None) -> str:
        d = self.to_dict()
        names = names or tuple(d)
        return "|".join(f"{n}={d[n]}" for n in names)

    def processing_key(self) -> str:
        return self.key(PROCESSING_FIELDS + ("dead_stock_days",))

    def forecast_key(self) -> str:
        return self.key(FORECAST_FIELDS)

    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})
