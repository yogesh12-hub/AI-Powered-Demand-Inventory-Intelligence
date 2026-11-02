"""Workspace: holds uploads, mappings, processed data and cached results.

Any change to an upload, mapping or day-first option invalidates ALL derived results, so stale numbers can never be
shown for a replaced dataset. Pure Python (no Streamlit import) so it is unit-testable and reusable by the API.
"""
from __future__ import annotations

import io
from datetime import datetime

import pandas as pd

from src.analytics import data_quality as dq
from src.config import DATASETS, Settings
from src.data import preprocessing as pp
from src.data.cleaning import clean_dataset
from src.data.column_mapping import apply_mapping, auto_map, mapping_problems
from src.data.schema_validation import ValidationReport, validate
from src.forecasting.forecast_pipeline import ForecastResult, run_forecast
from src.inventory.risk_scoring import build_inventory_view


def read_csv_bytes(data: bytes) -> pd.DataFrame:
    """Read every column as text so validation can see exactly what the user uploaded."""
    last = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(data), dtype=str, encoding=enc, sep=None, engine="python")
        except UnicodeDecodeError as e:
            last = e
    raise ValueError(f"Could not decode the file: {last}")


class Workspace:
    def __init__(self):
        self.raw: dict[str, pd.DataFrame] = {}
        self.meta: dict[str, dict] = {}
        self.mapping: dict[str, dict] = {}
        self.dayfirst: dict[str, bool] = {}
        self.version = 0
        self.reset_results()

    # ---- invalidation -------------------------------------------------------------------------
    def reset_results(self):
        self.reports: dict[str, ValidationReport] = {}
        self.clean: dict[str, pd.DataFrame] = {}
        self.logs: dict[str, list] = {}
        self.sales: pd.DataFrame | None = None
        self.weekly: pd.DataFrame | None = None
        self.weekly_log: list = []
        self.cross: list = []
        self.processed = False
        self.process_key: str | None = None
        self.forecast: ForecastResult | None = None
        self.forecast_key: str | None = None
        self.last_inventory_view: pd.DataFrame | None = None
        self._inv_cache: dict = {}
        self.version += 1

    # ---- uploads ------------------------------------------------------------------------------
    def set_dataset(self, name: str, df: pd.DataFrame, filename: str, size: int, file_id: str | None = None):
        self.raw[name] = df
        self.meta[name] = {"filename": filename, "size": size, "rows": len(df), "columns": list(df.columns),
                           "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "file_id": file_id}
        self.mapping[name] = auto_map(list(df.columns), name)
        self.dayfirst[name] = False
        self.reset_results()

    def remove_dataset(self, name: str):
        for d in (self.raw, self.meta, self.mapping, self.dayfirst):
            d.pop(name, None)
        self.reset_results()

    def set_mapping(self, name: str, mapping: dict, dayfirst: bool | None = None):
        changed = mapping != self.mapping.get(name) or (dayfirst is not None and dayfirst != self.dayfirst.get(name))
        if changed:
            self.mapping[name] = dict(mapping)
            if dayfirst is not None:
                self.dayfirst[name] = dayfirst
            self.reset_results()

    def mapped(self, name: str) -> pd.DataFrame:
        return apply_mapping(self.raw[name], self.mapping[name])

    @property
    def loaded(self) -> list[str]:
        return list(self.raw)

    # ---- processing ---------------------------------------------------------------------------
    def process(self, settings: Settings):
        """Validate every uploaded dataset; clean those without blocking errors; build weekly demand."""
        keep = {k: v for k, v in self.raw.items()}
        self.reset_results()
        for name in keep:
            probs = mapping_problems(self.mapping[name], name)
            m = self.mapped(name)
            rep = validate(m, name, self.dayfirst.get(name, False)) if not probs else ValidationReport(name, len(m))
            for p in probs:
                rep.add("error", "mapping", p, fix="Adjust the column mapping.")
            self.reports[name] = rep
            if rep.ok:
                c, log = clean_dataset(m, name, self.dayfirst.get(name, False))
                if len(c) == 0:
                    rep.add("error", "empty_after_cleaning", "No rows remain after cleaning; see the cleaning log.")
                else:
                    self.clean[name], self.logs[name] = c, log
        s = self.clean.get("sales")
        if s is not None:
            master = self.clean.get("sku_master")
            self.sales = pp.enrich_sales(s, master)
            w, self.weekly_log = pp.build_weekly(s, settings)
            self.weekly = pp.attach_sku_attrs(w, master)
        self.cross = dq.cross_dataset_checks(self.clean)
        self.processed = True
        self.process_key = settings.processing_key()

    def stale_reason(self, settings: Settings) -> str | None:
        if self.processed and self.process_key != settings.processing_key():
            return "Processing settings changed since the last run. Click 'Validate & process' on the Data Quality page."
        return None

    def features(self, settings: Settings):
        return dq.feature_availability(self.clean, self.weekly, settings.min_history_weeks)

    # ---- derived results (cached; keyed so any change recomputes) -------------------------------
    def run_forecast(self, settings: Settings) -> ForecastResult:
        key = f"{self.version}|{settings.forecast_key()}"
        if self.forecast is None or self.forecast_key != key:
            self.forecast = run_forecast(self.weekly, settings)
            self.forecast_key = key
            self._inv_cache = {}
        return self.forecast

    def inventory_latest(self) -> pd.DataFrame | None:
        inv = self.clean.get("inventory")
        return None if inv is None else pp.latest_snapshot(inv)

    def inventory_view(self, settings: Settings, what_if: dict | None = None) -> pd.DataFrame | None:
        inv, fr = self.inventory_latest(), self.forecast
        if inv is None or fr is None or fr.forecasts.empty or self.sales is None:
            return None
        key = f"{self.forecast_key}|{settings.to_dict()}|{what_if}"
        if key not in self._inv_cache:
            info = pp.sku_info(self.sales, self.clean.get("sku_master"))
            v = build_inventory_view(inv, fr.forecasts, info, fr.sigma, settings, self.sales["date"].max(), what_if)
            self._inv_cache[key] = v
        v = self._inv_cache[key]
        if not what_if:
            self.last_inventory_view = v
        return v
