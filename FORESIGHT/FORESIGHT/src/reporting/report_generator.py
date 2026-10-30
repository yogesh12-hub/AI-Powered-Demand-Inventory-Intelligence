"""Builds every exportable table from the active workspace + current filters (single source for pages and exports)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics import eda
from src.analytics import sales_analysis as sa
from src.config import CURRENCY, Settings
from src.forecasting import evaluation as ev
from src.inventory.risk_scoring import exposure_by_category, scored
from src.reporting.export_pdf import build_pdf
from src.utils.helpers import Filters, apply_filters, inr, pct


def filter_describe(f: Filters | None) -> str:
    if f is None:
        return "No filters"
    parts = []
    if f.date_from is not None and f.date_to is not None:
        parts.append(f"{f.date_from.date()} to {f.date_to.date()}")
    for lbl, v in (("Category", f.categories), ("Subcategory", f.subcategories), ("SKU", f.skus)):
        if v:
            parts.append(f"{lbl}: {', '.join(map(str, v[:5]))}{'…' if len(v) > 5 else ''}")
    return "; ".join(parts) or "No filters"


def filtered(ws, f: Filters | None, settings: Settings, horizon: int | None = None, what_if: dict | None = None) -> dict:
    """All filtered frames. Date range filters history; forecasts/inventory follow SKU/category filters (they are forward-looking)."""
    out = {"sales": None, "weekly": None, "forecasts": None, "backtest": None, "metrics": None, "view": None}
    if ws.sales is None:
        return out
    out["sales"] = apply_filters(ws.sales, f, "date")
    out["weekly"] = apply_filters(ws.weekly, f, "week_start")
    active = bool(f and (f.categories or f.subcategories or f.skus))     # restrict by SKU only when the user actually filtered
    nodate = Filters(categories=f.categories, subcategories=f.subcategories, skus=f.skus) if active else None
    allowed = set(apply_filters(ws.weekly, nodate, None)["sku_id"]) if active else None
    fr = ws.forecast
    if fr is not None and not fr.forecasts.empty:
        h = horizon or fr.horizon
        fc = fr.forecasts[fr.forecasts.step <= h]
        out["forecasts"] = fc if allowed is None else fc[fc.sku_id.isin(allowed)]
        if fr.evaluated:
            bt = fr.backtest[fr.backtest.step <= h]
            bt = bt if allowed is None else bt[bt.sku_id.isin(allowed)]
            out["backtest"], out["metrics"] = bt, (ev.summarise(bt) if len(bt) else None)
    v = ws.inventory_view(settings, what_if)
    if v is not None:
        out["view"] = v if allowed is None else v[v.sku_id.isin(allowed)]
    return out


def build_tables(ws, settings: Settings, f: Filters | None, horizon: int | None = None) -> tuple[dict, list, list]:
    """Returns (tables, insights, summary_lines) for exports."""
    d = filtered(ws, f, settings, horizon)
    t: dict[str, pd.DataFrame] = {}
    dq = []
    for name, rep in ws.reports.items():
        x = rep.to_frame()
        x.insert(0, "dataset", name)
        dq.append(x)
    for name, log in ws.logs.items():
        x = pd.DataFrame(log)
        x.insert(0, "dataset", name)
        x["level"] = "cleaning_log"
        dq.append(x.rename(columns={"action": "message"}))
    if dq:
        t["data_quality_report"] = pd.concat(dq, ignore_index=True)
    s = d["sales"]
    insights: list = []
    if s is not None and len(s):
        top, low = sa.top_low(s, 20)
        t["eda_top_skus"] = top
        t["eda_low_skus"] = low
        cp = sa.category_performance(s)
        if cp is not None:
            t["eda_category_performance"] = cp
        insights, _ = eda.generate_insights(s, d["weekly"], ws.clean.get("calendar"), ws.inventory_latest(), settings.dead_stock_days)
        if insights:
            t["business_insights"] = pd.DataFrame(insights)
    if d["forecasts"] is not None:
        t["forecast_outputs"] = d["forecasts"]
    if d["metrics"] is not None:
        t["model_evaluation"] = d["metrics"]
        t["model_evaluation_backtest_rows"] = d["backtest"]
    v = d["view"]
    if v is not None and len(v):
        t["inventory_risk_report"] = v
        r = scored(v)
        t["replenishment_recommendations"] = r[r.reorder_qty > 0].sort_values(["priority", "financial_exposure"], ascending=[True, False])[
            ["sku_id", "category", "on_hand", "on_order", "forecast_weekly_avg", "coverage_days", "lead_time_days", "stockout_risk",
             "overstock_risk", "target_stock", "position", "reorder_qty", "reorder_cost", "priority", "recommendation"]]
    lines = []
    if s is not None and len(s):
        k = sa.kpis(s)
        lines.append(f"Period: {k['start'].date()} to {k['end'].date()}; {k['skus']} SKUs; {k['units']:,.0f} units"
                     + (f"; revenue {inr(k['revenue'])}" if k["revenue"] is not None else "; revenue not available in the data") + ".")
    fr = ws.forecast
    if fr is not None and d["metrics"] is not None and len(d["metrics"]):
        m = d["metrics"].set_index("model")
        if fr.model in m.index and "seasonal_naive" in m.index:
            lines.append(f"Selected model {fr.model}: WAPE {pct(m.loc[fr.model, 'wape'])} vs seasonal-naive {pct(m.loc['seasonal_naive', 'wape'])} "
                         f"over {int(m.loc[fr.model, 'n_obs']):,} held-out SKU-weeks ({fr.folds} rolling-origin fold(s)).")
    if v is not None and len(v):
        r = scored(v)
        lines.append(f"{(r.stockout_risk == 'High').sum()} SKU(s) with high stockout risk, {(r.overstock_risk == 'High').sum()} with high overstock risk "
                     f"({len(v) - len(r)} not scored). Estimated exposure {inr(r.financial_exposure.sum())}; recommended reorder spend {inr(r.reorder_cost.sum())}. "
                     "All figures are estimates under the documented assumptions, not realised savings.")
    t["executive_summary"] = pd.DataFrame({"summary": lines + [f"{i['title']} - {i['relevance']}" for i in insights]})
    return t, insights, lines


def make_pdf(ws, settings: Settings, f: Filters | None, horizon: int | None = None) -> bytes:
    t, insights, lines = build_tables(ws, settings, f, horizon)
    secs = [("Scope", [f"Filters: {filter_describe(f)}", f"Datasets: {', '.join(ws.loaded)}"]),
            ("Executive summary", [("bullets", lines)] if lines else ["Nothing to summarise yet."])]
    if insights:
        secs.append(("Business insights", [("bullets", [f"{i['title']}. {i['relevance']}" for i in insights])]))
    if "model_evaluation" in t:
        secs.append(("Model evaluation (rolling-origin backtest, WAPE %)", [t["model_evaluation"].round(2)]))
    if "inventory_risk_report" in t:
        r = scored(t["inventory_risk_report"])
        hi = r[(r.stockout_risk == "High") | (r.overstock_risk == "High")].sort_values("financial_exposure", ascending=False)
        secs.append(("Highest-risk SKUs", [hi[["sku_id", "category", "stockout_risk", "overstock_risk", "coverage_days", "financial_exposure"]].round(1)]))
    if "replenishment_recommendations" in t:
        secs.append(("Replenishment recommendations (advisory)", [t["replenishment_recommendations"].head(20)[
            ["sku_id", "priority", "position", "target_stock", "reorder_qty", "reorder_cost"]].round(1)]))
    secs.append(("Assumptions and limitations", [("bullets", [
        "Forecasts are statistical projections from the uploaded history; accuracy is only what the backtest measured.",
        "On-order stock is counted once in the inventory position and assumed to arrive within the lead time.",
        f"Financial values are estimates in {CURRENCY}; excess value uses unit cost, lost sales use recent selling price.",
        "Recommendations are advisory; no purchase orders are placed."])]))
    return build_pdf("FORESIGHT - Demand & Inventory Intelligence Report", f"Generated from the active uploaded datasets. {filter_describe(f)}", secs)
