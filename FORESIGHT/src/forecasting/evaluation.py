"""Accuracy metrics. WAPE is primary; zero denominators are handled explicitly (returns NaN)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def wape(actual, forecast) -> float:
    """sum|a-f| / sum|a| x 100. Returns NaN when sum|a| == 0 (undefined; never reported as 0 or inf)."""
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    denom = np.abs(a).sum()
    if a.size == 0 or denom == 0:
        return float("nan")
    return float(np.abs(a - f).sum() / denom * 100)


def mae(actual, forecast) -> float:
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    return float(np.abs(a - f).mean()) if a.size else float("nan")


def rmse(actual, forecast) -> float:
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    return float(np.sqrt(((a - f) ** 2).mean())) if a.size else float("nan")


def bias_pct(actual, forecast) -> float:
    """(sum f - sum a) / sum a x 100; positive = over-forecast. NaN if no actual demand."""
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    return float((f.sum() - a.sum()) / a.sum() * 100) if a.sum() != 0 else float("nan")


def mape(actual, forecast) -> tuple[float, int]:
    """MAPE over observations with actual > 0 only; returns (value, number of zero-demand points excluded)."""
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    m = a > 0
    excl = int((~m).sum())
    if not m.any():
        return float("nan"), excl
    return float(np.mean(np.abs(a[m] - f[m]) / a[m]) * 100), excl


def summarise(df: pd.DataFrame, model_col: str = "model") -> pd.DataFrame:
    rows = []
    for m, g in df.groupby(model_col):
        rows.append({"model": m, "wape": wape(g.actual, g.forecast), "mae": mae(g.actual, g.forecast),
                     "rmse": rmse(g.actual, g.forecast), "bias_pct": bias_pct(g.actual, g.forecast),
                     "n_obs": len(g), "n_skus": g.sku_id.nunique(), "folds": g.fold.nunique()})
    return pd.DataFrame(rows).sort_values("wape", na_position="last").reset_index(drop=True)
