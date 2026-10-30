"""Rolling-origin (expanding window) backtesting. Training data for each fold ends strictly before the test weeks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.forecasting.baseline import seasonal_naive_array


def plan_folds(first_week: pd.Timestamp, last_week: pd.Timestamp, h: int, n_folds: int, min_train: int) -> list[pd.Timestamp]:
    """First-test-week cut-offs. Folds are non-overlapping blocks of h weeks ending at the last observed week.
    Reduces n_folds until at least min_train weeks precede the first cut-off."""
    total = int((last_week - first_week).days // 7) + 1
    k = n_folds
    while k > 0 and total - h * k < min_train:
        k -= 1
    return [last_week - pd.Timedelta(weeks=h * (k - i) - 1) for i in range(k)]


def rolling_origin_backtest(history: dict[str, pd.Series], models: dict, h: int, cutoffs: list[pd.Timestamp],
                            min_train: int, season: int = 52) -> pd.DataFrame:
    rows = []
    for fold, cut in enumerate(cutoffs):
        test_weeks = pd.date_range(cut, periods=h, freq="7D")
        train, actual = {}, {}
        for sku, s in history.items():
            tr = s[s.index < cut]
            te = s.reindex(test_weeks)
            if len(tr) >= min_train and te.notna().all():
                train[sku], actual[sku] = tr, te.values.astype(float)
        if not train:
            continue
        for name, fn in models.items():
            preds = fn(train, h, season)
            for sku, f in preds.items():
                kind = seasonal_naive_array(train[sku].values, 1, season)[1] if name == "seasonal_naive" else None
                for step in range(h):
                    rows.append((fold, cut, sku, test_weeks[step], step + 1, actual[sku][step], name, float(f[step]), kind))
    df = pd.DataFrame(rows, columns=["fold", "origin", "sku_id", "week_start", "step", "actual", "model", "forecast", "seasonal_used"])
    if df.empty:
        return df
    # compare models on identical observations only
    n_models = df["model"].nunique()
    keep = df.groupby(["fold", "sku_id", "week_start"])["model"].transform("nunique") == n_models
    return df[keep].reset_index(drop=True)
