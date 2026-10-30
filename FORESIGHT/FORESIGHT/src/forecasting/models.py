"""Candidate forecasting models. Each takes {sku: weekly Series (train only)} and horizon h and returns {sku: array(h)}.

Models never see data after the training cut-off, so leakage is impossible by construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.forecasting.baseline import seasonal_naive


def moving_average(history, h, season=52, window: int = 4):
    return {k: np.full(h, float(np.mean(s.values[-window:]))) for k, s in history.items()}


def ses(history, h, season=52):
    """Simple exponential smoothing; alpha chosen per SKU by in-sample squared error on a grid."""
    alphas = np.linspace(0.05, 0.95, 10)
    out = {}
    for k, s in history.items():
        y = s.values.astype(float)
        lvl = np.full(alphas.shape, y[0])
        sse = np.zeros_like(alphas)
        for t in range(1, len(y)):
            e = y[t] - lvl
            sse += e ** 2
            lvl = lvl + alphas * e
        out[k] = np.full(h, max(float(lvl[int(np.argmin(sse))]), 0.0))
    return out


LAGS = [1, 2, 3, 4, 5, 6, 7, 8, 13, 26, 52]


def _design(vals: np.ndarray, woy: np.ndarray, t: np.ndarray):
    c = np.concatenate([[0.0], np.cumsum(vals)])

    def rmean(w):
        lo = np.maximum(t - w, 0)
        return (c[t] - c[lo]) / np.maximum(t - lo, 1)

    scale = np.maximum(rmean(13), 0.5)
    cols = []
    for lag in LAGS:
        idx = t - lag
        cols.append(np.where(idx >= 0, vals[np.clip(idx, 0, len(vals) - 1)], np.nan) / scale)
    cols += [rmean(4) / scale, rmean(8) / scale,
             np.sin(2 * np.pi * woy[t] / 52.18), np.cos(2 * np.pi * woy[t] / 52.18)]
    return np.column_stack(cols), scale


def _woy(idx: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray(idx.isocalendar().week.values, float)


def gbm_lag(history, h, season=52):
    """One global gradient-boosting model across SKUs on scale-normalised lag features, forecast recursively.
    Features at time t only use observations < t. Returns {} if there is too little data to train."""
    X, y = [], []
    for s in history.values():
        v = s.values.astype(float)
        if len(v) < 12:
            continue
        t = np.arange(8, len(v))
        x, sc = _design(v, _woy(s.index), t)
        X.append(x)
        y.append(v[t] / sc)
    if not X or sum(len(a) for a in y) < 300:
        return {}
    model = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.08, max_leaf_nodes=15,
                                          min_samples_leaf=20, random_state=0)
    model.fit(np.vstack(X), np.concatenate(y))
    out = {}
    for k, s in history.items():
        v = list(s.values.astype(float))
        idx = s.index
        fut = pd.date_range(idx[-1] + pd.Timedelta(weeks=1), periods=h, freq="7D")
        woy = _woy(idx.append(fut))
        preds = []
        for i in range(h):
            arr = np.array(v)
            x, sc = _design(arr, woy, np.array([len(arr)]))
            p = max(float(model.predict(x)[0] * sc[0]), 0.0)
            preds.append(p)
            v.append(p)
        out[k] = np.array(preds)
    return out


MODELS = {
    "seasonal_naive": seasonal_naive,
    "moving_average_4w": moving_average,
    "exp_smoothing": ses,
    "gbm_lag": gbm_lag,
}
