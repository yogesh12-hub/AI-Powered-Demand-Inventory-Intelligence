"""Seasonal-naive baseline: forecast = value observed one season (52 weeks) earlier."""
from __future__ import annotations

import numpy as np
import pandas as pd


def seasonal_naive_array(train: np.ndarray, h: int, season: int = 52) -> tuple[np.ndarray, bool]:
    """Returns (forecast, used_seasonal). Fallback when history < one season: naive (repeat last value)."""
    n = len(train)
    if season > 1 and n >= season:
        return np.array([train[n - season + (i % season)] for i in range(h)], float), True
    return np.full(h, float(train[-1]) if n else 0.0), False


def seasonal_naive(history: dict[str, pd.Series], h: int, season: int = 52) -> dict[str, np.ndarray]:
    return {k: seasonal_naive_array(s.values, h, season)[0] for k, s in history.items()}


def seasonal_share(history: dict[str, pd.Series], season: int = 52) -> float:
    return float(np.mean([len(s) >= season for s in history.values()])) if history else 0.0
