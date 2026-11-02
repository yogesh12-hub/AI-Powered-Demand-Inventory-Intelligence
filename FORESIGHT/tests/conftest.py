"""Test fixtures. Synthetic data exists ONLY here (for unit tests); the application never imports this module."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Settings  # noqa: E402
from src.data.upload_manager import Workspace  # noqa: E402


def make_raw(n_skus=10, weeks=130, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.date_range("2022-01-03", periods=weeks * 7, freq="D")  # starts on a Monday
    rows = []
    for i in range(n_skus):
        base = 5 + 3 * i
        season = 1 + 0.4 * np.sin(2 * np.pi * (days.dayofyear / 365.25) + i)
        promo = rng.random(len(days)) < 0.1
        units = rng.poisson(base * season * np.where(promo, 1.5, 1.0))
        price = 100 + 10 * i
        for d, u, p in zip(days, units, promo):
            rows.append((d.strftime("%Y-%m-%d"), f"sku{i:02d}", str(u), str(u * price), str(price), "1" if p else "0"))
    sales = pd.DataFrame(rows, columns=["date", "sku_id", "units_sold", "revenue", "unit_price", "promo_flag"])
    master = pd.DataFrame({"sku_id": [f"SKU{i:02d}" for i in range(n_skus)],
                           "category": ["A" if i % 2 == 0 else "B" for i in range(n_skus)],
                           "subcategory": ["x"] * n_skus, "launch_date": ["2021-01-01"] * n_skus,
                           "unit_cost": [str(60 + 6 * i) for i in range(n_skus)], "list_price": [str(100 + 10 * i) for i in range(n_skus)]})
    last = days[-1].strftime("%Y-%m-%d")
    inv = pd.DataFrame({"date": [last] * n_skus, "sku_id": [f"SKU{i:02d}" for i in range(n_skus)],
                        "on_hand_units": [str(v) for v in rng.integers(0, 1500, n_skus)],
                        "on_order_units": [str(v) for v in rng.integers(0, 200, n_skus)],
                        "lead_time_days": [str(v) for v in rng.integers(5, 30, n_skus)],
                        "reorder_point": ["50"] * n_skus})
    return {"sales": sales, "sku_master": master, "inventory": inv}


@pytest.fixture(scope="session")
def raw():
    return make_raw()


@pytest.fixture()
def settings():
    return Settings(horizon_weeks=8, backtest_folds=3, min_history_weeks=26)


@pytest.fixture()
def ws(raw, settings):
    w = Workspace()
    for k, df in raw.items():
        w.set_dataset(k, df.copy(), f"{k}.csv", 1000)
    w.process(settings)
    return w
