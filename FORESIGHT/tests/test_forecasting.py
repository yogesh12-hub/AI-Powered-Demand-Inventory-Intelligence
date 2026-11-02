import numpy as np
import pandas as pd

from src.config import Settings
from src.forecasting.backtesting import plan_folds, rolling_origin_backtest
from src.forecasting.baseline import seasonal_naive_array
from src.forecasting.evaluation import mape, summarise, wape
from src.forecasting.forecast_pipeline import run_forecast
from src.forecasting.models import MODELS


def test_wape_value_and_zero_denominator():
    assert wape([10, 20], [12, 18]) == (2 + 2) / 30 * 100
    assert np.isnan(wape([0, 0], [1, 2]))        # undefined, not 0 or inf
    assert np.isnan(wape([], []))
    v, excluded = mape([0, 10], [5, 8])
    assert excluded == 1 and v == 20.0


def test_seasonal_naive_uses_value_one_season_ago_and_falls_back():
    x = np.arange(60, dtype=float)
    f, seasonal = seasonal_naive_array(x, 3, 52)
    assert seasonal and list(f) == [8.0, 9.0, 10.0]
    f2, seasonal2 = seasonal_naive_array(x[:20], 3, 52)
    assert not seasonal2 and list(f2) == [19.0] * 3


def test_forecast_generation_horizon_and_baseline_comparison(ws, settings):
    fr = run_forecast(ws.weekly, settings)
    assert fr.evaluated and fr.folds >= 1
    assert fr.forecasts.groupby("sku_id").step.max().eq(8).all()
    assert (fr.forecasts.forecast >= 0).all()
    assert {"seasonal_naive", fr.model} <= set(fr.metrics.model)
    assert np.isfinite(fr.baseline_wape) and np.isfinite(fr.model_wape)
    assert fr.model_wape <= fr.baseline_wape + 1e-9          # selection picks the lowest WAPE
    # forecasts start the week after the last observed week
    assert fr.forecasts.week_start.min() == ws.weekly.week_start.max() + pd.Timedelta(weeks=1)


def test_models_compared_on_identical_observations(ws, settings):
    fr = run_forecast(ws.weekly, settings)
    counts = fr.backtest.groupby("model").size()
    assert counts.nunique() == 1


def test_no_leakage_future_values_do_not_change_fold_forecasts():
    idx = pd.date_range("2022-01-03", periods=100, freq="7D")
    rng = np.random.default_rng(1)
    base = {f"S{i}": pd.Series(rng.poisson(20 + i, 100).astype(float), index=idx) for i in range(8)}   # enough rows for every model to train
    cuts = plan_folds(idx[0], idx[-1], 8, 2, 26)
    first = cuts[0]
    r1 = rolling_origin_backtest(base, MODELS, 8, cuts, 26)
    altered = {k: s.copy() for k, s in base.items()}
    for s in altered.values():
        s.loc[s.index >= first] = 9999.0                      # change ONLY data at/after the first test week
    r2 = rolling_origin_backtest(altered, MODELS, 8, cuts, 26)
    assert set(r1.model) == set(MODELS)                       # all four models actually ran
    for m in MODELS:
        a = r1[(r1.fold == 0) & (r1.model == m)].sort_values(["sku_id", "step"]).forecast.values
        b = r2[(r2.fold == 0) & (r2.model == m)].sort_values(["sku_id", "step"]).forecast.values
        assert len(a) == 8 * 8 and np.allclose(a, b), m


def test_folds_never_overlap_and_train_precedes_test():
    idx = pd.date_range("2022-01-03", periods=100, freq="7D")
    cuts = plan_folds(idx[0], idx[-1], 8, 4, 26)
    assert len(cuts) == 4 and all((b - a).days == 56 for a, b in zip(cuts, cuts[1:]))
    s = {"A": pd.Series(np.arange(100.0), index=idx)}
    bt = rolling_origin_backtest(s, {"seasonal_naive": MODELS["seasonal_naive"]}, 8, cuts, 26)
    assert (bt.week_start >= bt.origin).all()


def test_insufficient_history_warns_and_reports_no_accuracy():
    idx = pd.date_range("2024-01-01", periods=30, freq="7D")
    weekly = pd.DataFrame({"sku_id": "A", "week_start": idx, "units": np.arange(30.0) + 5})
    fr = run_forecast(weekly, Settings(min_history_weeks=26, backtest_folds=4))
    assert not fr.evaluated and np.isnan(fr.model_wape) and fr.warnings
    short = run_forecast(weekly.head(10), Settings(min_history_weeks=26))
    assert short.forecasts.empty and "A" in set(short.excluded.sku_id)


def test_intervals_only_when_backtest_supports_them(ws, settings):
    fr = run_forecast(ws.weekly, settings)
    assert fr.intervals and (fr.forecasts.upper >= fr.forecasts.lower).all()
