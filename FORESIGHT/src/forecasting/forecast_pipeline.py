"""End-to-end forecasting: eligibility → backtest → model selection → final forecast + intervals."""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np
import pandas as pd

from src.config import Settings
from src.forecasting import evaluation as ev
from src.forecasting.backtesting import plan_folds, rolling_origin_backtest
from src.forecasting.baseline import seasonal_naive, seasonal_share
from src.forecasting.models import MODELS

BASELINE = "seasonal_naive"


@dataclass
class ForecastResult:
    horizon: int
    model: str | None = None
    forecasts: pd.DataFrame = field(default_factory=pd.DataFrame)   # sku_id, week_start, step, forecast, lower, upper, baseline
    backtest: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    sku_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    sigma: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))  # weekly forecast-error sd per SKU
    excluded: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: list[str] = field(default_factory=list)
    folds: int = 0
    eval_period: tuple | None = None
    baseline_kind: str = ""
    evaluated: bool = False
    intervals: bool = False

    @property
    def model_wape(self):
        return self._m(self.model)

    @property
    def baseline_wape(self):
        return self._m(BASELINE)

    def _m(self, name):
        if self.metrics.empty or name is None:
            return float("nan")
        r = self.metrics[self.metrics.model == name]
        return float(r.wape.iloc[0]) if len(r) else float("nan")

    @property
    def improvement_pct(self):
        b, m = self.baseline_wape, self.model_wape
        return float((b - m) / b * 100) if np.isfinite(b) and b > 0 and np.isfinite(m) else float("nan")


def build_history(weekly: pd.DataFrame, s: Settings) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    """Eligible SKUs as weekly Series; everything else listed in `excluded` with its reason."""
    hist, excl = {}, []
    if weekly.empty:
        return hist, pd.DataFrame(columns=["sku_id", "reason"])
    last = weekly["week_start"].max()
    for sku, g in weekly.groupby("sku_id"):
        g = g.sort_values("week_start")
        ser = pd.Series(g["units"].values.astype(float), index=pd.DatetimeIndex(g["week_start"]))
        if ser.isna().any():
            excl.append((sku, "Gaps in weekly history and zero-fill is disabled"))
        elif ser.index[-1] != last:
            excl.append((sku, "No data through the latest week (zero-fill disabled)"))
        elif len(ser) < s.min_history_weeks:
            excl.append((sku, f"Only {len(ser)} weeks of history (minimum {s.min_history_weeks})"))
        else:
            hist[sku] = ser
    if len(hist) > s.max_skus:
        rank = sorted(hist, key=lambda k: hist[k].tail(52).sum(), reverse=True)
        excl += [(k, f"Beyond the top {s.max_skus} SKUs by recent volume (see Settings)") for k in rank[s.max_skus:]]
        hist = {k: hist[k] for k in rank[:s.max_skus]}
    return hist, pd.DataFrame(excl, columns=["sku_id", "reason"])


def run_forecast(weekly: pd.DataFrame, settings: Settings, models: dict | None = None) -> ForecastResult:
    h = int(settings.horizon_weeks)
    models = models or MODELS
    res = ForecastResult(horizon=h)
    hist, res.excluded = build_history(weekly, settings)
    if not hist:
        res.warnings.append("No SKU has enough clean weekly history to forecast. "
                            f"Need at least {settings.min_history_weeks} complete weeks per SKU (see Settings).")
        return res
    first = min(s.index[0] for s in hist.values())
    last = max(s.index[-1] for s in hist.values())
    cuts = plan_folds(first, last, h, settings.backtest_folds, settings.min_history_weeks)
    res.folds = len(cuts)
    bt = rolling_origin_backtest(hist, models, h, cuts, settings.min_history_weeks, settings.season_length) if cuts else pd.DataFrame()

    if bt.empty:
        res.warnings.append("History is too short for rolling-origin backtesting, so no accuracy is reported. "
                            "The seasonal-naive baseline is used as a documented fallback; treat it as unvalidated.")
        res.model = BASELINE
    else:
        res.evaluated = True
        res.backtest = bt
        res.metrics = ev.summarise(bt)
        valid = res.metrics.dropna(subset=["wape"])
        res.model = valid.iloc[0]["model"] if len(valid) else BASELINE
        res.eval_period = (bt["week_start"].min(), bt["week_start"].max())
        if res.folds < settings.backtest_folds:
            res.warnings.append(f"Only {res.folds} backtest fold(s) possible (requested {settings.backtest_folds}) with the available history.")
        if res.model != BASELINE and res.improvement_pct <= 0:
            res.warnings.append("The selected model does not beat the seasonal-naive baseline.")
        names = set(bt["model"].unique())
        for m in models:
            if m not in names:
                res.warnings.append(f"Model '{m}' was skipped: not enough data to train/evaluate it on the same observations.")

    # final forecasts with the chosen model on the full history
    chosen = models[res.model](hist, h, settings.season_length) or models[BASELINE](hist, h, settings.season_length)
    base = seasonal_naive(hist, h, settings.season_length)
    res.baseline_kind = ("seasonal-naive (52-week lag)" if settings.season_length > 1 else "naive") \
        + ("" if seasonal_share(hist, settings.season_length) == 1 else
           f"; naive fallback for {int(round((1 - seasonal_share(hist, settings.season_length)) * len(hist)))} SKU(s) with <{settings.season_length} weeks")
    rows = []
    for sku, f in chosen.items():
        fut = pd.date_range(hist[sku].index[-1] + pd.Timedelta(weeks=1), periods=h, freq="7D")
        for i in range(h):
            rows.append((sku, fut[i], i + 1, float(f[i]), float(base[sku][i])))
    fc = pd.DataFrame(rows, columns=["sku_id", "week_start", "step", "forecast", "baseline"])

    # residual-based intervals + sigma (only when the chosen model was actually backtested)
    fc["lower"], fc["upper"] = np.nan, np.nan
    sig = {}
    if res.evaluated:
        b = res.backtest[res.backtest.model == res.model].copy()
        b["err"] = b["actual"] - b["forecast"]
        sd = b.groupby("sku_id")["err"].agg(["std", "count"])
        sd = sd[sd["count"] >= 8]["std"]
        z = NormalDist().inv_cdf(0.5 + settings.interval_level / 2)
        fc["_sd"] = fc["sku_id"].map(sd)
        fc["lower"] = (fc["forecast"] - z * fc["_sd"]).clip(lower=0)
        fc["upper"] = fc["forecast"] + z * fc["_sd"]
        fc = fc.drop(columns="_sd")
        res.intervals = fc["lower"].notna().any()
        sig = sd.to_dict()
        res.sku_metrics = (b.merge(res.backtest[res.backtest.model == BASELINE][["fold", "sku_id", "week_start", "forecast"]]
                                   .rename(columns={"forecast": "baseline_fc"}), on=["fold", "sku_id", "week_start"])
                           .groupby("sku_id").apply(lambda g: pd.Series({
                               "wape_model": ev.wape(g.actual, g.forecast), "wape_baseline": ev.wape(g.actual, g.baseline_fc),
                               "n_obs": len(g)}), include_groups=False).reset_index())
    # fallback sigma: sd of the last 26 weeks of history (documented)
    for k, s in hist.items():
        sig.setdefault(k, float(s.tail(26).std(ddof=1)) if len(s) > 2 else 0.0)
    res.sigma = pd.Series(sig, name="sigma_weekly")
    res.forecasts = fc
    return res
