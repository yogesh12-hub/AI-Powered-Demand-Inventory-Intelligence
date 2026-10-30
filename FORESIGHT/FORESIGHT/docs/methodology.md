# FORESIGHT methodology

All outputs are computed from user-uploaded CSVs. Nothing is simulated or pre-loaded.

## 1. Data pipeline (D1)
1. **Read** every column as text so validation sees exactly what was uploaded.
2. **Map** columns (auto-suggested by name/alias, editable). Day-first dates are an explicit option.
3. **Validate** (`src/data/schema_validation.py`): required columns, numeric/date parsing, blanks, negatives, exact and key duplicates, SKU case/whitespace variants, outliers (reported, never removed), revenue vs units×price mismatch. Datasets with errors are excluded until fixed.
4. **Clean** (`cleaning.py`), each step logged with row counts and justification: SKU trim/upper-case; unparseable required dates and missing/negative required quantities dropped (never zero-filled); negative prices blanked; exact duplicates dropped; duplicate SKU-days summed (sales) or last/first kept (others).
5. **Weekly aggregation** (`preprocessing.py`): Monday-start weeks; only *complete* weeks kept. Weeks with no sales rows are zero only if *Treat missing weeks as zero* is on (default, visible in Settings and logged); otherwise they stay blank and those SKUs are excluded from forecasting. Joins use `validate="m:1"` and assert row counts.

## 2. Forecasting (D3)
* Unit: weekly units per SKU, horizon 6–8 weeks. Minimum history: 26 weeks (configurable); SKUs below are listed with reasons.
* **Baseline**: seasonal-naive (value 52 weeks earlier). When fewer than 52 weeks exist for a SKU it falls back to naive (last value); the share is disclosed.
* **Candidates**: moving average (4w), simple exponential smoothing (alpha by in-sample SSE), global gradient-boosting lag model (lags 1–8,13,26,52, rolling means, week-of-year; scale-normalised; recursive). Trained only on the training window.
* **Rolling-origin backtest**: expanding window, non-overlapping h-week test blocks ending at the last observed week (default 4 folds, reduced automatically if history is short). Training for a fold contains only weeks strictly before its test weeks; a unit test proves changing future data leaves earlier-fold forecasts unchanged.
* **Metric**: WAPE = Σ|a−f| / Σ|a| × 100, undefined (NaN) when Σ|a| = 0. Also MAE, RMSE, bias. Models are compared on identical SKU-weeks. The model with the lowest pooled WAPE is selected — the baseline if nothing beats it.
* **Intervals**: normal approximation from the selected model's per-SKU backtest residuals (needs ≥ 8 residuals); otherwise none are shown.
* If history is too short to backtest, no accuracy is reported and the baseline is used with a warning.

## 3. Inventory (D4)
* Latest snapshot per SKU is "current". `on_order_units` and `lead_time_days` are required and never assumed (opt-in setting to treat blank on-order as 0).
* `position = on_hand + on_order` (on-order counted once; assumed to arrive within the lead time).
* `lead-time demand` = sum of weekly forecasts applied piece-wise over the lead time (horizon mean extended beyond the forecast).
* `safety stock = z(service level) × σ_weekly × √(lead_time/7)`; σ = backtest residual sd per SKU, else sd of last 26 weeks.
* **Stockout risk**: High if position < ratio × lead-time demand (ratio default 1.0); Medium if position < lead-time demand + safety stock, or on-hand ≤ existing reorder point; else Low.
* **Overstock risk**: coverage = position ÷ forecast daily demand; Medium > 90 d, High > 150 d (configurable); zero forecast demand with stock = High.
* **Target stock** = demand over (lead time + review period) + safety stock. **Reorder qty** = max(0, target − position), rounded up, raised to MOQ if provided. Advisory only.
* **Scores** (0–100) are for sorting; classification uses the rules above.
* **Financials (₹, estimates, not realised savings)**: lost sales = max(0, lead-time demand − position) × recent selling price; excess value = units above the target-coverage policy (default 60 days) × unit cost; reorder cost = qty × unit cost. Blank cost/price gives blank values, never zero.
* Stale warning when a snapshot is older than the threshold relative to the latest sales date.

## 4. Known limitations
* Statistical models see no future promotions or price changes; demand shocks are not anticipated.
* The sales→weekly zero-fill assumption suits retail data that omits zero-sale days; turn it off if your file records zeros explicitly.
* Inventory risk uses a single latest snapshot and assumes it matches the forecast origin.
* Lost-sales value is an upper-bound style estimate (no substitution, no partial arrivals).
