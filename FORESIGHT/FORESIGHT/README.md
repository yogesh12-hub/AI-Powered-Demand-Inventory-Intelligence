# FORESIGHT — AI-Powered Demand & Inventory Intelligence

Forecast SKU-level weekly demand, flag stockout/overstock risk, and get explainable reorder recommendations with ₹ impact estimates. **Everything is computed from datasets you upload. There is no demo mode, no sample data and no synthetic records.** Before upload the app shows: *Welcome to FORESIGHT — Upload your datasets to begin.*

## Business problem
Planners juggle thousands of SKUs. Too little stock loses sales; too much ties up cash. FORESIGHT forecasts the next 6–8 weeks, compares against a seasonal-naive baseline using rolling-origin backtesting (WAPE), and turns forecasts plus stock levels into risk scores and reorder quantities.

## Features
Upload (4 CSVs, preview, column mapping, validation, cleaning log, replacement/invalidations) · EDA & insights · weekly forecasting with backtest · inventory risk · replenishment with what-if · alerts · CSV/ZIP/PDF exports · FastAPI scoring service · settings that drive real calculations.

## Datasets (CSV; header-only templates in `templates/`)
| File | Required | Optional |
|---|---|---|
| `sales_daily.csv` | date, sku_id, units_sold | revenue, unit_price, promo_flag |
| `sku_master.csv` | sku_id | category, subcategory, launch_date, unit_cost, list_price |
| `calendar.csv` | date | week, month, season, holiday, promo_event |
| `inventory_snapshots.csv` | date, sku_id, on_hand_units | on_order_units*, lead_time_days*, reorder_point, moq |

\* needed for risk scoring and never assumed. Column names may differ — map them on the Data Quality page. Field definitions are shown beside each upload. Sales alone enables sales analytics and forecasting; inventory risk needs the inventory file.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py                    # dashboard
uvicorn api.main:app --port 8000        # scoring API, docs at /docs
pytest                                  # tests
```
Workflow: **Data Quality** → one upload box (CSV or Excel; one combined file, or several files that are merged automatically) → the app shows *What we found* and a *Missing data check* → review mapping → open other pages (forecast runs on demand).

**Missing data never causes an error.** Whatever can be analysed is analysed; each missing dataset or column is listed with the exact features it limits (e.g. no inventory file → inventory risk and replenishment are off, sales analytics and forecasting still work). A header-only combined template is in `templates/foresight_combined_template.csv`.

## API
`GET /health` · `POST /forecasts/score` (weekly histories → forecasts + backtest WAPE) · `POST /inventory/score` (stock + forecast → risk, reorder qty). Request validation returns 422; it reuses `src/` logic and is stateless and unauthenticated.

## Method summary
See `docs/methodology.md` for formulas: WAPE, seasonal-naive baseline, rolling-origin folds, safety stock, target stock, reorder quantity, risk rules, financial estimates, assumptions.

## Testing
`pytest` currently runs 87 tests (validation, cleaning, forecasting incl. WAPE zero-denominator and leakage, inventory maths, filtered exports, dataset replacement, API, every page with data and in the empty state). Test fixtures are generated inside `tests/` only and are never loaded by the app. These tests used generated fixtures, so they verify correctness of the code, **not** forecast accuracy on real data.

## Privacy
Uploads stay in session memory; nothing is saved or sent externally. Never commit datasets (`.gitignore`).

## Limitations / next steps
Not yet run on a real business dataset; no screenshots included (add real ones after uploading your data); deployment configuration (Dockerfile, Streamlit config) is untested on a hosting platform; no promotion/price regressors; API has no auth. See `docs/`.
