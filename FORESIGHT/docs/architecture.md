# Architecture
```
Streamlit pages ─┐                         ┌─ src/data        upload, mapping, validation, cleaning, weekly build
FastAPI routes ──┼─> shared business logic ┼─ src/analytics   data quality, EDA, seasonality, alerts
                 │       (src/)            ├─ src/forecasting baseline, models, backtesting, evaluation, pipeline
                 │                         ├─ src/inventory   risk, stockout, overstock, replenishment
                 └─ src/visualization, src/reporting (charts, CSV/PDF/ZIP)
```
* `Workspace` (`src/data/upload_manager.py`) is the single state holder (pure Python). Any upload, removal or mapping change calls `reset_results()`, bumping `version` and discarding processed data, forecasts and inventory views, so stale results are never shown.
* Forecast and inventory results are cached under keys containing the version and relevant settings.
* Pages contain layout only; calculations live in `src/`. The API calls the same `run_forecast` and `build_inventory_view`.
* Storage: session memory only. Nothing is written to disk or sent to external services.
