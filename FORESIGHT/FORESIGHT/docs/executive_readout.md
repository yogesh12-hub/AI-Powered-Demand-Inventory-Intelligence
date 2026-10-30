# Executive readout — outline (6–10 slides)
No result below is pre-filled. Run FORESIGHT on your real data, then copy the numbers from **Reports & Exports → Executive summary / Model evaluation / Inventory risk report**.

1. **Business problem** – stockouts lose sales; overstock ties up cash; planning needs SKU-level weekly demand and explainable reorders.
2. **Dataset & quality** – source, period, SKUs, rows; issues from the Data Quality report and what cleaning did (see cleaning log).
3. **Methodology** – weekly aggregation, seasonal-naive baseline, candidate models, rolling-origin backtest, WAPE.
4. **Measured performance** – selected model WAPE vs baseline WAPE, evaluation period, held-out SKU-weeks, folds. Report only what the app shows.
5. **Inventory risk** – counts of High/Medium/Low stockout and overstock SKUs; category heatmap; estimated ₹ exposure (an estimate, not savings).
6. **Insights (≥3)** – from the Business insights export, each with its inventory-planning relevance.
7. **Recommendations** – top replenishment actions (Urgent/Soon), reorder spend, items to stop reordering.
8. **Limitations & next steps** – assumptions in `methodology.md`; promotions/price signals, supplier variability, fresher snapshots, deployment.
