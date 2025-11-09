# eda_coffee_shop

This repository contains exploratory data analysis (EDA) and Phase 1 advanced analytics for a coffee shop's sales data. The goal is to analyze transactions, revenue patterns, product performance and produce actionable business insights.

## Repository layout

Top-level files and directories (important ones shown):

- `basic_analysis.py`       - Core EDA scripts with initial preprocessing and visualizations.
- `coffee_sales_data_analysis.py` - Comprehensive EDA script with extended visualizations and exports.
- `phase1_advanced_analytics.py` - (Suggested) Phase 1 advanced analytics module (forecasting, segmentation, anomaly detection).
- `run_phase1.py`          - Runner script to execute Phase 1 analysis.
- `requirements.txt`      - Python dependencies used by the project.
- `data/`                 - Raw data files (CSV). Example: `Coffe_sales.csv`.
- `outputs/`              - Generated analysis outputs (CSV summaries, PNG charts, reports).

## What this project does

- Loads transaction-level data (hour, payment type, coffee/product, amount, date/time).
- Cleans and standardizes columns and datetimes.
- Produces descriptive statistics (total revenue, avg transaction, counts by product, payment methods).
- Visualizes temporal patterns (hourly, daily, monthly), product performance, price distributions, and correlations.
- Exports summary CSVs and charts into the `outputs/` folder.

## How to run the basic EDA

1. Create and activate a Python virtual environment (recommended).

On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:
```powershell
pip install -r requirements.txt
```

3. Run the main EDA script(s):
```powershell
python basic_analysis.py
# or
python coffee_sales_data_analysis.py
```

4. Outputs (charts and CSV summaries) will be placed under `outputs/`.

## Phase 1 — Advanced Analytics (Forecasting, Segmentation, Anomaly Detection)

Phase 1 includes time series forecasting (daily revenue), customer time-based segmentation, and anomaly detection. This code is designed to live in a separate module so you can import the preprocessing from `basic_analysis.py`.

Suggested files for Phase 1:
- `phase1_advanced_analytics.py` — a module/class that:
	- Loads preprocessed data (or calls functions in `basic_analysis.py`).
	- Builds daily aggregates for forecasting, trains a forecasting model (e.g., RandomForest or dedicated time-series model), and saves forecasts to `outputs/phase1/`.
	- Runs KMeans (or similar) to create time-based customer segments and saves them.
	- Performs anomaly detection (Z-score and IsolationForest) on daily revenue and flags unusual days.

- `run_phase1.py` — thin runner script to call the Phase 1 module and export results.

To run Phase 1 (after adding `phase1_advanced_analytics.py`):
```powershell
python run_phase1.py
```

## Notes, warnings and tips

- The dataset in `data/Coffe_sales.csv` had a column naming inconsistency (e.g. `Coffe_sales.csv` vs `Coffee_sales.csv`) — verify the filename and path before running.
- When combining date and time columns, pandas may issue a warning: `Could not infer format, so each element will be parsed individually...`. Fix this by specifying the exact format when calling `pd.to_datetime(..., format="%Y-%m-%d %H:%M:%S")` or by using `dt.strftime` on `date` and creating a normalized string.
- If you see `UsageError: Cell magic '%%' not found`, remove Jupyter-specific magics (`%%`) from `.py` files — those only work in `.ipynb` notebooks.

## Outputs

- `outputs/` will contain CSV summary files (hourly/daily/product reports) and PNG charts. Check this folder after running scripts.

## Next steps / advanced ideas

- Implement Phase 1 module and runner as separate files (keeps code modular).
- Add unit tests for key preprocessing steps (date parsing, categorical mapping).
- Create a small CLI wrapper or a single Jupyter notebook to explore outputs interactively.
- Optional: add an automated report (PDF/HTML) that compiles the key charts and metrics.

## Contact

If you need help wiring Phase 1 into the repository or want me to create `phase1_advanced_analytics.py` and `run_phase1.py` files, tell me which approach you prefer and I will add them.

---
Generated/updated on:
