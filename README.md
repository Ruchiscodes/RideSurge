# RideSurge

RideSurge is an end-to-end demand-adaptive dynamic-pricing project. It generates or ingests ride data, audits skew and outliers, creates bounded demand/supply surge features, tunes seven regressors with 5-fold `GridSearchCV`, trains a production CatBoost model, calculates exact SHAP rankings, serves fares through FastAPI, and estimates profit uplift using an explicit price-elasticity simulation.

> **Evidence note:** the numbers in a résumé are experimental results, not configuration values. On the included deterministic synthetic dataset, the current measured run is stored in `artifacts/run_summary.json`. Replace `data/rides.csv` with the original data to reproduce the original 0.93 correlation, 55% RMSE reduction, 0.89 R², and 18% simulated uplift claims. Never overwrite measured results with target claims.

## Architecture

```text
CSV / synthetic generator
        │
        ├── EDA: Z-score outliers + Box-Cox skew correction
        │
        ├── features: demand/supply ratio + bounded surge multiplier
        │
        ├── 7 models × 5-fold GridSearchCV
        │
        └── CatBoost ── SHAP rankings ── revenue simulation
                              │
                    FastAPI + Streamlit
```

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train.py --rows 6000
python -m uvicorn app:app --reload
```

Open `http://127.0.0.1:8000/docs`. In a second terminal, run `streamlit run dashboard.py` for the interactive dashboard.

Run tests with:

```powershell
python -m pytest -q
```

## Input schema

The CSV must contain `trip_cost` plus the feature columns in `src/config.py`: duration, distance, demand, available drivers, ratio, multiplier, traffic, weather, hour, weekend, pickup zone, and vehicle type. `timestamp` and `ride_id` are optional metadata. Run real data with:

```powershell
python train.py --data path\to\rides.csv
```

For raw data without engineered fields, compute them using `demand_supply_multiplier` in `src/data.py` before training.

## What each résumé point maps to

1. `src/eda.py` calculates absolute Z-score > 3 outliers, fitted Box-Cox lambdas, before/after skew, missingness, duplicates, and the duration/cost Pearson correlation.
2. `src/data.py` implements the market multiplier. `src/modeling.py` compares Linear Regression, Ridge, ElasticNet, KNN, Random Forest, Extra Trees, and Gradient Boosting using the same held-out split and five CV folds.
3. `src/modeling.py` saves `catboost_model.cbm`; `src/inference.py` loads it once for scalable inference; `src/explain.py` produces exact CatBoost SHAP rankings; `src/revenue.py` reports a clearly labeled simulation rather than a causal business claim.

## API example

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict `
  -ContentType application/json `
  -Body '{"ride_duration_min":28,"distance_km":10,"demand":75,"available_drivers":40,"traffic_index":1.4,"weather_severity":0.4,"hour":18,"is_weekend":0,"pickup_zone":"Central","vehicle_type":"Economy"}'
```

## Interpretation and limitations

- The multiplier is capped at 2.5× as a pricing guardrail.
- The test set is untouched by GridSearchCV and used once for final metrics.
- The uplift calculation assumes constant elasticity (recorded in its JSON output); validate elasticity through controlled experiments before production use.
- Production deployment should add authentication, request logging, drift checks, fairness monitoring, regional regulation checks, and model/version metadata.

