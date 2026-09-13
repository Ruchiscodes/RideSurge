# RideSurge

RideSurge is an end-to-end demand-adaptive dynamic-pricing project. It generates or ingests ride data, audits skew and outliers, creates bounded demand/supply surge features, tunes seven regressors with 5-fold `GridSearchCV`, trains a production CatBoost model, calculates exact SHAP rankings, serves fares through FastAPI, and estimates profit uplift using an explicit price-elasticity simulation.

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

 
 

 
 
