import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import ARTIFACT_DIR, CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, RANDOM_STATE, TARGET


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC_FEATURES),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL_FEATURES),
    ])


def model_searches() -> dict:
    return {
        "LinearRegression": (LinearRegression(), {}),
        "Ridge": (Ridge(), {"model__alpha": [0.1, 1.0, 10.0]}),
        "ElasticNet": (ElasticNet(max_iter=5000), {"model__alpha": [.001, .01], "model__l1_ratio": [.2, .7]}),
        "KNN": (KNeighborsRegressor(), {"model__n_neighbors": [5, 11], "model__weights": ["distance"]}),
        "RandomForest": (RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1), {"model__n_estimators": [150], "model__max_depth": [12, None]}),
        "ExtraTrees": (ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=1), {"model__n_estimators": [150], "model__max_depth": [12, None]}),
        "GradientBoosting": (GradientBoostingRegressor(random_state=RANDOM_STATE), {"model__n_estimators": [120, 200], "model__max_depth": [2, 3]}),
    }


def evaluate_seven_models(df: pd.DataFrame, output_dir: Path = ARTIFACT_DIR) -> tuple[pd.DataFrame, float]:
    X_train, X_test, y_train, y_test = train_test_split(df[FEATURES], df[TARGET], test_size=.2, random_state=RANDOM_STATE)
    rows, fitted = [], {}
    for name, (estimator, grid) in model_searches().items():
        pipe = Pipeline([("prep", _preprocessor()), ("model", estimator)])
        # n_jobs=1 is intentionally portable across constrained Windows/Linux hosts.
        search = GridSearchCV(pipe, grid or [{}], cv=5, scoring="neg_root_mean_squared_error", n_jobs=1)
        search.fit(X_train, y_train)
        pred = search.predict(X_test)
        rows.append({"model": name, "rmse": mean_squared_error(y_test, pred) ** .5,
                     "mae": mean_absolute_error(y_test, pred), "r2": r2_score(y_test, pred),
                     "best_params": json.dumps(search.best_params_)})
        fitted[name] = search.best_estimator_
    results = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "model_comparison.csv", index=False)
    baseline = float(results.loc[results.model == "LinearRegression", "rmse"].iloc[0])
    best = float(results.rmse.iloc[0])
    uplift = (baseline - best) / baseline * 100
    joblib.dump(fitted[results.model.iloc[0]], output_dir / "best_sklearn_pipeline.joblib")
    return results, uplift


def train_catboost(df: pd.DataFrame, output_dir: Path = ARTIFACT_DIR):
    from catboost import CatBoostRegressor
    X_train, X_test, y_train, y_test = train_test_split(df[FEATURES], df[TARGET], test_size=.2, random_state=RANDOM_STATE)
    cat_idx = [X_train.columns.get_loc(c) for c in CATEGORICAL_FEATURES]
    model = CatBoostRegressor(iterations=500, depth=7, learning_rate=.06, loss_function="RMSE", verbose=False, random_seed=RANDOM_STATE)
    model.fit(X_train, y_train, cat_features=cat_idx)
    pred = model.predict(X_test)
    metrics = {"rmse": float(mean_squared_error(y_test, pred) ** .5), "mae": float(mean_absolute_error(y_test, pred)), "r2": float(r2_score(y_test, pred))}
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_dir / "catboost_model.cbm"))
    (output_dir / "catboost_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return model, X_test, y_test, metrics
