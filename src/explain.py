import json
from pathlib import Path
import pandas as pd
from src.config import ARTIFACT_DIR


def save_feature_rankings(model, X: pd.DataFrame, output_dir: Path = ARTIFACT_DIR) -> pd.DataFrame:
    # CatBoost's exact SHAP implementation is fast and avoids explainer ambiguity.
    from catboost import Pool
    cat_idx = [X.columns.get_loc(c) for c in X.select_dtypes(include="object").columns]
    values = model.get_feature_importance(Pool(X, cat_features=cat_idx), type="ShapValues")[:, :-1]
    ranking = pd.DataFrame({"feature": X.columns, "mean_abs_shap": abs(values).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    ranking.to_csv(output_dir / "shap_feature_rankings.csv", index=False)
    return ranking

