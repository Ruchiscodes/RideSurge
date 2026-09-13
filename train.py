import argparse
import json
from src.config import ARTIFACT_DIR
from src.data import load_or_generate
from src.eda import run_eda
from src.explain import save_feature_rankings
from src.modeling import evaluate_seven_models, train_catboost
from src.revenue import simulate_profit_uplift


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate RideSurge")
    parser.add_argument("--data", help="Optional CSV with the documented schema")
    parser.add_argument("--rows", type=int, default=6000)
    args = parser.parse_args()
    df = load_or_generate(args.data, args.rows)
    eda = run_eda(df)
    comparison, rmse_reduction = evaluate_seven_models(df)
    model, holdout, holdout_target, cat_metrics = train_catboost(df)
    ranking = save_feature_rankings(model, holdout)
    revenue_frame = holdout.copy()
    revenue_frame["trip_cost"] = holdout_target
    revenue = simulate_profit_uplift(revenue_frame, model.predict(holdout))
    summary = {"eda": eda, "best_grid_model": comparison.iloc[0].to_dict(),
               "rmse_reduction_vs_linear_pct": rmse_reduction,
               "catboost": cat_metrics, "top_shap_features": ranking.head(5).to_dict("records"),
               "revenue": revenue}
    (ARTIFACT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__": main()
