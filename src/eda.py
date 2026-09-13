import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import boxcox, skew, zscore

from src.config import ARTIFACT_DIR


def run_eda(df: pd.DataFrame, output_dir: Path = ARTIFACT_DIR) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    numeric = df.select_dtypes(include=np.number)
    z = numeric.apply(zscore)
    outliers = (z.abs() > 3).sum().sort_values(ascending=False).to_dict()
    transforms = {}
    for col in ["ride_duration_min", "distance_km", "demand_supply_ratio", "trip_cost"]:
        values = df[col].to_numpy(float)
        shifted = values - values.min() + 1e-3 if values.min() <= 0 else values
        transformed, lam = boxcox(shifted)
        transforms[col] = {
            "lambda": float(lam), "skew_before": float(skew(values)),
            "skew_after": float(skew(transformed)),
        }
    report = {
        "rows": len(df), "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": int(df.isna().sum().sum()),
        "duration_cost_correlation": float(df["ride_duration_min"].corr(df["trip_cost"])),
        "zscore_outliers": outliers, "boxcox": transforms,
    }
    (output_dir / "eda_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

