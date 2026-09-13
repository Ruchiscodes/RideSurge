import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import ARTIFACT_DIR


def simulate_profit_uplift(df: pd.DataFrame, predictions: np.ndarray, output_dir: Path = ARTIFACT_DIR) -> dict:
    """Compare flat-price profit with predicted dynamic pricing.

    Acceptance probability uses a documented constant-elasticity demand curve;
    operating cost is distance/time based. This is a simulation, not a causal claim.
    """
    actual = df["trip_cost"].to_numpy()
    operating_cost = 1.8 + .33 * df["distance_km"].to_numpy() + .09 * df["ride_duration_min"].to_numpy()
    flat_price = np.full(len(df), np.median(actual))
    dynamic_price = np.clip(predictions, flat_price * .75, flat_price * 2.2)
    elasticity = 1.15
    flat_accept = np.clip((actual / flat_price) ** elasticity, .35, 1)
    dynamic_accept = np.clip((actual / dynamic_price) ** elasticity, .35, 1)
    flat_profit = np.sum((flat_price - operating_cost) * flat_accept)
    dynamic_profit = np.sum((dynamic_price - operating_cost) * dynamic_accept)
    report = {"flat_profit": float(flat_profit), "dynamic_profit": float(dynamic_profit),
              "profit_uplift_pct": float((dynamic_profit / flat_profit - 1) * 100),
              "assumed_price_elasticity": elasticity, "rides_simulated": len(df)}
    (output_dir / "revenue_simulation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

