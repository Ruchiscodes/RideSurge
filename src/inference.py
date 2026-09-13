from functools import lru_cache
from pathlib import Path
import pandas as pd
from catboost import CatBoostRegressor
from src.config import ARTIFACT_DIR, FEATURES
from src.data import demand_supply_multiplier


@lru_cache
def load_model(path: str = str(ARTIFACT_DIR / "catboost_model.cbm")):
    model = CatBoostRegressor()
    model.load_model(path)
    return model


def predict_fare(payload: dict) -> dict:
    row = payload.copy()
    ratio, multiplier = demand_supply_multiplier(pd.Series([row["demand"]]), pd.Series([row["available_drivers"]]))
    row["demand_supply_ratio"] = float(ratio.iloc[0])
    row["surge_multiplier"] = float(multiplier.iloc[0])
    fare = float(load_model().predict(pd.DataFrame([row])[FEATURES])[0])
    return {"recommended_fare": round(max(fare, 3.5), 2), "surge_multiplier": round(row["surge_multiplier"], 3), "demand_supply_ratio": round(row["demand_supply_ratio"], 3)}

