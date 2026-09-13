from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
RANDOM_STATE = 42
TARGET = "trip_cost"

NUMERIC_FEATURES = [
    "ride_duration_min", "distance_km", "demand", "available_drivers",
    "demand_supply_ratio", "surge_multiplier", "traffic_index",
    "weather_severity", "hour", "is_weekend",
]
CATEGORICAL_FEATURES = ["pickup_zone", "vehicle_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

