"""Data generation and loading.

The deterministic generator makes the project reproducible. Replace its output
with a real CSV having the same schema to use production data.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import DATA_DIR, RANDOM_STATE


def demand_supply_multiplier(demand: pd.Series, drivers: pd.Series) -> tuple[pd.Series, pd.Series]:
    ratio = demand / drivers.clip(lower=1)
    # Smooth, bounded pricing guardrail: no discount below 1x, max 2.5x.
    multiplier = (1 + 0.42 * np.maximum(ratio - 1, 0)).clip(1, 2.5)
    return ratio, multiplier


def generate_rides(n_rows: int = 6000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    zones = np.array(["Airport", "Central", "North", "South", "Tech Park", "University"])
    vehicles = np.array(["Economy", "Comfort", "Premium"])
    hour = rng.integers(0, 24, n_rows)
    weekend = rng.binomial(1, 2 / 7, n_rows)
    peak = (((hour >= 7) & (hour <= 10)) | ((hour >= 17) & (hour <= 21))).astype(int)
    zone = rng.choice(zones, n_rows, p=[.12, .25, .15, .14, .20, .14])
    vehicle = rng.choice(vehicles, n_rows, p=[.64, .25, .11])
    weather = np.clip(rng.gamma(1.4, .55, n_rows), 0, 3)
    traffic = np.clip(1 + .55 * peak + .18 * weather + rng.normal(0, .18, n_rows), .6, 2.5)
    distance = np.clip(rng.lognormal(2.05, .55, n_rows), .8, 42)
    duration = np.clip(distance * (2.05 + .85 * traffic) + rng.normal(1.5, 2.2, n_rows), 3, 125)
    zone_demand = pd.Series(zone).map({"Airport": 15, "Central": 22, "North": 5, "South": 4, "Tech Park": 17, "University": 9}).to_numpy()
    demand = np.maximum(5, rng.poisson(30 + 21 * peak + 8 * weekend + 5 * weather + zone_demand))
    drivers = np.maximum(5, rng.poisson(42 - 5 * peak - 2 * weekend + zone_demand * .25))
    ratio, multiplier = demand_supply_multiplier(pd.Series(demand), pd.Series(drivers))
    vehicle_rate = pd.Series(vehicle).map({"Economy": 1.0, "Comfort": 1.35, "Premium": 1.85}).to_numpy()
    # Cost is duration-led by design, while nonlinear market effects reward boosted trees.
    pre_surge_fare = 2.5 + vehicle_rate * (0.47 * duration + 0.18 * distance)
    market_effect = 1 + .10 * weather + .055 * traffic**2
    noise = rng.normal(0, 1.3 + .025 * duration, n_rows)
    cost = np.maximum(3.5, pre_surge_fare * multiplier.to_numpy() * market_effect + noise)
    frame = pd.DataFrame({
        "ride_id": [f"R{i:07d}" for i in range(n_rows)],
        "timestamp": pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365 * 24 * 60, n_rows), unit="m"),
        "pickup_zone": zone, "vehicle_type": vehicle, "hour": hour,
        "is_weekend": weekend, "weather_severity": weather.round(3),
        "traffic_index": traffic.round(3), "distance_km": distance.round(3),
        "ride_duration_min": duration.round(3), "demand": demand,
        "available_drivers": drivers, "demand_supply_ratio": ratio.round(4),
        "surge_multiplier": multiplier.round(4), "trip_cost": cost.round(2),
    })
    return frame


def load_or_generate(path: str | Path | None = None, n_rows: int = 6000) -> pd.DataFrame:
    target = Path(path) if path else DATA_DIR / "rides.csv"
    if target.exists():
        return pd.read_csv(target, parse_dates=["timestamp"])
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_rides(n_rows=n_rows)
    frame.to_csv(target, index=False)
    return frame

