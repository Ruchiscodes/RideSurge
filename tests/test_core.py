import numpy as np
from src.data import demand_supply_multiplier, generate_rides
from src.eda import run_eda
from src.revenue import simulate_profit_uplift


def test_generated_schema_and_bounds(tmp_path):
    df = generate_rides(200)
    assert len(df) == 200
    assert (df.trip_cost >= 3.5).all()
    assert df.demand_supply_ratio.notna().all()
    report = run_eda(df, tmp_path)
    assert report["missing_values"] == 0


def test_multiplier_is_bounded_and_monotonic():
    ratio, mult = demand_supply_multiplier(__import__("pandas").Series([10, 20, 50]), __import__("pandas").Series([20, 20, 20]))
    assert np.all(np.diff(mult) >= 0)
    assert mult.between(1, 2.5).all()


def test_revenue_report(tmp_path):
    df = generate_rides(100)
    report = simulate_profit_uplift(df, df.trip_cost.to_numpy(), tmp_path)
    assert set(report) >= {"flat_profit", "dynamic_profit", "profit_uplift_pct"}

