from pathlib import Path
import json
import pandas as pd
import streamlit as st

from src.config import ARTIFACT_DIR
from src.inference import predict_fare

st.set_page_config(page_title="RideSurge", page_icon="🚕", layout="wide")
st.title("🚕 RideSurge")
st.caption("Demand-adaptive fare intelligence and revenue simulation")

summary_path = ARTIFACT_DIR / "run_summary.json"
if not summary_path.exists():
    st.error("No trained model found. Run `python train.py` first.")
    st.stop()
summary = json.loads(summary_path.read_text(encoding="utf-8"))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Duration ↔ cost", f'{summary["eda"]["duration_cost_correlation"]:.3f}')
c2.metric("CatBoost R²", f'{summary["catboost"]["r2"]:.3f}')
c3.metric("CatBoost RMSE", f'{summary["catboost"]["rmse"]:.2f}')
c4.metric("Simulated uplift", f'{summary["revenue"]["profit_uplift_pct"]:.1f}%')

tab1, tab2, tab3 = st.tabs(["Fare estimator", "Model benchmark", "Explainability"])
with tab1:
    a, b = st.columns(2)
    with a:
        duration = st.number_input("Ride duration (minutes)", 1.0, 240.0, 28.0)
        distance = st.number_input("Distance (km)", .1, 200.0, 10.0)
        demand = st.number_input("Active demand", 1, 1000, 75)
        drivers = st.number_input("Available drivers", 1, 1000, 40)
        zone = st.selectbox("Pickup zone", ["Airport", "Central", "North", "South", "Tech Park", "University"])
    with b:
        vehicle = st.selectbox("Vehicle", ["Economy", "Comfort", "Premium"])
        traffic = st.slider("Traffic index", .3, 4.0, 1.4)
        weather = st.slider("Weather severity", 0.0, 5.0, .4)
        hour = st.slider("Hour", 0, 23, 18)
        weekend = st.checkbox("Weekend")
    if st.button("Recommend fare", type="primary"):
        result = predict_fare({"ride_duration_min": duration, "distance_km": distance,
            "demand": demand, "available_drivers": drivers, "traffic_index": traffic,
            "weather_severity": weather, "hour": hour, "is_weekend": int(weekend),
            "pickup_zone": zone, "vehicle_type": vehicle})
        st.success(f'Recommended fare: ${result["recommended_fare"]:.2f}')
        st.write(f'Market ratio: {result["demand_supply_ratio"]:.2f} · Surge: {result["surge_multiplier"]:.2f}×')
with tab2:
    st.dataframe(pd.read_csv(ARTIFACT_DIR / "model_comparison.csv"), use_container_width=True, hide_index=True)
with tab3:
    ranking = pd.read_csv(ARTIFACT_DIR / "shap_feature_rankings.csv").set_index("feature")
    st.bar_chart(ranking)
    st.caption("Mean absolute CatBoost SHAP value on the held-out test set.")

