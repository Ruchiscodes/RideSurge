from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.inference import predict_fare

app = FastAPI(title="RideSurge Pricing API", version="1.0.0")


class RideRequest(BaseModel):
    ride_duration_min: float = Field(gt=0, le=240)
    distance_km: float = Field(gt=0, le=200)
    demand: int = Field(gt=0)
    available_drivers: int = Field(gt=0)
    traffic_index: float = Field(ge=.3, le=4)
    weather_severity: float = Field(ge=0, le=5)
    hour: int = Field(ge=0, le=23)
    is_weekend: int = Field(ge=0, le=1)
    pickup_zone: Literal["Airport", "Central", "North", "South", "Tech Park", "University"]
    vehicle_type: Literal["Economy", "Comfort", "Premium"]


@app.get("/health")
def health(): return {"status": "ok"}


@app.post("/predict")
def predict(request: RideRequest):
    try:
        return predict_fare(request.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(503, "Model not trained. Run: python train.py") from exc

