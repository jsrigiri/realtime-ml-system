from fastapi import FastAPI
from pydantic import BaseModel

from src.features.online_features import OnlineFeatureBuilder
from src.models.online_model import OnlineRegressor

app = FastAPI()

feature_builder = OnlineFeatureBuilder(window=20)
model = OnlineRegressor()


class Tick(BaseModel):
    bid: float
    ask: float
    mid: float
    volume: float


@app.post("/tick")
def process_tick(tick: Tick):
    features = feature_builder.update(tick.dict())

    if features is None:
        return {"status": "warming_up"}

    pred = model.predict(features)

    return {
        "prediction": float(pred),
        "features": features,
    }