from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.features.online_features import OnlineFeatureBuilder
from src.models.online_model import OnlineRegressor
from src.execution.simulator import ExecutionSimulator
from src.monitoring.metrics import MetricsTracker
from src.stream.processor import StreamProcessor
from config import (
    INITIAL_CAPITAL,
    TRANSACTION_COST,
    SLIPPAGE,
    POSITION_SIZE,
    ROLLING_WINDOW,
)

app = FastAPI()

feature_builder = OnlineFeatureBuilder(window=ROLLING_WINDOW)
model = OnlineRegressor()
execution = ExecutionSimulator(
    initial_capital=INITIAL_CAPITAL,
    transaction_cost=TRANSACTION_COST,
    slippage=SLIPPAGE,
    position_size=POSITION_SIZE,
)
metrics = MetricsTracker()
processor = StreamProcessor(feature_builder, model, execution, metrics)


class TickRequest(BaseModel):
    mid: float = Field(..., example=100.12)
    bid: float = Field(..., example=100.11)
    ask: float = Field(..., example=100.13)
    volume: float = Field(..., example=5.0)


@app.get("/")
def root():
    return {"status": "ok", "message": "Use POST /tick or open /docs"}


@app.get("/metrics")
def get_metrics():
    return metrics.summary()


@app.post("/tick")
def process_tick(data: TickRequest):
    tick = data.dict()
    result = processor.process_tick(tick)
    return result