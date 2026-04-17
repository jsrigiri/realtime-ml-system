from src.stream.data_stream import stream_ticks
from src.features.online_features import OnlineFeatureBuilder
from src.models.online_model import OnlineRegressor
from src.execution.simulator import ExecutionSimulator
from src.monitoring.metrics import MetricsTracker
from src.stream.processor import StreamProcessor
from src.utils.helpers import pretty_float_dict
from config import (
    INITIAL_CAPITAL,
    TRANSACTION_COST,
    SLIPPAGE,
    POSITION_SIZE,
    ROLLING_WINDOW,
)

def main():
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

    count = 0
    last_result = None

    for tick in stream_ticks("data/historical_ticks.csv"):
        result = processor.process_tick(tick)
        last_result = result
        count += 1

        if count % 250 == 0 and result["status"] == "ok":
            print(f"Tick {count}: {pretty_float_dict(result['portfolio'])}")

    print("Final result:")
    if last_result and last_result["status"] == "ok":
        print(pretty_float_dict(last_result["portfolio"]))
    print("Metrics:")
    print(pretty_float_dict(metrics.summary()))

if __name__ == "__main__":
    main()