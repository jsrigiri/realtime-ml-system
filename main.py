from collections import deque
import pandas as pd

from src.stream.data_stream import stream_ticks
from src.features.online_features import OnlineFeatureBuilder
from src.models.online_model import OnlineRegressor
from src.execution.simulator import ExecutionSimulator
from src.monitoring.metrics import MetricsTracker
from src.monitoring.plots import (
    ensure_artifacts_dir,
    plot_equity_curve,
    plot_predictions_vs_realized,
    plot_batch_session_pnl,
)
from src.stream.processor import StreamProcessor
from src.models.session_batch import build_session_frame, evaluate_batch_session
from src.utils.helpers import pretty_float_dict
from config import (
    INITIAL_CAPITAL,
    TRANSACTION_COST,
    SLIPPAGE,
    POSITION_SIZE,
    ROLLING_WINDOW,
    SESSION_SIZE,
    MAX_BATCH_SESSIONS,
)


def main():
    ensure_artifacts_dir()

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
    session_num = 1
    last_result = None
    prior_batch_metrics = None

    recent_session_frames = deque(maxlen=MAX_BATCH_SESSIONS)

    equity_curve = []
    batch_session_pnls = []
    batch_session_ids = []

    for tick in stream_ticks("data/historical_ticks.csv"):
        processor.set_prior_batch_metrics(prior_batch_metrics)
        result = processor.process_tick(tick)
        last_result = result
        count += 1

        if result["status"] == "ok":
            equity_curve.append(result["portfolio"]["equity"])

        if count % SESSION_SIZE == 0:
            print(f"--- End of session {session_num} ---")

            if processor.session_records:
                session_df = build_session_frame(
                    processor.session_records,
                    prior_batch_metrics=prior_batch_metrics
                )

                if len(session_df) > 5:
                    recent_session_frames.append(session_df)

                    if len(recent_session_frames) == 1:
                        combined_df = recent_session_frames[0].copy()
                    else:
                        combined_df = pd.concat(
                            list(recent_session_frames),
                            axis=0,
                            ignore_index=True
                        )

                    _, batch_metrics = evaluate_batch_session(combined_df)
                    prior_batch_metrics = batch_metrics

                    batch_session_pnls.append(batch_metrics.get("batch_session_pnl", 0.0))
                    batch_session_ids.append(session_num)

                    print(f"Batch metrics from last {len(recent_session_frames)} session(s):")
                    print(pretty_float_dict(batch_metrics))
                else:
                    print("Not enough session data to fit batch model.")
                    prior_batch_metrics = None

            processor.reset_session_records()
            session_num += 1

        if count % 250 == 0 and result["status"] == "ok":
            print(f"Tick {count}: {pretty_float_dict(result['portfolio'])}")

    print("Final result:")
    if last_result and last_result["status"] == "ok":
        print(pretty_float_dict(last_result["portfolio"]))

    print("Metrics:")
    print(pretty_float_dict(metrics.summary()))

    plot_equity_curve(equity_curve)
    plot_predictions_vs_realized(metrics.predictions, metrics.targets)
    plot_batch_session_pnl(batch_session_ids, batch_session_pnls)


if __name__ == "__main__":
    main()