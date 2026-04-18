from collections import deque
from config import ENTRY_THRESHOLD, EXIT_THRESHOLD, POSITION_SIZE, ONLINE_HORIZON, MIN_WARMUP_TICKS, USE_SIGNAL_SMOOTHING, SIGNAL_SMOOTHING_ALPHA


class StreamProcessor:
    def __init__(self, feature_builder, model, execution_simulator, metrics_tracker):
        self.feature_builder = feature_builder
        self.model = model
        self.execution_simulator = execution_simulator
        self.metrics_tracker = metrics_tracker

        self.prior_batch_metrics = None
        self.session_records = []

        self.feature_queue = deque()
        self.mid_queue = deque()

        self.tick_count = 0
        self.smoothed_pred = 0.0

    def set_prior_batch_metrics(self, metrics: dict | None):
        self.prior_batch_metrics = metrics

    def reset_session_records(self):
        self.session_records = []

    def process_tick(self, tick: dict):
        self.tick_count += 1

        features = self.feature_builder.update(
            tick,
            prior_batch_metrics=self.prior_batch_metrics
        )

        if features is None:
            return {"status": "warming_up", "tick": tick}

        raw_pred = self.metrics_tracker.time_call(self.model.predict, features)

        if USE_SIGNAL_SMOOTHING:
            self.smoothed_pred = (
                SIGNAL_SMOOTHING_ALPHA * raw_pred
                + (1 - SIGNAL_SMOOTHING_ALPHA) * self.smoothed_pred
            )
            pred = self.smoothed_pred
        else:
            pred = raw_pred

        trend_filter = features.get("mom_5", 0.0)

        if self.tick_count < MIN_WARMUP_TICKS:
            target_position = 0
        else:
            if pred > ENTRY_THRESHOLD and trend_filter > 0:
                target_position = POSITION_SIZE
            elif pred < -ENTRY_THRESHOLD and trend_filter < 0:
                target_position = -POSITION_SIZE
            elif abs(pred) < EXIT_THRESHOLD:
                target_position = 0
            else:
                target_position = self.execution_simulator.position

        self.execution_simulator.trade_to_target(
            target_position=target_position,
            bid=tick["bid"],
            ask=tick["ask"],
        )

        snap = self.execution_simulator.snapshot(tick["mid"])
        self.metrics_tracker.add_equity(snap["equity"])

        current_record = dict(features)
        current_record["mid"] = tick["mid"]
        self.session_records.append(current_record)

        self.feature_queue.append(features)
        self.mid_queue.append(tick["mid"])

        if len(self.mid_queue) > ONLINE_HORIZON:
            old_features = self.feature_queue.popleft()
            old_mid = self.mid_queue.popleft()
            realized_target = tick["mid"] - old_mid

            self.metrics_tracker.add_prediction(pred, realized_target)

            if hasattr(self.model, "update"):
                self.model.update(old_features, realized_target)

        return {
            "status": "ok",
            "prediction": float(pred),
            "features": features,
            "portfolio": snap,
        }