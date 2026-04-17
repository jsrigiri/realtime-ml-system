from config import ENTRY_THRESHOLD, EXIT_THRESHOLD, POSITION_SIZE


class StreamProcessor:
    def __init__(self, feature_builder, model, execution_simulator, metrics_tracker):
        self.feature_builder = feature_builder
        self.model = model
        self.execution_simulator = execution_simulator
        self.metrics_tracker = metrics_tracker
        self.prev_features = None
        self.prev_mid = None

        self.prior_batch_metrics = None
        self.session_records = []

    def set_prior_batch_metrics(self, metrics: dict | None):
        self.prior_batch_metrics = metrics

    def reset_session_records(self):
        self.session_records = []

    def process_tick(self, tick: dict):
        features = self.feature_builder.update(
            tick,
            prior_batch_metrics=self.prior_batch_metrics
        )

        if features is None:
            return {
                "status": "warming_up",
                "tick": tick,
            }

        pred = self.metrics_tracker.time_call(self.model.predict, features)

        if pred > ENTRY_THRESHOLD:
            target_position = POSITION_SIZE
        elif pred < -ENTRY_THRESHOLD:
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

        current_record = dict(features)
        current_record["mid"] = tick["mid"]
        self.session_records.append(current_record)

        if self.prev_features is not None and self.prev_mid is not None:
            realized_target = tick["mid"] - self.prev_mid
            self.metrics_tracker.add_prediction(pred, realized_target)

            if hasattr(self.model, "update"):
                self.model.update(self.prev_features, realized_target)

        self.prev_features = features
        self.prev_mid = tick["mid"]

        return {
            "status": "ok",
            "prediction": float(pred),
            "features": features,
            "portfolio": snap,
        }