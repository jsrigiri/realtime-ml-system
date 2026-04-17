import time


class MetricsTracker:
    def __init__(self):
        self.prediction_latencies = []
        self.predictions = []
        self.targets = []

    def time_call(self, func, *args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        self.prediction_latencies.append(elapsed)
        return result

    def add_prediction(self, pred: float, target: float):
        self.predictions.append(float(pred))
        self.targets.append(float(target))

    def summary(self):
        avg_latency = sum(self.prediction_latencies) / len(self.prediction_latencies) if self.prediction_latencies else 0.0

        mse = 0.0
        if self.predictions:
            mse = sum((p - y) ** 2 for p, y in zip(self.predictions, self.targets)) / len(self.predictions)

        return {
            "avg_latency_sec": float(avg_latency),
            "mse": float(mse),
            "n_predictions": len(self.predictions),
        }