from collections import deque
import numpy as np


class OnlineFeatureBuilder:
    def __init__(self, window: int = 20):
        self.window = window
        self.mid_prices = deque(maxlen=window + 1)
        self.volumes = deque(maxlen=window)

    def update(self, tick: dict, prior_batch_metrics: dict | None = None):
        self.mid_prices.append(tick["mid"])
        self.volumes.append(tick["volume"])

        if len(self.mid_prices) < 3:
            return None

        mids = np.array(self.mid_prices, dtype=float)
        rets = np.diff(mids)

        features = {
            "mid": float(mids[-1]),
            "ret_1": float(rets[-1]),
            "ret_mean": float(rets.mean()),
            "ret_std": float(rets.std()) if len(rets) > 1 else 0.0,
            "mom_3": float(mids[-1] - mids[-3]) if len(mids) >= 3 else 0.0,
            "spread": float(tick["ask"] - tick["bid"]),
            "volume_mean": float(np.mean(self.volumes)),
        }

        # session-level feedback from prior batch model
        prior_batch_metrics = prior_batch_metrics or {}
        features["batch_mse_prev"] = float(prior_batch_metrics.get("batch_mse", 0.0))
        features["batch_mae_prev"] = float(prior_batch_metrics.get("batch_mae", 0.0))
        features["batch_mean_pred_prev"] = float(prior_batch_metrics.get("batch_mean_pred", 0.0))
        features["batch_hit_ratio_prev"] = float(prior_batch_metrics.get("batch_hit_ratio", 0.0))
        features["batch_session_pnl_prev"] = float(prior_batch_metrics.get("batch_session_pnl", 0.0))

        return features