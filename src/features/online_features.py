from collections import deque
import numpy as np


class OnlineFeatureBuilder:
    def __init__(self, window: int = 20):
        self.window = window
        self.mid_prices = deque(maxlen=window + 5)
        self.volumes = deque(maxlen=window)

    def update(self, tick: dict, prior_batch_metrics: dict | None = None):
        self.mid_prices.append(tick["mid"])
        self.volumes.append(tick["volume"])

        # only need 3 ticks to start producing basic features
        if len(self.mid_prices) < 3:
            return None

        mids = np.array(self.mid_prices, dtype=float)
        rets = np.diff(mids)

        def safe_diff(arr, k):
            if len(arr) > k:
                return float(arr[-1] - arr[-(k + 1)])
            return 0.0

        def safe_zscore(arr, lookback):
            if len(arr) >= lookback:
                window = arr[-lookback:]
                return float((window[-1] - np.mean(window)) / (np.std(window) + 1e-8))
            return 0.0

        features = {
            "mid": float(mids[-1]),
            "ret_1": float(rets[-1]) if len(rets) >= 1 else 0.0,
            "ret_3": safe_diff(mids, 3),
            "ret_5": safe_diff(mids, 5),
            "ret_mean": float(rets.mean()) if len(rets) >= 1 else 0.0,
            "ret_std": float(rets.std()) if len(rets) > 1 else 0.0,
            "mom_3": safe_diff(mids, 3),
            "mom_5": safe_diff(mids, 5),
            "spread": float(tick["ask"] - tick["bid"]),
            "volume_mean": float(np.mean(self.volumes)) if len(self.volumes) > 0 else 0.0,
            "zscore_5": safe_zscore(mids, 5),
        }

        prior_batch_metrics = prior_batch_metrics or {}
        features["batch_mse_prev"] = float(prior_batch_metrics.get("batch_mse", 0.0))
        features["batch_mae_prev"] = float(prior_batch_metrics.get("batch_mae", 0.0))
        features["batch_mean_pred_prev"] = float(prior_batch_metrics.get("batch_mean_pred", 0.0))
        features["batch_hit_ratio_prev"] = float(prior_batch_metrics.get("batch_hit_ratio", 0.0))
        features["batch_session_pnl_prev"] = float(prior_batch_metrics.get("batch_session_pnl", 0.0))

        return features