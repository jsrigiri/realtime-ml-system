import numpy as np
from sklearn.linear_model import SGDRegressor


class OnlineRegressor:
    def __init__(self):
        self.model = SGDRegressor(
            loss="huber",
            penalty="l2",
            alpha=0.001,
            random_state=42,
            max_iter=1,
            learning_rate="adaptive",
            eta0=0.01,
            warm_start=True,
        )
        self.is_initialized = False
        self.feature_names = [
            "mid",
            "ret_1",
            "ret_3",
            "ret_5",
            "ret_mean",
            "ret_std",
            "mom_3",
            "mom_5",
            "spread",
            "volume_mean",
            "zscore_5",
            "batch_mse_prev",
            "batch_mae_prev",
            "batch_mean_pred_prev",
            "batch_hit_ratio_prev",
            "batch_session_pnl_prev",
        ]

    def _to_array(self, features: dict):
        return np.array(
            [[features.get(name, 0.0) for name in self.feature_names]],
            dtype=float
        )

    def predict(self, features: dict) -> float:
        X = self._to_array(features)
        if not self.is_initialized:
            return 0.0
        return float(self.model.predict(X)[0])

    def update(self, features: dict, y: float):
        X = self._to_array(features)
        y_arr = np.array([y], dtype=float)

        if not self.is_initialized:
            self.model.partial_fit(X, y_arr)
            self.is_initialized = True
        else:
            self.model.partial_fit(X, y_arr)