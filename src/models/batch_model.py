import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

from config import BATCH_MODEL_TYPE, USE_GPU, LIGHTGBM_GPU_BACKEND


class BatchModel:
    def __init__(self):
        self.feature_names = [
            "mid",
            "ret_1",
            "ret_mean",
            "ret_std",
            "mom_3",
            "spread",
            "volume_mean",
            "batch_mse_prev",
            "batch_mae_prev",
            "batch_mean_pred_prev",
            "batch_hit_ratio_prev",
            "batch_session_pnl_prev",
        ]
        self.model = None
        self.is_fitted = False
        self.used_device = "cpu"

    def _build_model(self, use_gpu: bool):
        used_device = "cpu"

        if BATCH_MODEL_TYPE == "linear_reg":
            return LinearRegression(), used_device

        if BATCH_MODEL_TYPE == "logistic":
            return LogisticRegression(max_iter=1000), used_device

        if BATCH_MODEL_TYPE == "xgboost_reg":
            params = {
                "n_estimators": 150,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "random_state": 42,
            }
            if USE_GPU and use_gpu:
                params["device"] = "cuda"
                used_device = "gpu"
            return XGBRegressor(**params), used_device

        if BATCH_MODEL_TYPE == "xgboost_clf":
            params = {
                "n_estimators": 150,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "random_state": 42,
                "eval_metric": "logloss",
            }
            if USE_GPU and use_gpu:
                params["device"] = "cuda"
                used_device = "gpu"
            return XGBClassifier(**params), used_device

        if BATCH_MODEL_TYPE == "lightgbm_reg":
            params = {
                "n_estimators": 150,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "random_state": 42,
            }
            if USE_GPU and use_gpu:
                params["device_type"] = LIGHTGBM_GPU_BACKEND
                used_device = f"gpu:{LIGHTGBM_GPU_BACKEND}"
            return LGBMRegressor(**params), used_device

        if BATCH_MODEL_TYPE == "lightgbm_clf":
            params = {
                "n_estimators": 150,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "random_state": 42,
            }
            if USE_GPU and use_gpu:
                params["device_type"] = LIGHTGBM_GPU_BACKEND
                used_device = f"gpu:{LIGHTGBM_GPU_BACKEND}"
            return LGBMClassifier(**params), used_device

        raise ValueError(f"Unsupported BATCH_MODEL_TYPE: {BATCH_MODEL_TYPE}")

    def fit_from_frame(self, df: pd.DataFrame):
        X = df[self.feature_names]
        y = df["target"]

        model, intended_device = self._build_model(use_gpu=True)

        try:
            model.fit(X, y)
            self.model = model
            self.used_device = intended_device
        except Exception as e:
            fallback_model, _ = self._build_model(use_gpu=False)
            fallback_model.fit(X, y)
            self.model = fallback_model
            self.used_device = f"{intended_device} -> fallback_cpu ({type(e).__name__})"

        self.is_fitted = True

    def predict_from_frame(self, df: pd.DataFrame):
        X = df[self.feature_names]
        return self.model.predict(X)

    def predict(self, features: dict):
        if not self.is_fitted:
            return 0.0

        X = pd.DataFrame(
            [[features[n] for n in self.feature_names]],
            columns=self.feature_names
        )

        if hasattr(self.model, "predict_proba"):
            return float(self.model.predict_proba(X)[0, 1])

        return float(self.model.predict(X)[0])