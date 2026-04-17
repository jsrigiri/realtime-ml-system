import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


class BatchRegressor:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42,
        )
        self.is_fitted = False
        self.feature_names = [
            "mid",
            "ret_1",
            "ret_mean",
            "ret_std",
            "mom_3",
            "spread",
            "volume_mean",
        ]

    def fit_from_frame(self, df: pd.DataFrame):
        X = df[self.feature_names]
        y = df["target"]
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, features: dict) -> float:
        if not self.is_fitted:
            return 0.0
        X = pd.DataFrame([[features[n] for n in self.feature_names]], columns=self.feature_names)
        return float(self.model.predict(X)[0])