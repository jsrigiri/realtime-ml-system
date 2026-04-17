import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def batch_feature_frame():
    np.random.seed(42)

    n = 40
    df = pd.DataFrame({
        "mid": np.linspace(100, 102, n),
        "ret_1": np.random.normal(0, 0.1, n),
        "ret_mean": np.random.normal(0, 0.05, n),
        "ret_std": np.abs(np.random.normal(0.02, 0.01, n)),
        "mom_3": np.random.normal(0, 0.15, n),
        "spread": np.random.uniform(0.01, 0.03, n),
        "volume_mean": np.random.uniform(1, 10, n),
        "batch_mse_prev": np.random.uniform(0, 0.5, n),
        "batch_mae_prev": np.random.uniform(0, 0.5, n),
        "batch_mean_pred_prev": np.random.normal(0, 0.1, n),
        "batch_hit_ratio_prev": np.random.uniform(0.4, 0.8, n),
        "batch_session_pnl_prev": np.random.normal(0, 1.0, n),
    })
    return df


@pytest.fixture
def regression_training_frame(batch_feature_frame):
    df = batch_feature_frame.copy()
    df["target"] = (
        0.4 * df["ret_1"]
        + 0.3 * df["mom_3"]
        - 0.2 * df["spread"]
        + 0.1 * df["batch_mean_pred_prev"]
    )
    df["future_ret_for_eval"] = df["target"]
    return df


@pytest.fixture
def classification_training_frame(batch_feature_frame):
    df = batch_feature_frame.copy()
    raw = (
        0.5 * df["ret_1"]
        + 0.4 * df["mom_3"]
        - 0.2 * df["spread"]
        + 0.1 * df["batch_hit_ratio_prev"]
    )
    df["future_ret_for_eval"] = raw
    df["target"] = (raw > raw.median()).astype(int)
    return df


@pytest.fixture
def session_records():
    np.random.seed(42)
    records = []
    mids = 100 + np.cumsum(np.random.normal(0, 0.05, 60))

    for i in range(60):
        records.append({
            "mid": float(mids[i]),
            "ret_1": float(np.random.normal(0, 0.1)),
            "ret_mean": float(np.random.normal(0, 0.05)),
            "ret_std": float(abs(np.random.normal(0.02, 0.01))),
            "mom_3": float(np.random.normal(0, 0.15)),
            "spread": float(np.random.uniform(0.01, 0.03)),
            "volume_mean": float(np.random.uniform(1, 10)),
            "batch_mse_prev": 0.0,
            "batch_mae_prev": 0.0,
            "batch_mean_pred_prev": 0.0,
            "batch_hit_ratio_prev": 0.0,
            "batch_session_pnl_prev": 0.0,
        })
    return records