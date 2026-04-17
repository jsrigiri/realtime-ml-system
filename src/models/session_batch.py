import pandas as pd
import numpy as np

from src.models.batch_model import BatchModel
from config import BATCH_HORIZON_K, BATCH_TASK_TYPE


def build_session_frame(session_records: list[dict], prior_batch_metrics: dict | None = None) -> pd.DataFrame:
    df = pd.DataFrame(session_records).copy()
    prior_batch_metrics = prior_batch_metrics or {}

    df["batch_mse_prev"] = float(prior_batch_metrics.get("batch_mse", 0.0))
    df["batch_mae_prev"] = float(prior_batch_metrics.get("batch_mae", 0.0))
    df["batch_mean_pred_prev"] = float(prior_batch_metrics.get("batch_mean_pred", 0.0))
    df["batch_hit_ratio_prev"] = float(prior_batch_metrics.get("batch_hit_ratio", 0.0))
    df["batch_session_pnl_prev"] = float(prior_batch_metrics.get("batch_session_pnl", 0.0))

    future_ret = df["mid"].shift(-BATCH_HORIZON_K) - df["mid"]

    if BATCH_TASK_TYPE == "regression":
        df["target"] = future_ret
    elif BATCH_TASK_TYPE == "classification":
        df["target"] = (future_ret > 0).astype(int)
    else:
        raise ValueError(f"Unsupported BATCH_TASK_TYPE: {BATCH_TASK_TYPE}")

    df["future_ret_for_eval"] = future_ret

    df = df.dropna().reset_index(drop=True)
    return df


def evaluate_batch_session(df: pd.DataFrame):
    model = BatchModel()
    model.fit_from_frame(df)

    preds = model.predict_from_frame(df)
    y = df["target"].values
    future_ret = df["future_ret_for_eval"].values

    if BATCH_TASK_TYPE == "regression":
        mse = float(np.mean((preds - y) ** 2))
        mae = float(np.mean(np.abs(preds - y)))
        hit_ratio = float(np.mean((np.sign(preds) == np.sign(future_ret)).astype(float)))
        mean_pred = float(np.mean(preds))
        session_pnl = float(np.sum(np.sign(preds) * future_ret))

        metrics = {
            "batch_horizon_k": BATCH_HORIZON_K,
            "batch_task_type": BATCH_TASK_TYPE,
            "batch_model_type": model.model.__class__.__name__,
            "batch_used_device": model.used_device,
            "batch_mse": mse,
            "batch_mae": mae,
            "batch_mean_pred": mean_pred,
            "batch_hit_ratio": hit_ratio,
            "batch_session_pnl": session_pnl,
        }
        return model, metrics

    # classification
    if hasattr(model.model, "predict_proba"):
        probas = model.model.predict_proba(df[model.feature_names])[:, 1]
        pred_labels = (probas >= 0.5).astype(int)
        mean_pred = float(np.mean(probas))
    else:
        pred_labels = preds
        mean_pred = float(np.mean(preds))

    accuracy = float(np.mean((pred_labels == y).astype(float)))
    hit_ratio = float(np.mean((pred_labels == y).astype(float)))
    session_pnl = float(np.sum(np.where(pred_labels == 1, 1.0, -1.0) * future_ret))

    metrics = {
        "batch_horizon_k": BATCH_HORIZON_K,
        "batch_task_type": BATCH_TASK_TYPE,
        "batch_model_type": model.model.__class__.__name__,
        "batch_used_device": model.used_device,
        "batch_mse": 0.0,
        "batch_mae": 0.0,
        "batch_mean_pred": mean_pred,
        "batch_hit_ratio": hit_ratio,
        "batch_session_pnl": session_pnl,
        "batch_accuracy": accuracy,
    }
    return model, metrics