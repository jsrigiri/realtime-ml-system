import numpy as np
import pytest

from src.models.batch_model import BatchModel


def assert_valid_device_string(used_device: str):
    assert isinstance(used_device, str)
    assert (
        used_device == "cpu"
        or "gpu" in used_device
        or "fallback_cpu" in used_device
    )


def test_batch_model_feature_names():
    model = BatchModel()

    expected = [
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

    assert model.feature_names == expected


def test_batch_model_linear_regression_fit_predict(regression_training_frame, monkeypatch):
    monkeypatch.setattr("src.models.batch_model.BATCH_MODEL_TYPE", "linear_reg")

    model = BatchModel()
    model.fit_from_frame(regression_training_frame)

    preds = model.predict_from_frame(regression_training_frame)

    assert len(preds) == len(regression_training_frame)
    assert model.is_fitted is True
    assert_valid_device_string(model.used_device)


def test_batch_model_logistic_fit_predict(classification_training_frame, monkeypatch):
    monkeypatch.setattr("src.models.batch_model.BATCH_MODEL_TYPE", "logistic")

    model = BatchModel()
    model.fit_from_frame(classification_training_frame)

    preds = model.predict_from_frame(classification_training_frame)

    assert len(preds) == len(classification_training_frame)
    assert set(np.unique(preds)).issubset({0, 1})
    assert model.is_fitted is True
    assert_valid_device_string(model.used_device)


def test_batch_model_single_predict_before_fit():
    model = BatchModel()

    features = {
        "mid": 100.0,
        "ret_1": 0.1,
        "ret_mean": 0.05,
        "ret_std": 0.02,
        "mom_3": 0.15,
        "spread": 0.02,
        "volume_mean": 5.0,
        "batch_mse_prev": 0.0,
        "batch_mae_prev": 0.0,
        "batch_mean_pred_prev": 0.0,
        "batch_hit_ratio_prev": 0.0,
        "batch_session_pnl_prev": 0.0,
    }

    pred = model.predict(features)
    assert pred == 0.0


def test_batch_model_single_predict_after_fit(regression_training_frame, monkeypatch):
    monkeypatch.setattr("src.models.batch_model.BATCH_MODEL_TYPE", "linear_reg")

    model = BatchModel()
    model.fit_from_frame(regression_training_frame)

    features = regression_training_frame[model.feature_names].iloc[0].to_dict()
    pred = model.predict(features)

    assert isinstance(pred, float)


def test_batch_model_xgboost_reg_gpu_requested(regression_training_frame, monkeypatch):
    pytest.importorskip("xgboost")
    monkeypatch.setattr("src.models.batch_model.BATCH_MODEL_TYPE", "xgboost_reg")
    monkeypatch.setattr("src.models.batch_model.USE_GPU", True)

    model = BatchModel()
    model.fit_from_frame(regression_training_frame)

    preds = model.predict_from_frame(regression_training_frame)

    assert len(preds) == len(regression_training_frame)
    assert_valid_device_string(model.used_device)


def test_batch_model_xgboost_clf_gpu_requested(classification_training_frame, monkeypatch):
    pytest.importorskip("xgboost")
    monkeypatch.setattr("src.models.batch_model.BATCH_MODEL_TYPE", "xgboost_clf")
    monkeypatch.setattr("src.models.batch_model.USE_GPU", True)

    model = BatchModel()
    model.fit_from_frame(classification_training_frame)

    preds = model.predict_from_frame(classification_training_frame)

    assert len(preds) == len(classification_training_frame)
    assert set(np.unique(preds)).issubset({0, 1})
    assert_valid_device_string(model.used_device)


def test_batch_model_lightgbm_reg_gpu_requested(regression_training_frame, monkeypatch):
    pytest.importorskip("lightgbm")
    monkeypatch.setattr("src.models.batch_model.BATCH_MODEL_TYPE", "lightgbm_reg")
    monkeypatch.setattr("src.models.batch_model.USE_GPU", True)

    model = BatchModel()
    model.fit_from_frame(regression_training_frame)

    preds = model.predict_from_frame(regression_training_frame)

    assert len(preds) == len(regression_training_frame)
    assert_valid_device_string(model.used_device)


def test_batch_model_lightgbm_clf_gpu_requested(classification_training_frame, monkeypatch):
    pytest.importorskip("lightgbm")
    monkeypatch.setattr("src.models.batch_model.BATCH_MODEL_TYPE", "lightgbm_clf")
    monkeypatch.setattr("src.models.batch_model.USE_GPU", True)

    model = BatchModel()
    model.fit_from_frame(classification_training_frame)

    preds = model.predict_from_frame(classification_training_frame)

    assert len(preds) == len(classification_training_frame)
    assert set(np.unique(preds)).issubset({0, 1})
    assert_valid_device_string(model.used_device)


def test_batch_model_invalid_type(monkeypatch, regression_training_frame):
    monkeypatch.setattr("src.models.batch_model.BATCH_MODEL_TYPE", "not_real")

    model = BatchModel()

    with pytest.raises(ValueError, match="Unsupported BATCH_MODEL_TYPE"):
        model.fit_from_frame(regression_training_frame)