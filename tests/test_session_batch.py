import pytest

from src.models.session_batch import build_session_frame, evaluate_batch_session


def test_build_session_frame_regression(session_records, monkeypatch):
    monkeypatch.setattr("src.models.session_batch.BATCH_TASK_TYPE", "regression")
    monkeypatch.setattr("src.models.session_batch.BATCH_HORIZON_K", 5)

    df = build_session_frame(session_records, prior_batch_metrics=None)

    assert "target" in df.columns
    assert "future_ret_for_eval" in df.columns
    assert len(df) < len(session_records)  # because of shift(-k)
    assert df["target"].isnull().sum() == 0


def test_build_session_frame_classification(session_records, monkeypatch):
    monkeypatch.setattr("src.models.session_batch.BATCH_TASK_TYPE", "classification")
    monkeypatch.setattr("src.models.session_batch.BATCH_HORIZON_K", 5)

    df = build_session_frame(session_records, prior_batch_metrics=None)

    assert "target" in df.columns
    assert set(df["target"].unique()).issubset({0, 1})
    assert "future_ret_for_eval" in df.columns


def test_build_session_frame_uses_prior_batch_metrics(session_records, monkeypatch):
    monkeypatch.setattr("src.models.session_batch.BATCH_TASK_TYPE", "regression")
    monkeypatch.setattr("src.models.session_batch.BATCH_HORIZON_K", 3)

    prior = {
        "batch_mse": 0.25,
        "batch_mae": 0.20,
        "batch_mean_pred": 0.05,
        "batch_hit_ratio": 0.7,
        "batch_session_pnl": 1.5,
    }

    df = build_session_frame(session_records, prior_batch_metrics=prior)

    assert (df["batch_mse_prev"] == 0.25).all()
    assert (df["batch_mae_prev"] == 0.20).all()
    assert (df["batch_mean_pred_prev"] == 0.05).all()
    assert (df["batch_hit_ratio_prev"] == 0.7).all()
    assert (df["batch_session_pnl_prev"] == 1.5).all()


def test_build_session_frame_k_horizon_changes_length(session_records, monkeypatch):
    monkeypatch.setattr("src.models.session_batch.BATCH_TASK_TYPE", "regression")

    monkeypatch.setattr("src.models.session_batch.BATCH_HORIZON_K", 1)
    df1 = build_session_frame(session_records, None)

    monkeypatch.setattr("src.models.session_batch.BATCH_HORIZON_K", 10)
    df2 = build_session_frame(session_records, None)

    assert len(df2) < len(df1)


def test_evaluate_batch_session_regression(session_records, monkeypatch):
    monkeypatch.setattr("src.models.session_batch.BATCH_TASK_TYPE", "regression")
    monkeypatch.setattr("src.models.session_batch.BATCH_HORIZON_K", 5)
    monkeypatch.setattr("src.models.batch_model.BATCH_MODEL_TYPE", "linear_reg")

    df = build_session_frame(session_records, None)
    model, metrics = evaluate_batch_session(df)

    assert model.is_fitted is True
    assert "batch_horizon_k" in metrics
    assert "batch_task_type" in metrics
    assert "batch_model_type" in metrics
    assert "batch_used_device" in metrics
    assert "batch_mse" in metrics
    assert "batch_mae" in metrics
    assert "batch_hit_ratio" in metrics
    assert "batch_session_pnl" in metrics


def test_evaluate_batch_session_classification(session_records, monkeypatch):
    monkeypatch.setattr("src.models.session_batch.BATCH_TASK_TYPE", "classification")
    monkeypatch.setattr("src.models.session_batch.BATCH_HORIZON_K", 5)
    monkeypatch.setattr("src.models.batch_model.BATCH_MODEL_TYPE", "logistic")

    df = build_session_frame(session_records, None)
    model, metrics = evaluate_batch_session(df)

    assert model.is_fitted is True
    assert "batch_horizon_k" in metrics
    assert "batch_task_type" in metrics
    assert "batch_model_type" in metrics
    assert "batch_used_device" in metrics
    assert "batch_hit_ratio" in metrics
    assert "batch_session_pnl" in metrics
    assert "batch_accuracy" in metrics


def test_evaluate_batch_session_invalid_task(session_records, monkeypatch):
    monkeypatch.setattr("src.models.session_batch.BATCH_TASK_TYPE", "invalid")
    monkeypatch.setattr("src.models.session_batch.BATCH_HORIZON_K", 5)

    with pytest.raises(ValueError, match="Unsupported BATCH_TASK_TYPE"):
        build_session_frame(session_records, None)