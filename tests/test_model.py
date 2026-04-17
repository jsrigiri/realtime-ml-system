from src.models.online_model import OnlineRegressor


def test_online_model_predict_and_update():
    model = OnlineRegressor()

    features = {
        "mid": 100.0,
        "ret_1": 0.1,
        "ret_mean": 0.05,
        "ret_std": 0.02,
        "mom_3": 0.2,
        "spread": 0.02,
        "volume_mean": 5.0,
    }

    pred_before = model.predict(features)
    assert isinstance(pred_before, float)

    model.update(features, 0.15)
    pred_after = model.predict(features)
    assert isinstance(pred_after, float)