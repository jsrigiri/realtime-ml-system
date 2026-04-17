from src.features.online_features import OnlineFeatureBuilder


def test_online_features_build_after_warmup():
    builder = OnlineFeatureBuilder(window=5)

    ticks = [
        {"mid": 100.0, "bid": 99.99, "ask": 100.01, "volume": 2},
        {"mid": 100.1, "bid": 100.09, "ask": 100.11, "volume": 3},
        {"mid": 100.2, "bid": 100.19, "ask": 100.21, "volume": 4},
    ]

    out1 = builder.update(ticks[0])
    out2 = builder.update(ticks[1])
    out3 = builder.update(ticks[2])

    assert out1 is None
    assert out2 is None
    assert out3 is not None
    assert "ret_1" in out3
    assert "spread" in out3