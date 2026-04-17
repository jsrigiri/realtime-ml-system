from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_metrics():
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "avg_latency_sec" in r.json()