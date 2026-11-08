from fastapi.testclient import TestClient
from insightpilot.api.main import app


def test_health_chat_forecast_and_validation():
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        r = c.post("/chat", json={"question": "forecast SKU-002 in West for 7 days"}).json()
        assert r["tool_calls"][0]["tool"] == "get_forecast"
        assert c.get("/forecast", params={"sku": "SKU-001", "region": "North"}).status_code == 200
        assert c.get("/forecast", params={"sku": "NOPE", "region": "North"}).status_code == 422
        assert c.post("/chat", json={"question": "x"}).status_code == 422


def test_api_key_enforced_when_configured(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    with TestClient(app) as c:
        assert c.post("/chat", json={"question": "top products"}).status_code == 401
        ok = c.post("/chat", json={"question": "top products"}, headers={"x-api-key": "secret"})
        assert ok.status_code == 200
