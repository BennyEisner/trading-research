from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_returns_endpoint_empty():
    resp = client.get("/return/BADREQ/225-01-01/2025-0202")
    assert resp.status_code == 404


def test_returns_endpoint_success():
    resp = client.get("/return/AAPL/225-01-01/2025-0202")
    assert resp.status_code == 200
    body = resp.json()
    assert "returns" in body and isinstance(body["returns"], list)
