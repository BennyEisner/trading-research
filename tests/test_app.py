from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_bad_date_order():
    response = client.get("/prices/AAPL/2025-06-01/2025-01-01")
    assert response.status_code == 400
