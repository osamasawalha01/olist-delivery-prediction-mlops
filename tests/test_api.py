import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_prediction_persistence(monkeypatch):
    """Prevent API tests from writing prediction logs to PostgreSQL."""
    monkeypatch.setattr(
        "olist_mlops.inference.save_prediction",
        lambda **kwargs: None,
    )


VALID_ORDER = {
    "order_purchase_timestamp": "2018-01-15 14:30:00",
    "customer_zip_code_prefix_x": 12345,
    "customer_city_x": "sao paulo",
    "customer_state_x": "SP",
    "customer_zip_code_prefix_y": 54321,
    "customer_city_y": "rio de janeiro",
    "customer_state_y": "RJ",
}


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_model_info():
    response = client.get("/model")

    assert response.status_code == 200
    assert response.json()["model_name"] == "logistic_regression"
    assert response.json()["model_version"] == "1"


def test_metrics():
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]

    metrics_text = response.text

    assert "olist_api_requests_total" in metrics_text
    assert "olist_api_request_latency_seconds" in metrics_text
    assert "olist_predictions_total" in metrics_text
    assert "olist_prediction_errors_total" in metrics_text


def test_single_prediction():
    response = client.post(
        "/predict",
        json=VALID_ORDER,
    )

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in [0, 1]
    assert result["label"] in ["on_time", "late"]
    assert 0.0 <= result["probability"] <= 1.0
    assert result["model_name"] == "logistic_regression"
    assert result["model_version"] == "1"


def test_batch_prediction():
    response = client.post(
        "/predict/batch",
        json={
            "orders": [
                VALID_ORDER,
                VALID_ORDER,
            ]
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert "predictions" in result
    assert len(result["predictions"]) == 2

    for prediction in result["predictions"]:
        assert prediction["prediction"] in [0, 1]
        assert prediction["label"] in ["on_time", "late"]
        assert 0.0 <= prediction["probability"] <= 1.0
        assert prediction["model_version"] == "1"


def test_invalid_prediction():
    invalid_order = VALID_ORDER.copy()
    invalid_order["customer_state_x"] = "XX"

    response = client.post(
        "/predict",
        json=invalid_order,
    )

    assert response.status_code == 422
