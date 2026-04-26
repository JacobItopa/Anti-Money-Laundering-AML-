"""
Integration Tests for the FastAPI prediction service (src/app/main.py).

Uses FastAPI's TestClient (backed by httpx) to spin up the full ASGI app
in-process — no server process needed. Tests the full request → preprocess
→ model → response lifecycle.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from src.app.main import app

client = TestClient(app)

# ── Valid payload fixture ──────────────────────────────────────────────────────

VALID_PAYLOAD = {
    "Timestamp": "2022/09/01 00:20",
    "From_Bank": "10",
    "Account": "8000EBD30",
    "To_Bank": "10",
    "Account_1": "8000EBD30",
    "Amount_Received": 10.50,
    "Receiving_Currency": "US Dollar",
    "Amount_Paid": 10.50,
    "Payment_Currency": "US Dollar",
    "Payment_Format": "Reinvestment",
}

# ── Health endpoint ────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_status_healthy(self):
        data = client.get("/health").json()
        assert data == {"status": "healthy"}

# ── /predict endpoint — happy path ────────────────────────────────────────────

class TestPredictEndpoint:
    def test_predict_returns_200(self):
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200, response.text

    def test_predict_response_has_required_keys(self):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert "fraud_probability" in data
        assert "is_laundering" in data
        assert "flagged_for_review" in data

    def test_fraud_probability_is_float(self):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert isinstance(data["fraud_probability"], float)

    def test_fraud_probability_between_0_and_1(self):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert 0.0 <= data["fraud_probability"] <= 1.0

    def test_is_laundering_is_boolean(self):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert isinstance(data["is_laundering"], bool)

    def test_flagged_for_review_is_boolean(self):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert isinstance(data["flagged_for_review"], bool)

    def test_low_probability_not_flagged(self):
        """A normal small-amount same-bank transfer should have very low fraud probability."""
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        # If probability is low it should not be flagged
        if data["fraud_probability"] < 0.5:
            assert data["flagged_for_review"] is False

    def test_unknown_currency_handled_gracefully(self):
        """An unseen currency should not crash the API (mapped to freq 0.0)."""
        payload = {**VALID_PAYLOAD, "Receiving_Currency": "UnknownCoin9999"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_high_amount_request_succeeds(self):
        """Large transaction amounts should not cause errors."""
        payload = {**VALID_PAYLOAD, "Amount_Received": 9_999_999.99, "Amount_Paid": 9_999_999.99}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

# ── /predict endpoint — validation errors ─────────────────────────────────────

class TestPredictValidation:
    def test_missing_field_returns_422(self):
        """Pydantic must reject payloads with missing required fields."""
        incomplete = {k: v for k, v in VALID_PAYLOAD.items() if k != "Amount_Received"}
        response = client.post("/predict", json=incomplete)
        assert response.status_code == 422

    def test_wrong_type_returns_422(self):
        """Strings in numeric fields must be rejected."""
        bad_payload = {**VALID_PAYLOAD, "Amount_Received": "not_a_number"}
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422

    def test_empty_payload_returns_422(self):
        response = client.post("/predict", json={})
        assert response.status_code == 422

# ── Response header checks ────────────────────────────────────────────────────

class TestResponseHeaders:
    def test_predict_response_has_process_time_header(self):
        """Latency middleware must inject X-Process-Time into every response."""
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert "x-process-time" in response.headers

    def test_process_time_is_positive_float(self):
        response = client.post("/predict", json=VALID_PAYLOAD)
        latency = float(response.headers["x-process-time"])
        assert latency > 0.0
