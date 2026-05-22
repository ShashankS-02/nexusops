"""
Tests for the FastAPI Gateway
"""
import pytest
from fastapi.testclient import TestClient

from api.main import app, metric_buffer, incidents


@pytest.fixture(autouse=True)
def clear_state():
    """Reset in-memory state between tests."""
    metric_buffer.clear()
    incidents.clear()
    app.state.sentinel = None  # Don't load models in tests
    yield
    metric_buffer.clear()
    incidents.clear()


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_has_required_fields(self, client):
        data = client.get("/api/v1/health").json()
        assert "status" in data
        assert "timestamp" in data
        assert "active_incidents" in data

    def test_health_status_is_healthy(self, client):
        data = client.get("/api/v1/health").json()
        assert data["status"] == "healthy"


class TestMetricIngestion:
    def test_ingest_single_metric(self, client):
        payload = {
            "pod_name": "payment-service-7d4f9b-xk2pq",
            "namespace": "prod",
            "cpu_usage_percent": 30.0,
            "memory_usage_percent": 50.0,
            "request_latency_ms": 120.0,
            "error_rate_percent": 0.5,
            "network_io_kbps": 2000.0,
        }
        response = client.post("/api/v1/metrics/ingest", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["pod_name"] == payload["pod_name"]
        assert data["anomaly_detected"] is False  # no model loaded

    def test_metric_buffered(self, client):
        payload = {
            "pod_name": "test-pod",
            "namespace": "test",
            "cpu_usage_percent": 25.0,
            "memory_usage_percent": 40.0,
            "request_latency_ms": 100.0,
            "error_rate_percent": 0.1,
            "network_io_kbps": 1000.0,
        }
        client.post("/api/v1/metrics/ingest", json=payload)
        buffer_response = client.get("/api/v1/metrics/buffer")
        assert buffer_response.status_code == 200
        buffer_data = buffer_response.json()
        assert "test-pod" in buffer_data
        assert buffer_data["test-pod"]["count"] == 1

    def test_invalid_cpu_rejected(self, client):
        """CPU > 100 should fail validation."""
        payload = {
            "pod_name": "test-pod",
            "namespace": "test",
            "cpu_usage_percent": 150.0,  # invalid
            "memory_usage_percent": 40.0,
            "request_latency_ms": 100.0,
            "error_rate_percent": 0.1,
            "network_io_kbps": 1000.0,
        }
        response = client.post("/api/v1/metrics/ingest", json=payload)
        assert response.status_code == 422  # Pydantic validation error

    def test_negative_latency_rejected(self, client):
        payload = {
            "pod_name": "test-pod",
            "namespace": "test",
            "cpu_usage_percent": 30.0,
            "memory_usage_percent": 40.0,
            "request_latency_ms": -10.0,  # invalid
            "error_rate_percent": 0.1,
            "network_io_kbps": 1000.0,
        }
        response = client.post("/api/v1/metrics/ingest", json=payload)
        assert response.status_code == 422


class TestIncidentEndpoints:
    def test_list_incidents_empty(self, client):
        response = client.get("/api/v1/incidents")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["incidents"] == []

    def test_get_nonexistent_incident(self, client):
        response = client.get("/api/v1/incidents/nonexistent")
        assert response.status_code == 404

    def test_approve_nonexistent_incident(self, client):
        response = client.post(
            "/api/v1/incidents/nonexistent/approve",
            json={"approved": True, "approver": "test"},
        )
        assert response.status_code == 404


class TestAlertWebhook:
    def test_prometheus_webhook(self, client):
        """Test receiving a Prometheus Alertmanager webhook."""
        payload = {
            "alerts": [
                {
                    "labels": {
                        "alertname": "HighCPU",
                        "pod": "payment-service-xyz",
                        "namespace": "prod",
                        "severity": "high",
                    },
                    "status": "firing",
                }
            ]
        }
        response = client.post("/api/v1/alerts/webhook", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert len(data["incidents_created"]) == 1

    def test_webhook_creates_incident(self, client):
        """Alert webhook should create a traceable incident."""
        payload = {
            "alerts": [
                {
                    "labels": {
                        "alertname": "MemoryLeak",
                        "pod": "data-pipeline-abc",
                        "namespace": "staging",
                        "severity": "critical",
                    }
                }
            ]
        }
        client.post("/api/v1/alerts/webhook", json=payload)
        incidents_response = client.get("/api/v1/incidents")
        assert incidents_response.json()["total"] == 1
