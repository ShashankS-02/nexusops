"""Demo step 2 — inject 30 anomalous metrics and wait for the pipeline to pause."""
import time
import requests

BASE = "http://localhost:8000"

print("=" * 50)
print("  Injecting anomalous metrics: payment-service")
print("=" * 50)

incident_id = None
for i in range(30):
    resp = requests.post(f"{BASE}/api/v1/metrics/ingest", json={
        "pod_name": "payment-service",
        "namespace": "prod",
        "cpu_usage_percent": 93.0 + (i % 5),
        "memory_usage_percent": 90.0 + (i % 4),
        "request_latency_ms": 4700.0 + (i * 40),
        "error_rate_percent": 18.5 + (i % 3),
        "network_io_kbps": 460.0,
    }).json()

    score = resp.get("anomaly_score")
    if score is not None:
        filled = int(score * 30)
        bar = "X" * filled + "." * (30 - filled)
        print(f"  [{i+1:2d}/30] score={score:.3f}  [{bar}]")

    if resp.get("anomaly_detected"):
        incident_id = resp["incident_id"]
        print()
        print(f"  ANOMALY DETECTED  score={score:.3f} > threshold 0.60")
        print(f"  Incident ID : {incident_id}")
        print(f"  LangGraph pipeline started in background...")
        break

if not incident_id:
    print("  No anomaly triggered")
    raise SystemExit(1)

print()
print("  Running: Detective -> Oracle -> Surgeon (propose)...")
for _ in range(30):
    time.sleep(2)
    status = requests.get(f"{BASE}/api/v1/incidents/{incident_id}").json().get("status")
    if status == "awaiting_approval":
        print("  Pipeline paused at HITL gate -- awaiting human approval")
        print("=" * 50)
        break
