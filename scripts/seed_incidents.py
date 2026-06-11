"""
Seed the dashboard with a diverse incident set for testing.

Fires off a matrix of incidents covering:
  - Different severities (marginal, clear, critical)
  - Different anomaly shapes (CPU, memory, latency, errors, network, cascade)
  - Different namespaces (prod, staging, dev)
  - Different lifecycle stages (left in awaiting_approval, executed, failed)

Usage:
    python scripts/seed_incidents.py                  # seed the default set
    python scripts/seed_incidents.py --count 20       # seed more
    python scripts/seed_incidents.py --reset          # leave nothing pending (approve+reject everything)
    python scripts/seed_incidents.py --scenario memory_leak --only

Run after `uvicorn api.main:app --port 8000` is up.
"""
from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass

import requests

BASE = "http://localhost:8000"

# ── Scenario definitions ──────────────────────────────────────────────────────
# Each scenario is tuned so the 30-sample injection reliably crosses the
# Sentinel threshold (0.60), but at different rates and shapes so Detective's
# RAG retrieval picks distinct root causes.


@dataclass(frozen=True)
class Scenario:
    key: str
    pod: str
    namespace: str
    description: str
    expected_severity: str  # "marginal" | "clear" | "critical"

    # Metric ranges (low, high) over the 30-sample window
    cpu: tuple[float, float]
    mem: tuple[float, float]
    latency_ms: tuple[float, float]
    error_rate: tuple[float, float]
    net_kbps: tuple[float, float]

    # Lifecycle: "leave_pending" | "approve" | "reject"
    disposition: str = "leave_pending"


SCENARIOS: list[Scenario] = [
    # ── Critical, classic shapes — should resolve fast via RAG ──────────────
    Scenario(
        key="cpu_spike",
        pod="payment-service",
        namespace="prod",
        description="Sustained CPU spike from a runaway request handler",
        expected_severity="critical",
        cpu=(93.0, 99.0),
        mem=(60.0, 72.0),
        latency_ms=(4500.0, 5800.0),
        error_rate=(18.5, 22.0),
        net_kbps=(380.0, 520.0),
        disposition="approve",
    ),
    Scenario(
        key="memory_leak",
        pod="order-service",
        namespace="prod",
        description="Linear RSS growth — OOM kill imminent",
        expected_severity="critical",
        cpu=(55.0, 68.0),
        mem=(88.0, 97.0),
        latency_ms=(800.0, 1400.0),
        error_rate=(2.5, 4.0),
        net_kbps=(180.0, 220.0),
        disposition="leave_pending",  # wait at HITL gate
    ),
    Scenario(
        key="redis_pool_exhaustion",
        pod="auth-service",
        namespace="prod",
        description="Redis connection pool maxed — token validation failing",
        expected_severity="clear",
        cpu=(42.0, 55.0),
        mem=(60.0, 70.0),
        latency_ms=(2800.0, 3600.0),
        error_rate=(28.0, 35.0),
        net_kbps=(120.0, 180.0),
        disposition="approve",
    ),
    # ── Cascade — exercises Oracle's blast-radius prediction ────────────────
    Scenario(
        key="cascade_upstream",
        pod="api-gateway",
        namespace="prod",
        description="Upstream gateway failure cascading to 4 downstream services",
        expected_severity="critical",
        cpu=(75.0, 88.0),
        mem=(70.0, 85.0),
        latency_ms=(6500.0, 9000.0),
        error_rate=(45.0, 60.0),
        net_kbps=(50.0, 90.0),
        disposition="leave_pending",
    ),
    # ── Marginal anomaly — tests false-positive risk ────────────────────────
    Scenario(
        key="marginal_latency",
        pod="search-service",
        namespace="prod",
        description="Borderline p99 latency drift — could be normal or early warning",
        expected_severity="marginal",
        cpu=(45.0, 58.0),
        mem=(55.0, 68.0),
        latency_ms=(900.0, 1300.0),
        error_rate=(1.5, 3.0),
        net_kbps=(2100.0, 2400.0),
        disposition="reject",  # human decides this is noise
    ),
    # ── Staging — same shape, different namespace (priority differs) ────────
    Scenario(
        key="staging_memory_leak",
        pod="user-service",
        namespace="staging",
        description="Memory leak in staging deploy — low urgency",
        expected_severity="clear",
        cpu=(50.0, 62.0),
        mem=(85.0, 94.0),
        latency_ms=(700.0, 1100.0),
        error_rate=(2.0, 3.5),
        net_kbps=(150.0, 200.0),
        disposition="approve",
    ),
    # ── Network saturation — distinct shape (high net, low CPU) ─────────────
    Scenario(
        key="network_saturation",
        pod="data-pipeline",
        namespace="prod",
        description="Egress saturation — backpressure on data pipeline",
        expected_severity="clear",
        cpu=(48.0, 60.0),
        mem=(62.0, 75.0),
        latency_ms=(2200.0, 3000.0),
        error_rate=(4.5, 7.0),
        net_kbps=(48000.0, 65000.0),
        disposition="leave_pending",
    ),
    # ── Error-rate surge with low resource usage — tests root-cause ambiguity
    Scenario(
        key="error_surge_no_resource",
        pod="notification-svc",
        namespace="prod",
        description="High error rate with normal CPU/memory — probably a config bug",
        expected_severity="clear",
        cpu=(28.0, 38.0),
        mem=(48.0, 58.0),
        latency_ms=(180.0, 280.0),
        error_rate=(32.0, 42.0),
        net_kbps=(1800.0, 2100.0),
        disposition="approve",
    ),
    # ── GPU OOM — should fail remediation (no auto-fix for GPU) ─────────────
    Scenario(
        key="gpu_oom",
        pod="recommendation-engine",
        namespace="prod",
        description="GPU OOM on inference batch — remediation likely to fail",
        expected_severity="critical",
        cpu=(88.0, 95.0),
        mem=(92.0, 98.0),
        latency_ms=(5500.0, 7000.0),
        error_rate=(38.0, 45.0),
        net_kbps=(280.0, 380.0),
        disposition="approve",
    ),
    # ── Recurring incident — same pod, second time ──────────────────────────
    Scenario(
        key="recurring_cpu",
        pod="payment-service",
        namespace="prod",
        description="Same CPU spike as before — Detective should find prior incident",
        expected_severity="critical",
        cpu=(91.0, 97.0),
        mem=(62.0, 74.0),
        latency_ms=(4200.0, 5500.0),
        error_rate=(17.0, 21.0),
        net_kbps=(390.0, 510.0),
        disposition="leave_pending",
    ),
    # ── Dev environment ─────────────────────────────────────────────────────
    Scenario(
        key="dev_test_anomaly",
        pod="ci-runner",
        namespace="dev",
        description="Anomaly during CI test run — should be deprioritized",
        expected_severity="clear",
        cpu=(82.0, 92.0),
        mem=(75.0, 85.0),
        latency_ms=(1800.0, 2400.0),
        error_rate=(8.0, 12.0),
        net_kbps=(200.0, 280.0),
        disposition="reject",
    ),
    # ── Cold-start latency — tests if model handles non-stationary baselines
    Scenario(
        key="cold_start",
        pod="ml-inference",
        namespace="prod",
        description="Cold-start latency after pod scale-up",
        expected_severity="marginal",
        cpu=(70.0, 82.0),
        mem=(58.0, 70.0),
        latency_ms=(3500.0, 4500.0),
        error_rate=(0.5, 1.5),
        net_kbps=(800.0, 1200.0),
        disposition="reject",  # known false positive pattern
    ),
]


def jitter(lo_hi: tuple[float, float], i: int, total: int) -> float:
    """Interpolate within range with a slight upward drift + noise."""
    lo, hi = lo_hi
    progress = i / max(1, total - 1)
    base = lo + (hi - lo) * progress
    return round(base + random.uniform(-1.5, 1.5), 2)


def inject_scenario(s: Scenario, samples: int = 30, dwell: float = 0.05) -> str | None:
    """Push `samples` metric points for one scenario. Returns the incident_id or None."""
    print(f"  -> {s.key:25s}  {s.pod}/{s.namespace}", end="", flush=True)

    incident_id: str | None = None
    for i in range(samples):
        try:
            resp = requests.post(
                f"{BASE}/api/v1/metrics/ingest",
                json={
                    "pod_name": s.pod,
                    "namespace": s.namespace,
                    "cpu_usage_percent": jitter(s.cpu, i, samples),
                    "memory_usage_percent": jitter(s.mem, i, samples),
                    "request_latency_ms": jitter(s.latency_ms, i, samples),
                    "error_rate_percent": jitter(s.error_rate, i, samples),
                    "network_io_kbps": jitter(s.net_kbps, i, samples),
                },
                timeout=5,
            ).json()
        except Exception as e:
            print(f"  !! ingest failed: {e}")
            return None

        if resp.get("anomaly_detected") and not incident_id:
            incident_id = resp.get("incident_id")
            score = resp.get("anomaly_score") or 0.0
            print(f"  triggered @ sample {i+1:2d}  score={score:.3f}  id={incident_id[:8]}")
            return incident_id
        time.sleep(dwell)

    final_score = (resp.get("anomaly_score") if resp else None) or 0.0
    print(f"  no trigger after {samples} samples (final score={final_score:.3f})")
    return None


def dispose_incident(incident_id: str, disposition: str, scenario_key: str) -> None:
    """Apply the configured lifecycle action to the incident."""
    if disposition == "leave_pending":
        return

    # Wait briefly for the LangGraph pipeline to reach awaiting_approval
    for _ in range(20):
        try:
            inc = requests.get(f"{BASE}/api/v1/incidents/{incident_id}", timeout=5).json()
            if inc.get("status") == "awaiting_approval":
                break
        except Exception:
            pass
        time.sleep(0.5)

    payload = {"approved": disposition == "approve", "approver": "seeder"}
    if disposition == "reject":
        payload["reason"] = f"Seeder-rejected: known false-positive pattern ({scenario_key})"

    try:
        requests.post(f"{BASE}/api/v1/incidents/{incident_id}/approve", json=payload, timeout=5)
        print(f"     dispose={disposition}")
    except Exception as e:
        print(f"     !! dispose failed: {e}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=len(SCENARIOS), help="how many scenarios to inject")
    parser.add_argument("--scenario", type=str, default=None, help="run only this scenario key")
    parser.add_argument("--only", action="store_true", help="if --scenario set, run just that one")
    parser.add_argument("--dwell", type=float, default=0.05, help="seconds between metric samples")
    parser.add_argument("--gap", type=float, default=1.5, help="seconds between scenarios")
    args = parser.parse_args()

    # Health check
    try:
        h = requests.get(f"{BASE}/api/v1/health", timeout=3).json()
        if not h.get("sentinel_loaded"):
            print("Sentinel model is not loaded — refusing to seed.")
            return 1
    except Exception as e:
        print(f"Backend at {BASE} is not reachable: {e}")
        return 1

    # Build the scenario plan
    if args.scenario:
        plan = [s for s in SCENARIOS if s.key == args.scenario]
        if not plan:
            print(f"No scenario named {args.scenario!r}. Options: {[s.key for s in SCENARIOS]}")
            return 1
    else:
        plan = SCENARIOS[: args.count]

    print(f"\nSeeding {len(plan)} incident{'s' if len(plan) != 1 else ''}...\n")

    triggered: list[tuple[Scenario, str]] = []
    for s in plan:
        incident_id = inject_scenario(s, dwell=args.dwell)
        if incident_id:
            triggered.append((s, incident_id))
        time.sleep(args.gap)

    print(f"\n{len(triggered)} incidents triggered. Dispatching lifecycle actions...\n")
    for s, incident_id in triggered:
        if s.disposition != "leave_pending":
            print(f"  -> {s.key:25s}  id={incident_id[:8]}", end="")
            dispose_incident(incident_id, s.disposition, s.key)

    print("\nDone. Refresh the dashboard to see the populated incident list.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
