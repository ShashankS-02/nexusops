"""
NexusOps FastAPI Gateway
========================

This is the central HTTP entry point for NexusOps.
It receives:
  - Metric ingestion from the simulator / real Prometheus scrapes
  - Webhook alerts from CloudWatch / Prometheus Alertmanager
  - REST queries about current incidents

Routes:
  POST /api/v1/metrics/ingest    → Buffer metrics, trigger Sentinel
  POST /api/v1/alerts/webhook    → Receive external alert payloads
  GET  /api/v1/incidents         → List active incidents
  GET  /api/v1/incidents/{id}    → Get incident details
  POST /api/v1/incidents/{id}/approve → Human approval for remediation
  GET  /api/v1/health            → Liveness probe (for k8s)
  GET  /metrics                  → Prometheus metrics endpoint

Design Notes:
  - Fully async (asyncio + httpx)
  - Background task queue for agent graph execution
  - In-memory incident store (Phase 1 — PostgreSQL in Phase 3)
"""
from __future__ import annotations

import asyncio
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel

from nexusops.config import settings
from nexusops.models import (
    AnomalyAlert,
    Incident,
    IncidentStatus,
    LogEntry,
    MetricPoint,
    Severity,
)

# ── Prometheus Metrics (meta-observability) ────────────────────────────────────

METRICS_INGESTED = Counter(
    "nexusops_metrics_ingested_total",
    "Total metric points ingested",
    ["namespace"],
)
ALERTS_RECEIVED = Counter(
    "nexusops_alerts_received_total",
    "Total alert webhooks received",
    ["severity"],
)
INCIDENTS_CREATED = Counter(
    "nexusops_incidents_created_total",
    "Total incidents created",
)
INGEST_LATENCY = Histogram(
    "nexusops_ingest_latency_seconds",
    "Metric ingestion latency",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)


# ── In-Memory Stores (Phase 1) ────────────────────────────────────────────────

# Recent metric buffer: pod_name → deque of MetricPoints (last 100)
metric_buffer: dict[str, deque[MetricPoint]] = {}

# Active incidents: incident_id → Incident
incidents: dict[str, Incident] = {}

# Background tasks queue
background_tasks_queue: asyncio.Queue = asyncio.Queue()


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    # On startup: load the ML models (lazy — only if available)
    print("🚀 NexusOps API starting up...")
    try:
        from agents.sentinel.detector import SentinelDetector
        app.state.sentinel = SentinelDetector()
        print("  ✓ Sentinel detector loaded")
    except Exception as e:
        print(f"  ⚠ Sentinel not loaded (run training first): {e}")
        app.state.sentinel = None

    yield  # App is running

    # On shutdown
    print("👋 NexusOps API shutting down...")


# ── App Factory ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="NexusOps — Autonomous AI-SRE",
    description=(
        "Multi-agent system for autonomous infrastructure monitoring, "
        "anomaly detection, and remediation."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response Models ────────────────────────────────────────────────────

class IngestionResponse(BaseModel):
    status: str
    pod_name: str
    anomaly_detected: bool
    anomaly_score: Optional[float] = None
    incident_id: Optional[str] = None


class ApprovalRequest(BaseModel):
    approved: bool
    approver: str = "human"
    reason: Optional[str] = None


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/api/v1/health", tags=["System"])
async def health():
    """Kubernetes liveness/readiness probe."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sentinel_loaded": app.state.sentinel is not None,
        "active_incidents": len(incidents),
        "metric_buffers": len(metric_buffer),
    }


@app.get("/metrics", response_class=PlainTextResponse, tags=["System"])
async def prometheus_metrics():
    """Prometheus scrape endpoint."""
    return PlainTextResponse(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# ── Metric Ingestion ───────────────────────────────────────────────────────────

@app.post("/api/v1/metrics/ingest", response_model=IngestionResponse, tags=["Metrics"])
async def ingest_metric(
    metric: MetricPoint,
    background_tasks: BackgroundTasks,
):
    """
    Ingest a single metric point from a pod.

    The Sentinel Agent runs anomaly detection on every ingestion.
    If an anomaly is detected above threshold, an incident is created
    and the full LangGraph agent pipeline is triggered in the background.
    """
    import time
    start = time.monotonic()

    METRICS_INGESTED.labels(namespace=metric.namespace).inc()

    # Buffer the metric
    key = metric.pod_name
    if key not in metric_buffer:
        metric_buffer[key] = deque(maxlen=100)
    metric_buffer[key].append(metric)

    # Run Sentinel detection if model is loaded
    anomaly_detected = False
    anomaly_score = None
    incident_id = None

    if app.state.sentinel and len(metric_buffer[key]) >= 30:
        window = list(metric_buffer[key])[-30:]  # last 30 points = seq_len
        anomaly_score = app.state.sentinel.score(window)
        anomaly_detected = anomaly_score > settings.SENTINEL_ANOMALY_THRESHOLD

        if anomaly_detected:
            incident_id = str(uuid.uuid4())[:8]
            alert = AnomalyAlert(
                alert_id=incident_id,
                pod_name=metric.pod_name,
                namespace=metric.namespace,
                severity=_score_to_severity(anomaly_score),
                anomaly_score=anomaly_score,
                triggered_metrics=_find_triggered_metrics(metric),
                raw_values={
                    "cpu": metric.cpu_usage_percent,
                    "memory": metric.memory_usage_percent,
                    "latency_ms": metric.request_latency_ms,
                    "error_rate": metric.error_rate_percent,
                },
            )
            incident = Incident(
                incident_id=incident_id,
                severity=alert.severity,
                alert=alert,
            )
            incidents[incident_id] = incident
            INCIDENTS_CREATED.inc()

            # Trigger the full agent pipeline in the background
            background_tasks.add_task(
                _run_agent_pipeline,
                incident_id=incident_id,
            )

    INGEST_LATENCY.observe(time.monotonic() - start)

    return IngestionResponse(
        status="ok",
        pod_name=metric.pod_name,
        anomaly_detected=anomaly_detected,
        anomaly_score=anomaly_score,
        incident_id=incident_id,
    )


async def _run_agent_pipeline(incident_id: str) -> None:
    """
    Background task: run the LangGraph agent pipeline.
    Runs Detective + Oracle, then pauses before Surgeon for human approval.
    The incident state is checkpointed to SQLite after the pause.
    """
    incident = incidents.get(incident_id)
    if not incident:
        return

    print(f"\U0001f6a8 [Incident {incident_id}] Starting LangGraph pipeline...")
    incident.status = IncidentStatus.INVESTIGATING

    try:
        from supervisor.graph import run_graph
        from supervisor.state import IncidentState

        initial_state: IncidentState = {
            "incident_id": incident_id,
            "severity": incident.severity.value,
            "pod_name": incident.alert.pod_name,
            "namespace": incident.alert.namespace,
            "anomaly_score": incident.alert.anomaly_score,
            "triggered_metrics": incident.alert.triggered_metrics,
            "raw_metric_values": incident.alert.raw_values,
            "alert_timestamp": incident.alert.timestamp.isoformat(),
        }

        graph_state = await run_graph(initial_state)

        # Persist graph outputs back to the in-memory incident
        incident.root_cause = graph_state.get("root_cause")
        incident.blast_radius = graph_state.get("blast_radius")
        incident.proposed_actions = graph_state.get("proposed_actions", [])
        incident.status = IncidentStatus.AWAITING_APPROVAL

        print(f"  [Incident {incident_id}] Paused — awaiting human approval at /api/v1/incidents/{incident_id}/approve")

    except Exception as e:
        print(f"  [Incident {incident_id}] Pipeline error: {e}")
        incident.status = IncidentStatus.OPEN


# ── Alert Webhook ──────────────────────────────────────────────────────────────

@app.post("/api/v1/alerts/webhook", tags=["Alerts"])
async def receive_alert_webhook(
    payload: dict[str, Any],
    background_tasks: BackgroundTasks,
):
    """
    Receive alert webhooks from Prometheus Alertmanager or AWS CloudWatch.

    Normalizes the payload into our internal AnomalyAlert format.
    """
    ALERTS_RECEIVED.labels(severity=payload.get("severity", "unknown")).inc()

    # Normalize alert payload (Prometheus format)
    alerts = payload.get("alerts", [payload])
    created_incidents = []

    for alert_data in alerts:
        labels = alert_data.get("labels", {})
        incident_id = str(uuid.uuid4())[:8]

        alert = AnomalyAlert(
            alert_id=incident_id,
            pod_name=labels.get("pod", "unknown"),
            namespace=labels.get("namespace", "default"),
            severity=Severity(labels.get("severity", "medium")),
            anomaly_score=0.9,  # external alerts are assumed high confidence
            triggered_metrics=[labels.get("alertname", "unknown")],
            raw_values={},
        )
        incident = Incident(
            incident_id=incident_id,
            severity=alert.severity,
            alert=alert,
        )
        incidents[incident_id] = incident
        INCIDENTS_CREATED.inc()
        background_tasks.add_task(_run_agent_pipeline, incident_id=incident_id)
        created_incidents.append(incident_id)

    return {"status": "ok", "incidents_created": created_incidents}


# ── Incident Management ────────────────────────────────────────────────────────

@app.get("/api/v1/incidents", tags=["Incidents"])
async def list_incidents(status: Optional[str] = None):
    """List all incidents, optionally filtered by status."""
    result = list(incidents.values())
    if status:
        result = [i for i in result if i.status.value == status]
    return {
        "total": len(result),
        "incidents": [i.model_dump() for i in result],
    }


@app.get("/api/v1/incidents/{incident_id}", tags=["Incidents"])
async def get_incident(incident_id: str):
    """Get full details of a specific incident."""
    incident = incidents.get(incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")
    return incident.model_dump()


@app.post("/api/v1/incidents/{incident_id}/approve", tags=["Incidents"])
async def approve_incident(
    incident_id: str,
    approval: ApprovalRequest,
    background_tasks: BackgroundTasks,
):
    """
    Human-in-the-loop approval for proposed remediation actions.
    This is the key safety gate before the Surgeon Agent executes anything.
    """
    incident = incidents.get(incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")

    if incident.status != IncidentStatus.AWAITING_APPROVAL:
        raise HTTPException(
            status_code=400,
            detail=f"Incident is in status '{incident.status}', not awaiting approval",
        )

    if approval.approved:
        incident.approved_actions = incident.proposed_actions.copy()
        incident.status = IncidentStatus.REMEDIATING
        background_tasks.add_task(_execute_approved_actions, incident_id=incident_id)
        return {"status": "approved", "incident_id": incident_id, "message": "Remediation started"}
    else:
        incident.status = IncidentStatus.CLOSED
        return {"status": "rejected", "incident_id": incident_id, "reason": approval.reason}


async def _execute_approved_actions(incident_id: str) -> None:
    """
    Resume the paused LangGraph graph after human approval.
    Continues from the Surgeon node through to Scribe.
    """
    incident = incidents.get(incident_id)
    if not incident:
        return
    try:
        from supervisor.graph import resume_graph
        graph_state = await resume_graph(
            incident_id=incident_id,
            approved=True,
            approver=incident.approved_actions[0].get("approver", "human") if incident.approved_actions else "human",
        )
        incident.executed_actions = graph_state.get("executed_actions", [])
        incident.incident_report = graph_state.get("incident_report")
        incident.status = IncidentStatus.RESOLVED
        print(f"  [Incident {incident_id}] Resolved. Report generated. Stored in Qdrant: {graph_state.get('report_stored_in_qdrant', False)}")
    except Exception as e:
        print(f"  [Incident {incident_id}] Resume error: {e}")
        incident.status = IncidentStatus.OPEN


# ── Metric Buffer Inspection ───────────────────────────────────────────────────

@app.get("/api/v1/metrics/buffer", tags=["Metrics"])
async def inspect_buffer():
    """Inspect the current in-memory metric buffer (debug endpoint)."""
    return {
        pod: {
            "count": len(buf),
            "latest": buf[-1].model_dump() if buf else None,
        }
        for pod, buf in metric_buffer.items()
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _score_to_severity(score: float) -> Severity:
    if score > 0.95:
        return Severity.CRITICAL
    elif score > 0.90:
        return Severity.HIGH
    elif score > 0.85:
        return Severity.MEDIUM
    return Severity.LOW


def _find_triggered_metrics(metric: MetricPoint) -> list[str]:
    """Identify which individual metrics are outside normal ranges."""
    triggered = []
    if metric.cpu_usage_percent > 80:
        triggered.append("cpu_usage_percent")
    if metric.memory_usage_percent > 85:
        triggered.append("memory_usage_percent")
    if metric.request_latency_ms > 1000:
        triggered.append("request_latency_ms")
    if metric.error_rate_percent > 5:
        triggered.append("error_rate_percent")
    return triggered or ["unknown"]
