"""
NexusOps Shared Data Models (Pydantic)

These schemas are used across:
  - The FastAPI gateway (request/response bodies)
  - The LangGraph agent state
  - The metrics simulator output
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AgentName(str, Enum):
    SUPERVISOR = "supervisor"
    SENTINEL = "sentinel"
    DETECTIVE = "detective"
    ORACLE = "oracle"
    SURGEON = "surgeon"
    SCRIBE = "scribe"


class IncidentStatus(str, Enum):
    OPEN = "open"
    INVESTIGATING = "investigating"
    AWAITING_APPROVAL = "awaiting_approval"
    REMEDIATING = "remediating"
    RESOLVED = "resolved"
    CLOSED = "closed"


# ── Metrics ───────────────────────────────────────────────────────────────────

class MetricPoint(BaseModel):
    """A single time-series metric observation from a pod/node."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    pod_name: str
    namespace: str
    cpu_usage_percent: float = Field(ge=0.0, le=100.0)
    memory_usage_percent: float = Field(ge=0.0, le=100.0)
    request_latency_ms: float = Field(ge=0.0)
    error_rate_percent: float = Field(ge=0.0, le=100.0)
    network_io_kbps: float = Field(ge=0.0)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class MetricWindow(BaseModel):
    """A sliding window of metric points fed into the anomaly model."""
    pod_name: str
    namespace: str
    window_size: int
    points: list[MetricPoint]


# ── Alerts ────────────────────────────────────────────────────────────────────

class AnomalyAlert(BaseModel):
    """Fired by the Sentinel Agent when an anomaly is detected."""
    alert_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    pod_name: str
    namespace: str
    severity: Severity
    anomaly_score: float = Field(ge=0.0, le=1.0, description="Higher = more anomalous")
    triggered_metrics: list[str] = Field(description="Which metrics spiked")
    raw_values: dict[str, float] = Field(description="Actual metric values at alert time")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# ── Log Entries ───────────────────────────────────────────────────────────────

class LogEntry(BaseModel):
    """A single structured log line from a pod."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    pod_name: str
    namespace: str
    level: str  # INFO, WARN, ERROR, CRITICAL
    message: str
    trace_id: Optional[str] = None
    extra: dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# ── Incidents ─────────────────────────────────────────────────────────────────

class Incident(BaseModel):
    """The central object that flows through the LangGraph agent graph."""
    incident_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    status: IncidentStatus = IncidentStatus.OPEN
    severity: Severity

    # Source data
    alert: AnomalyAlert
    recent_logs: list[LogEntry] = Field(default_factory=list)

    # Agent outputs (filled as graph progresses)
    root_cause: Optional[str] = None
    root_cause_confidence: Optional[float] = None
    blast_radius: Optional[str] = None
    predicted_failure_time_minutes: Optional[int] = None
    proposed_actions: list[dict[str, Any]] = Field(default_factory=list)
    approved_actions: list[dict[str, Any]] = Field(default_factory=list)
    executed_actions: list[dict[str, Any]] = Field(default_factory=list)
    incident_report: Optional[str] = None

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# ── Remediation ───────────────────────────────────────────────────────────────

class RemediationAction(BaseModel):
    """A proposed or executed remediation action."""
    action_id: str
    action_type: str  # e.g., "restart_pod", "scale_deployment", "rollback"
    target: str       # e.g., "payment-service/prod"
    command: str      # e.g., "kubectl rollout restart deployment/payment-service -n prod"
    risk_level: Severity
    estimated_downtime_seconds: int = 0
    dry_run: bool = True
    executed: bool = False
    result: Optional[str] = None
