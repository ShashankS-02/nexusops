"""
LangGraph Shared State — IncidentState
=======================================

This TypedDict is the single object that ALL agents read from and write to
as they process an incident through the graph.

Think of it as a baton being passed between runners in a relay race.
Each agent adds its findings before passing it to the next.
"""
from __future__ import annotations
from typing import Optional, Any
from typing_extensions import TypedDict


class IncidentState(TypedDict, total=False):
    # ── Core incident identity ─────────────────────────────────────────────
    incident_id: str
    severity: str
    pod_name: str
    namespace: str

    # ── Raw signals (from Sentinel, set before graph starts) ───────────────
    anomaly_score: float
    triggered_metrics: list[str]
    raw_metric_values: dict[str, float]
    alert_timestamp: str

    # ── Detective outputs ──────────────────────────────────────────────────
    root_cause: Optional[str]
    root_cause_confidence: Optional[float]
    similar_past_incidents: list[dict]   # from Qdrant RAG search

    # ── Oracle outputs ─────────────────────────────────────────────────────
    blast_radius: Optional[str]
    predicted_failure_time_minutes: Optional[int]
    proposed_actions: list[dict]         # what Oracle recommends doing

    # ── Human approval ─────────────────────────────────────────────────────
    human_approved: Optional[bool]       # set after /approve endpoint is called
    approver: Optional[str]
    rejection_reason: Optional[str]

    # ── Surgeon outputs ────────────────────────────────────────────────────
    executed_actions: list[dict]         # what was actually run (dry-run in Phase 2)

    # ── Scribe outputs ─────────────────────────────────────────────────────
    incident_report: Optional[str]       # final Markdown report
    report_stored_in_qdrant: bool        # confirmation it's in vector DB

    # ── Graph internal ─────────────────────────────────────────────────────
    error: Optional[str]                 # any agent error is captured here
