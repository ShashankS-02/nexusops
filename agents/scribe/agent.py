"""
Scribe Agent — Post-Incident Report Generator & Knowledge Storer
================================================================

The Scribe is the LAST agent in the graph. It:
1. Synthesizes everything into a structured Markdown incident report
2. Stores the report + root cause as an embedding in Qdrant

This is how NexusOps gets SMARTER over time: every resolved incident
becomes a searchable entry in the Detective's knowledge base.

Cost: ~1 LLM call per incident (~800-1000 tokens). At GPT-4o-mini: <$0.002
"""

from __future__ import annotations

from datetime import UTC, datetime

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from db.qdrant_store import store_incident_report
from nexusops.llm import get_embedding_model, get_llm
from supervisor.state import IncidentState

SCRIBE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are the Scribe Agent for NexusOps. Write a concise but complete post-incident report in Markdown.

Include these sections:
# Incident Report: {incident_id}
## Summary
## Timeline
## Root Cause
## Impact (Blast Radius)
## Remediation Actions Taken
## Prevention Recommendations

Be factual and technical. Use bullet points. Keep under 400 words.""",
        ),
        (
            "human",
            """Incident Data:
- ID: {incident_id}
- Pod: {pod_name} ({namespace})
- Severity: {severity}
- Anomaly Score: {anomaly_score:.3f}
- Triggered: {triggered_metrics}
- Root Cause: {root_cause} (confidence: {confidence:.0%})
- Blast Radius: {blast_radius}
- Predicted Failure: {failure_minutes} min without intervention
- Actions Executed: {executed_actions}
- Human Approved: {approved}

Write the Markdown report.""",
        ),
    ]
)


def scribe_node(state: IncidentState) -> dict:
    """LangGraph node: generate report and store in Qdrant for future RAG."""
    try:
        llm = get_llm(temperature=0.2)  # slight creativity for better writing
        chain = SCRIBE_PROMPT | llm | StrOutputParser()

        # Format executed actions for the prompt
        actions_text = (
            "\n".join(
                f"  - {a.get('action_type', 'unknown')}: {a.get('command', '')} "
                f"→ {'✓' if a.get('execution_result', {}).get('success') else '✗'}"
                for a in state.get("executed_actions", [])
            )
            or "No actions executed (rejected or no actions proposed)"
        )

        report = chain.invoke(
            {
                "incident_id": state.get("incident_id", "unknown"),
                "pod_name": state.get("pod_name", "unknown"),
                "namespace": state.get("namespace", "unknown"),
                "severity": state.get("severity", "unknown"),
                "anomaly_score": state.get("anomaly_score", 0.0),
                "triggered_metrics": state.get("triggered_metrics", []),
                "root_cause": state.get("root_cause", "Unknown"),
                "confidence": state.get("root_cause_confidence", 0.0),
                "blast_radius": state.get("blast_radius", "Unknown"),
                "failure_minutes": state.get("predicted_failure_time_minutes", -1),
                "executed_actions": actions_text,
                "approved": state.get("human_approved", False),
            }
        )

        # Store in Qdrant for future Detective lookups
        stored = False
        try:
            embedder = get_embedding_model()
            store_incident_report(
                incident_id=state.get("incident_id", "unknown"),
                root_cause=state.get("root_cause", "Unknown"),
                report=report,
                metadata={
                    "severity": state.get("severity"),
                    "pod_name": state.get("pod_name"),
                    "namespace": state.get("namespace"),
                    "resolved_at": datetime.now(UTC).isoformat(),
                },
                embedder=embedder,
            )
            stored = True
        except Exception:
            pass  # Qdrant storage failure is non-fatal

        return {
            "incident_report": report,
            "report_stored_in_qdrant": stored,
        }

    except Exception as e:
        fallback = f"# Incident {state.get('incident_id')}\n\nReport generation failed: {e}"
        return {"incident_report": fallback, "report_stored_in_qdrant": False, "error": str(e)}
