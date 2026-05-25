"""
Oracle Agent — Blast Radius Prediction & Remediation Planning
==============================================================

Given the Detective's root cause, the Oracle:
1. Predicts which other services could be affected (blast radius)
2. Estimates time to complete failure if nothing is done
3. Proposes specific remediation actions with risk levels

Cost: ~1 LLM call per incident (~600-800 tokens). At GPT-4o-mini: <$0.001
"""

from __future__ import annotations

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from nexusops.llm import get_llm
from supervisor.state import IncidentState

ORACLE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are the Oracle Agent for NexusOps, an AI Site Reliability Engineer.
Your job: given a confirmed root cause, predict the blast radius and propose remediation actions.

Respond with JSON:
{{
  "blast_radius": "description of which services/users are/will be impacted",
  "predicted_failure_time_minutes": integer (estimated minutes until complete failure, -1 if already failed),
  "proposed_actions": [
    {{
      "action_type": "restart_pod|scale_deployment|rollback|increase_resources|circuit_break",
      "target": "service-name/namespace",
      "command": "kubectl command to run",
      "risk_level": "low|medium|high",
      "rationale": "why this action"
    }}
  ]
}}
Propose 1-3 actions ordered by recommendation (best first).""",
        ),
        (
            "human",
            """INCIDENT CONTEXT:
- Pod: {pod_name} (namespace: {namespace})
- Severity: {severity}
- Root Cause: {root_cause} (confidence: {confidence:.0%})
- Triggered Metrics: {triggered_metrics}
- Raw Values: {raw_values}

Predict blast radius and propose remediation. Return JSON.""",
        ),
    ]
)


def oracle_node(state: IncidentState) -> dict:
    """LangGraph node: blast radius + remediation planning."""
    try:
        llm = get_llm(temperature=0.0)
        chain = ORACLE_PROMPT | llm | JsonOutputParser()

        result = chain.invoke(
            {
                "pod_name": state.get("pod_name", "unknown"),
                "namespace": state.get("namespace", "unknown"),
                "severity": state.get("severity", "unknown"),
                "root_cause": state.get("root_cause", "Unknown"),
                "confidence": state.get("root_cause_confidence", 0.5),
                "triggered_metrics": state.get("triggered_metrics", []),
                "raw_values": state.get("raw_metric_values", {}),
            }
        )

        return {
            "blast_radius": result.get("blast_radius", "Unknown blast radius"),
            "predicted_failure_time_minutes": int(result.get("predicted_failure_time_minutes", -1)),
            "proposed_actions": result.get("proposed_actions", []),
        }

    except Exception as e:
        return {
            "blast_radius": "Unable to predict",
            "predicted_failure_time_minutes": -1,
            "proposed_actions": [],
            "error": str(e),
        }
