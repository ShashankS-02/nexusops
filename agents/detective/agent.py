"""
Detective Agent — RAG-Based Root Cause Analysis
================================================

Strategy:
  1. Search Qdrant for past incidents similar to the current alert (semantic RAG)
  2. Feed the alert context + past incidents to the LLM
  3. LLM reasons about the most likely root cause

On first run, Qdrant will be empty — the LLM reasons from the alert alone.
As more incidents are resolved and stored, the RAG gets progressively better.
This is called "progressive knowledge accumulation" — a key production pattern.

Cost: ~1 LLM call per incident (~300-500 tokens). At GPT-4o-mini prices: <$0.001
"""

from __future__ import annotations

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from db.qdrant_store import search_similar_incidents
from nexusops.llm import get_embedding_model, get_llm
from supervisor.state import IncidentState

DETECTIVE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are the Detective Agent for NexusOps, an AI Site Reliability Engineer.
Your job: analyze a Kubernetes infrastructure alert and identify the most likely root cause.

You will be given:
- The alert details (which pod, what metrics spiked, anomaly score)
- Similar past incidents from our knowledge base (may be empty on first runs)

Respond with a JSON object:
{{
  "root_cause": "concise description of the most likely root cause (1-2 sentences)",
  "root_cause_confidence": 0.0-1.0,
  "reasoning": "brief explanation of how you reached this conclusion"
}}""",
        ),
        (
            "human",
            """CURRENT ALERT:
- Pod: {pod_name} (namespace: {namespace})
- Severity: {severity}
- Anomaly Score: {anomaly_score:.3f}
- Triggered Metrics: {triggered_metrics}
- Raw Values: {raw_values}

SIMILAR PAST INCIDENTS (from knowledge base):
{past_incidents}

Analyze and return JSON.""",
        ),
    ]
)


def detective_node(state: IncidentState) -> dict:
    """
    LangGraph node: RAG-based root cause analysis.
    Reads alert fields from state, writes root_cause + confidence back.
    """
    try:
        embedder = get_embedding_model()
        llm = get_llm(temperature=0.0)

        # Build search query from the alert context
        search_query = (
            f"pod {state.get('pod_name')} "
            f"metrics: {', '.join(state.get('triggered_metrics', []))} "
            f"severity: {state.get('severity')}"
        )

        # RAG: search past incidents (gracefully handles empty DB)
        similar = search_similar_incidents(search_query, embedder, top_k=3)
        past_text = (
            "\n\n".join(
                f"[Incident {i + 1}] Score: {r['score']:.2f}\nRoot cause: {r['root_cause']}\nSummary: {r['report'][:300]}..."
                for i, r in enumerate(similar)
            )
            if similar
            else "No past incidents in knowledge base yet."
        )

        # Call the LLM
        chain = DETECTIVE_PROMPT | llm | JsonOutputParser()
        result = chain.invoke(
            {
                "pod_name": state.get("pod_name", "unknown"),
                "namespace": state.get("namespace", "unknown"),
                "severity": state.get("severity", "unknown"),
                "anomaly_score": state.get("anomaly_score", 0.0),
                "triggered_metrics": state.get("triggered_metrics", []),
                "raw_values": state.get("raw_metric_values", {}),
                "past_incidents": past_text,
            }
        )

        return {
            "root_cause": result.get("root_cause", "Unable to determine root cause"),
            "root_cause_confidence": float(result.get("root_cause_confidence", 0.5)),
            "similar_past_incidents": similar,
        }

    except Exception as e:
        return {
            "root_cause": f"Detective agent error: {e}",
            "root_cause_confidence": 0.0,
            "similar_past_incidents": [],
            "error": str(e),
        }
