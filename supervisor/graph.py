"""
NexusOps Supervisor — LangGraph StateGraph
==========================================

This is the orchestration brain. It defines the flow:
  START → detective → oracle → [PAUSE for human] → surgeon → scribe → END

Key LangGraph concepts used here:
  - StateGraph:      graph whose nodes share one state object (IncidentState)
  - interrupt_before: graph pauses BEFORE surgeon, saves checkpoint to SQLite
  - AsyncSqliteSaver: free local checkpointer (no Redis needed in Phase 2)
  - Command(resume=): how the /approve API endpoint resumes the paused graph

Thread IDs:
  Each incident_id is used as the thread_id for the checkpointer.
  This means each incident's graph state is independently stored and resumable.
"""
from __future__ import annotations

import asyncio
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from supervisor.state import IncidentState
from agents.detective.agent import detective_node
from agents.oracle.agent import oracle_node
from agents.surgeon.agent import surgeon_node
from agents.scribe.agent import scribe_node
from nexusops.config import settings

# ── Build the graph ────────────────────────────────────────────────────────────

def _build_graph():
    builder = StateGraph(IncidentState)

    # Register all agent nodes
    builder.add_node("detective", detective_node)
    builder.add_node("oracle", oracle_node)
    builder.add_node("surgeon", surgeon_node)
    builder.add_node("scribe", scribe_node)

    # Define the linear flow
    builder.add_edge(START, "detective")
    builder.add_edge("detective", "oracle")
    builder.add_edge("oracle", "surgeon")
    builder.add_edge("surgeon", "scribe")
    builder.add_edge("scribe", END)

    return builder


# ── Graph runner ───────────────────────────────────────────────────────────────

async def run_graph(initial_state: IncidentState) -> IncidentState:
    """
    Run the full agent graph for a new incident.

    The graph will run Detective + Oracle, then PAUSE before Surgeon.
    The function returns after the pause — state is checkpointed to SQLite.

    To resume after human approval, call resume_graph().
    """
    builder = _build_graph()

    async with AsyncSqliteSaver.from_conn_string(settings.CHECKPOINT_DB_PATH) as checkpointer:
        graph = builder.compile(
            checkpointer=checkpointer,
            interrupt_before=["surgeon"],  # ← this is the HITL pause point
        )

        config = {"configurable": {"thread_id": initial_state["incident_id"]}}

        # Stream events for observability (LangSmith will trace these automatically)
        final_state = initial_state
        async for event in graph.astream(initial_state, config=config):
            node_name = list(event.keys())[0] if event else "unknown"
            node_output = list(event.values())[0] if event else {}
            print(f"  [Graph] ✓ Node '{node_name}' complete")
            if isinstance(node_output, dict):
                final_state = {**final_state, **node_output}

        return final_state


async def resume_graph(incident_id: str, approved: bool, approver: str = "human", reason: str = "") -> IncidentState:
    """
    Resume a paused graph after human approval/rejection.

    Called by the FastAPI /approve endpoint.
    Updates state with approval decision, then continues surgeon → scribe.
    """
    from langgraph.types import Command

    builder = _build_graph()

    async with AsyncSqliteSaver.from_conn_string(settings.CHECKPOINT_DB_PATH) as checkpointer:
        graph = builder.compile(
            checkpointer=checkpointer,
            interrupt_before=["surgeon"],
        )

        config = {"configurable": {"thread_id": incident_id}}

        # Inject the human's decision into the state, then resume
        approval_update = {
            "human_approved": approved,
            "approver": approver,
            "rejection_reason": reason if not approved else None,
        }

        final_state = {}
        async for event in graph.astream(
            Command(resume=approval_update),
            config=config,
        ):
            node_name = list(event.keys())[0] if event else "unknown"
            node_output = list(event.values())[0] if event else {}
            print(f"  [Graph] ✓ Node '{node_name}' complete (resumed)")
            if isinstance(node_output, dict):
                final_state.update(node_output)

        return final_state


async def get_graph_state(incident_id: str) -> dict:
    """
    Retrieve the current state of a paused graph (for the /incidents/{id} endpoint).
    """
    builder = _build_graph()

    async with AsyncSqliteSaver.from_conn_string(settings.CHECKPOINT_DB_PATH) as checkpointer:
        graph = builder.compile(
            checkpointer=checkpointer,
            interrupt_before=["surgeon"],
        )
        config = {"configurable": {"thread_id": incident_id}}
        snapshot = await graph.aget_state(config)
        return dict(snapshot.values) if snapshot and snapshot.values else {}
