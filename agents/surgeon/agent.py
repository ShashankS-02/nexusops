"""
Surgeon Agent — Human-Approved Remediation Executor
====================================================

This agent is the ONLY one that makes real changes to infrastructure.
It NEVER executes without explicit human approval (HITL).

In Phase 2:  All commands run in DRY-RUN mode (--dry-run=client flag).
             This means kubectl validates the command but doesn't apply it.

In Phase 3:  Real execution once you're confident in the model quality.

The LangGraph graph pauses BEFORE this node (interrupt_before=["surgeon"]).
The API's /approve endpoint resumes the graph with the human's decision.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime

from supervisor.state import IncidentState


def _run_dry_run(command: str) -> dict:
    """
    Execute a kubectl command in dry-run mode.
    Validates the command without making any real changes.
    Returns result dict with output and success flag.
    """
    dry_run_cmd = command.strip()
    # Inject --dry-run=client if it's a kubectl command and not already dry-run
    if dry_run_cmd.startswith("kubectl") and "--dry-run" not in dry_run_cmd:
        dry_run_cmd += " --dry-run=client"

    try:
        result = subprocess.run(
            dry_run_cmd.split(),
            capture_output=True,
            text=True,
            timeout=10,
        )
        return {
            "command": dry_run_cmd,
            "success": result.returncode == 0,
            "output": result.stdout or result.stderr,
            "dry_run": True,
            "executed_at": datetime.now(UTC).isoformat(),
        }
    except FileNotFoundError:
        # kubectl not installed locally — simulate success for demo
        return {
            "command": dry_run_cmd,
            "success": True,
            "output": "[SIMULATED] kubectl not found locally — would execute in cluster",
            "dry_run": True,
            "simulated": True,
            "executed_at": datetime.now(UTC).isoformat(),
        }
    except Exception as e:
        return {
            "command": dry_run_cmd,
            "success": False,
            "output": str(e),
            "dry_run": True,
            "executed_at": datetime.now(UTC).isoformat(),
        }


def surgeon_node(state: IncidentState) -> dict:
    """
    LangGraph node: execute approved remediation actions.

    This node runs AFTER the human approval interrupt.
    The graph resumes here with state["human_approved"] already set.
    """
    if not state.get("human_approved", False):
        # Human rejected — skip execution, record reason
        return {
            "executed_actions": [],
            "error": f"Remediation rejected by human. Reason: {state.get('rejection_reason', 'No reason given')}",
        }

    proposed = state.get("proposed_actions", [])
    if not proposed:
        return {"executed_actions": [], "error": "No proposed actions to execute"}

    executed = []
    for action in proposed:
        command = action.get("command", "")
        if not command:
            continue
        result = _run_dry_run(command)
        executed.append(
            {
                **action,
                "execution_result": result,
            }
        )
        # Stop on failure — don't chain risky actions if one fails
        if not result["success"] and not result.get("simulated"):
            break

    return {"executed_actions": executed}
