"""Demo step 3 — show the pending incident: root cause and proposed actions."""
import requests

BASE = "http://localhost:8000"

incidents = requests.get(f"{BASE}/api/v1/incidents").json()["incidents"]
pending = [i for i in incidents if i["status"] == "awaiting_approval"]

if not pending:
    print("No incident awaiting approval.")
    raise SystemExit(1)

inc = pending[0]

print("=" * 55)
print(f"  Incident  : {inc['incident_id']}")
print(f"  Pod       : {inc['alert']['pod_name']} / {inc['alert']['namespace']}")
print(f"  Severity  : {inc['severity'].upper()}")
print(f"  Score     : {inc['alert']['anomaly_score']:.3f}")
print()
print("  ROOT CAUSE (Detective Agent)")
print(f"  {inc['root_cause']}")
print()
print("  BLAST RADIUS")
print(f"  {inc['blast_radius'][:120]}...")
print()
print("  PROPOSED ACTIONS (Surgeon Agent)")
for idx, action in enumerate(inc.get("proposed_actions", []), 1):
    risk = action.get("risk_level", "?").upper()
    cmd = action.get("command", "")
    print(f"  {idx}. [{risk}] {cmd}")
print()
print("  Status: AWAITING HUMAN APPROVAL")
print("=" * 55)
