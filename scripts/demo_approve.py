"""Demo step 4 — approve the pending incident and wait for Scribe to resolve it."""
import time
import requests

BASE = "http://localhost:8000"

incidents = requests.get(f"{BASE}/api/v1/incidents").json()["incidents"]
inc = next((i for i in incidents if i["status"] == "awaiting_approval"), None)

if not inc:
    print("No incident awaiting approval.")
    raise SystemExit(1)

iid = inc["incident_id"]

print("=" * 55)
print(f"  APPROVING INCIDENT {iid}")
print(f"  Approver : shandilya")
print(f"  Actions  : 1 (rollback) + 2 (scale-out)")
print("=" * 55)

requests.post(f"{BASE}/api/v1/incidents/{iid}/approve", json={
    "approved": True,
    "approver": "shandilya",
    "approved_action_indices": [0, 1],
})

print()
print("  Surgeon executing approved actions...")
print("  Scribe generating post-mortem report...")

# Poll every 1s (faster than before)
for _ in range(45):
    time.sleep(1)
    result = requests.get(f"{BASE}/api/v1/incidents/{iid}").json()
    if result.get("status") == "resolved":
        report = result.get("incident_report", "")
        print()
        print("  STATUS: RESOLVED")
        print()
        print("  POST-MORTEM (Scribe Agent)")
        # Print first 8 lines only — keeps the GIF tight
        for line in report.splitlines()[:8]:
            print(f"  {line}")
        print()
        print("  Stored in Qdrant for future RAG lookups.")
        print("=" * 55)
        break
