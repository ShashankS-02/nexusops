"""Demo step 1 — verify the API and Sentinel model are ready."""
import requests

r = requests.get("http://localhost:8000/api/v1/health").json()
print("=" * 50)
print("  NexusOps API Health Check")
print("=" * 50)
print(f"  Status          : {r['status'].upper()}")
print(f"  Sentinel loaded : {r['sentinel_loaded']}")
print(f"  Active incidents: {r['active_incidents']}")
print("=" * 50)
