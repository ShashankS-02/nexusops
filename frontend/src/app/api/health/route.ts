import { backendFetch } from "@/lib/backend";
import { adaptIncidentList } from "@/lib/adapters";

export async function GET() {
  try {
    // Pull health + the live incident list in parallel.
    // Per-agent status is derived from real incident statuses since the
    // FastAPI /health endpoint only exposes sentinel_loaded + counts.
    const [healthRes, incidentsRes] = await Promise.all([
      backendFetch("/api/v1/health"),
      backendFetch("/api/v1/incidents"),
    ]);
    const healthRaw = (await healthRes.json()) as Record<string, unknown>;
    const incidentsRaw = await incidentsRes.json();
    const incidents = adaptIncidentList(incidentsRaw);

    const isHealthy =
      healthRaw.status === "healthy" || healthRaw.status === "ok";
    const sentinelLoaded = Boolean(healthRaw.sentinel_loaded);

    // Derive each agent's state from current incident statuses
    const anyAnalyzing = incidents.some((i) => i.status === "analyzing");
    const anyAwaiting = incidents.some((i) => i.status === "awaiting_approval");
    const anyExecuting = incidents.some((i) => i.status === "executing");
    // Scribe writes post-mortems right after resolution — treat any resolution
    // in the last 30s as Scribe being active.
    const recentlyResolved = incidents.some((i) => {
      if (i.status !== "resolved") return false;
      const updated = new Date(i.updated_at).getTime();
      return !Number.isNaN(updated) && Date.now() - updated < 30_000;
    });

    const activeCount = incidents.filter(
      (i) => i.status !== "resolved" && i.status !== "failed" && i.status !== "rejected"
    ).length;

    const normalized = {
      status: isHealthy ? ("ok" as const) : ("degraded" as const),
      pipeline: activeCount > 0 ? ("active" as const) : ("idle" as const),
      agents: {
        sentinel: sentinelLoaded ? "active" : "idle",
        detective: anyAnalyzing ? "active" : "idle",
        oracle: anyAnalyzing || anyAwaiting ? "active" : "idle",
        surgeon: anyExecuting ? "active" : "idle",
        scribe: recentlyResolved ? "active" : "idle",
      },
    };

    return Response.json(normalized, { status: healthRes.status });
  } catch {
    return Response.json(
      {
        status: "degraded",
        pipeline: "idle",
        agents: {
          sentinel: "idle",
          detective: "idle",
          oracle: "idle",
          surgeon: "idle",
          scribe: "idle",
        },
      },
      { status: 200 }
    );
  }
}
