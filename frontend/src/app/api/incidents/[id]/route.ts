import type { NextRequest } from "next/server";
import { backendFetch } from "@/lib/backend";
import { adaptIncident } from "@/lib/adapters";

export async function GET(
  _req: NextRequest,
  ctx: RouteContext<"/api/incidents/[id]">
) {
  const { id } = await ctx.params;
  try {
    const res = await backendFetch(`/api/v1/incidents/${id}`);
    const raw = await res.json();
    const incident = adaptIncident(raw as Record<string, unknown>);
    return Response.json(incident, { status: res.status });
  } catch {
    return Response.json({ error: "Backend unavailable" }, { status: 503 });
  }
}
