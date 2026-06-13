import type { NextRequest } from "next/server";
import { backendFetch } from "@/lib/backend";

export async function POST(
  req: NextRequest,
  ctx: RouteContext<"/api/incidents/[id]/approve">
) {
  const { id } = await ctx.params;

  // Read any body the client sent; default to {approved: true, approver: "human"}
  // so the existing one-click "Approve" button works without a payload.
  let payload: Record<string, unknown> = {
    approved: true,
    approver: "human",
  };
  try {
    const body = await req.text();
    if (body) {
      const parsed = JSON.parse(body) as Record<string, unknown>;
      payload = { ...payload, ...parsed };
    }
  } catch {
    // fall through with default payload
  }

  try {
    const res = await backendFetch(`/api/v1/incidents/${id}/approve`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    return Response.json(data, { status: res.status });
  } catch {
    return Response.json({ error: "Backend unavailable" }, { status: 503 });
  }
}
