import type { NextRequest } from "next/server";
import { backendFetch } from "@/lib/backend";

export async function POST(req: NextRequest) {
  try {
    const body = await req.text();
    const res = await backendFetch("/api/v1/metrics/ingest", {
      method: "POST",
      body,
    });
    const data = await res.json();
    return Response.json(data, { status: res.status });
  } catch {
    return Response.json({ error: "Backend unavailable" }, { status: 503 });
  }
}
