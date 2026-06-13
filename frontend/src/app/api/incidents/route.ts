import { backendFetch } from "@/lib/backend";
import { adaptIncidentList } from "@/lib/adapters";

export async function GET() {
  try {
    const res = await backendFetch("/api/v1/incidents");
    const raw = await res.json();
    const incidents = adaptIncidentList(raw);
    return Response.json(incidents, { status: res.status });
  } catch {
    return Response.json([], { status: 200 });
  }
}
