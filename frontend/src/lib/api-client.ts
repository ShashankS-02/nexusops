/**
 * Client-side fetch helpers that call the Next.js API proxy routes.
 * These never call the FastAPI backend directly — that's the proxy's job.
 */

import type { Incident, HealthResponse, IngestResponse, MetricPayload } from "./types";

const BASE = "/api";

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`API ${path}: ${res.status} ${text}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: (): Promise<HealthResponse> =>
    apiFetch("/health"),

  incidents: {
    list: (): Promise<Incident[]> =>
      apiFetch("/incidents"),

    get: (id: string): Promise<Incident> =>
      apiFetch(`/incidents/${id}`),

    approve: (
      id: string,
      opts?: { approved?: boolean; approver?: string; reason?: string }
    ): Promise<{ status: string; incident_id: string }> =>
      apiFetch(`/incidents/${id}/approve`, {
        method: "POST",
        body: JSON.stringify({
          approved: opts?.approved ?? true,
          approver: opts?.approver ?? "human",
          ...(opts?.reason ? { reason: opts.reason } : {}),
        }),
      }),
  },

  metrics: {
    ingest: (payload: MetricPayload): Promise<IngestResponse> =>
      apiFetch("/metrics/ingest", {
        method: "POST",
        body: JSON.stringify(payload),
      }),
  },
};
