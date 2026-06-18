/**
 * Client-side fetch helpers that call the Next.js API proxy routes.
 * These never call the FastAPI backend directly — that's the proxy's job.
 */

import type { Incident, HealthResponse, IngestResponse, MetricPayload } from "./types";
import { demo, isDemoMode } from "./demo-engine";

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
    isDemoMode() ? Promise.resolve(demo.health()) : apiFetch("/health"),

  incidents: {
    list: (): Promise<Incident[]> =>
      isDemoMode() ? Promise.resolve(demo.list()) : apiFetch("/incidents"),

    get: (id: string): Promise<Incident> => {
      if (isDemoMode()) {
        const inc = demo.get(id);
        return inc
          ? Promise.resolve(inc)
          : Promise.reject(new Error(`Incident ${id} not found`));
      }
      return apiFetch(`/incidents/${id}`);
    },

    approve: (
      id: string,
      opts?: { approved?: boolean; approver?: string; reason?: string }
    ): Promise<{ status: string; incident_id: string }> => {
      if (isDemoMode()) {
        return Promise.resolve(demo.approve(id, opts?.approved ?? true));
      }
      return apiFetch(`/incidents/${id}/approve`, {
        method: "POST",
        body: JSON.stringify({
          approved: opts?.approved ?? true,
          approver: opts?.approver ?? "human",
          ...(opts?.reason ? { reason: opts.reason } : {}),
        }),
      });
    },
  },

  metrics: {
    ingest: (payload: MetricPayload): Promise<IngestResponse> => {
      if (isDemoMode()) {
        demo.trigger();
        return Promise.resolve({
          incident_id: null,
          anomaly_score: 0,
          triggered: true,
        });
      }
      return apiFetch("/metrics/ingest", {
        method: "POST",
        body: JSON.stringify(payload),
      });
    },
  },
};

/** Spawn a demo incident immediately (used by the "Trigger incident" button). */
export function triggerDemoIncident(): void {
  demo.trigger();
}
