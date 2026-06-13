/**
 * Adapters that normalize the FastAPI backend's response shape into the
 * frontend's Incident type. Backend schema lives in nexusops/models.py and
 * has different field names + nested structure than the UI expects.
 */

import type { Incident, IncidentStatus } from "./types";

/**
 * The FastAPI backend emits naive UTC timestamps with no timezone suffix
 * (e.g. "2026-06-14T10:31:56.275137"). Browsers parse those as LOCAL time,
 * which skews every "x minutes ago" label by the user's UTC offset. Append a
 * "Z" when no timezone is present so they're correctly interpreted as UTC.
 */
function normalizeTimestamp(value: unknown): string {
  if (typeof value !== "string" || !value) return new Date().toISOString();
  // Already has a timezone designator (Z, +HH:MM, -HHMM, etc.)?
  if (/([zZ]|[+-]\d{2}:?\d{2})$/.test(value)) return value;
  return `${value}Z`;
}

// Backend status enum → frontend status union
const STATUS_MAP: Record<string, IncidentStatus> = {
  open: "detected",
  investigating: "analyzing",
  awaiting_approval: "awaiting_approval",
  remediating: "executing",
  resolved: "resolved",
  closed: "rejected",
  // Pass-through any value the frontend already understands
  detected: "detected",
  analyzing: "analyzing",
  executing: "executing",
  failed: "failed",
};

/** Normalize one backend incident object into the frontend's Incident shape. */
export function adaptIncident(raw: Record<string, unknown>): Incident {
  const alert = (raw.alert ?? {}) as Record<string, unknown>;

  // proposed_actions: backend → list[dict], frontend → string[]
  const proposedActions = Array.isArray(raw.proposed_actions)
    ? (raw.proposed_actions as unknown[]).map((a) => {
        if (typeof a === "string") return a;
        if (a && typeof a === "object") {
          const obj = a as Record<string, unknown>;
          return (
            (obj.command as string) ||
            (obj.description as string) ||
            (obj.action_type as string) ||
            JSON.stringify(obj)
          );
        }
        return String(a);
      })
    : [];

  // blast_radius: backend → string | null OR string[], frontend → string[].
  // The Oracle agent returns free-text prose (e.g. "X is impacted, with
  // cascading risk to A, B"). Only treat a string as a service LIST when every
  // comma-separated token looks like a short identifier; otherwise keep the
  // whole sentence as a single descriptive entry so we don't fabricate a
  // misleading "N services" count from sentence clauses.
  let blastRadius: string[] = [];
  if (Array.isArray(raw.blast_radius)) {
    blastRadius = (raw.blast_radius as unknown[]).map(String);
  } else if (typeof raw.blast_radius === "string" && raw.blast_radius.trim()) {
    const text = raw.blast_radius.trim();
    const parts = text.split(/[,\n]/).map((s) => s.trim()).filter(Boolean);
    const looksLikeServiceList =
      parts.length > 0 &&
      parts.every((p) => p.length <= 40 && p.split(/\s+/).length <= 3);
    blastRadius = looksLikeServiceList ? parts : [text];
  }

  const status = STATUS_MAP[(raw.status as string) ?? "open"] ?? "detected";

  return {
    id: (raw.incident_id as string) ?? (raw.id as string) ?? "unknown",
    pod:
      (alert.pod_name as string) ??
      (raw.pod as string) ??
      (raw.pod_name as string) ??
      "unknown",
    namespace:
      (alert.namespace as string) ??
      (raw.namespace as string) ??
      "default",
    anomaly_score:
      (alert.anomaly_score as number) ??
      (raw.anomaly_score as number) ??
      0,
    status,
    root_cause: (raw.root_cause as string) ?? null,
    blast_radius: blastRadius,
    proposed_actions: proposedActions,
    report:
      (raw.incident_report as string) ??
      (raw.report as string) ??
      null,
    created_at: normalizeTimestamp(raw.created_at),
    updated_at: normalizeTimestamp(raw.updated_at ?? raw.created_at),
  };
}

/** Normalize the backend's list response. Accepts either {incidents:[...]} or a bare array. */
export function adaptIncidentList(raw: unknown): Incident[] {
  if (Array.isArray(raw)) {
    return raw.map((r) => adaptIncident(r as Record<string, unknown>));
  }
  if (raw && typeof raw === "object") {
    const obj = raw as Record<string, unknown>;
    if (Array.isArray(obj.incidents)) {
      return (obj.incidents as Record<string, unknown>[]).map(adaptIncident);
    }
  }
  return [];
}
