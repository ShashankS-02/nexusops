// ──────────────────────────────────────────────
// NexusOps API types — mirrors FastAPI schemas
// ──────────────────────────────────────────────

export type IncidentStatus =
  | "detected"
  | "analyzing"
  | "awaiting_approval"
  | "executing"
  | "resolved"
  | "rejected"
  | "failed";

export interface Incident {
  id: string;
  pod: string;
  namespace: string;
  anomaly_score: number;
  status: IncidentStatus;
  root_cause: string | null;
  blast_radius: string[];
  proposed_actions: string[];
  report: string | null;
  created_at: string;
  updated_at: string;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  pipeline: "active" | "idle";
  agents: Record<string, "active" | "idle" | "error">;
}

export interface MetricPayload {
  pod: string;
  namespace: string;
  cpu_usage: number;
  memory_usage: number;
  error_rate: number;
  latency_p99: number;
  timestamp?: string;
}

export interface IngestResponse {
  incident_id: string | null;
  anomaly_score: number;
  triggered: boolean;
}
