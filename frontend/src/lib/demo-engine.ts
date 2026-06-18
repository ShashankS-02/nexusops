/**
 * Self-contained Demo Mode engine.
 *
 * When there is no FastAPI backend to talk to (e.g. the public Vercel deploy),
 * this in-browser engine simulates the full incident lifecycle so visitors can
 * watch the system work: incidents spawn over time, advance through
 *   detected → analyzing → awaiting_approval → executing → resolved
 * on a wall-clock timeline, the agent pipeline illuminates stage-by-stage, and
 * the HITL "Approve" action pushes an incident through remediation.
 *
 * It exposes the same shape as the real `api.*` client, so the rest of the app
 * is unchanged. Statuses are computed from elapsed time on every read, so the
 * UI just needs to re-poll (the useApi hook polls faster in demo mode).
 */

import type { Incident, IncidentStatus, HealthResponse } from "./types";

/** Demo mode is ON by default so the public deploy works with zero config.
 *  Set NEXT_PUBLIC_DEMO_MODE=0 (or configure NEXT_PUBLIC_BACKEND_URL) to use a
 *  real backend through the /api proxy instead. */
export function isDemoMode(): boolean {
  const flag = process.env.NEXT_PUBLIC_DEMO_MODE;
  if (flag === "1") return true;
  if (flag === "0") return false;
  return !process.env.NEXT_PUBLIC_BACKEND_URL;
}

// ── Lifecycle timing (seconds from spawn) ──────────────────────────────
const T_DETECT = 3; // detected
const T_ANALYZE_END = 10; // analyzing ends → HITL gate
const T_AUTO_APPROVE = 28; // auto-approve 18s after the gate (so auto-play flows)
const T_EXEC = 5; // executing → resolved
const SPAWN_EVERY_MS = 22_000; // auto-play cadence
const MAX_ACTIVE = 4; // don't pile up at the gate
const MAX_KEEP = 16; // cap history

interface Scenario {
  pod: string;
  namespace: string;
  score: number;
  rootCause: string;
  blast: string[];
  actions: string[];
  report: string;
}

const SCENARIOS: Scenario[] = [
  {
    pod: "payment-service",
    namespace: "prod",
    score: 0.83,
    rootCause:
      "Sustained CPU saturation from a runaway request handler; SERVICE_TIMEOUT misconfigured to 0ms causing immediate downstream failures.",
    blast: ["checkout-service", "billing-worker", "notification-svc"],
    actions: [
      "kubectl rollout restart deployment/payment-service -n prod",
      "kubectl patch configmap payment-config -n prod --patch SERVICE_TIMEOUT=5000",
    ],
    report:
      "Resolved. Root cause: misconfigured SERVICE_TIMEOUT. Restarted deployment and patched ConfigMap. MTTR 4m12s.",
  },
  {
    pod: "order-service",
    namespace: "prod",
    score: 0.69,
    rootCause:
      "Memory leak in the order-processing loop. RSS growing ~45MB/min; OOM kill projected within ~20 minutes.",
    blast: ["cart-service", "inventory-service"],
    actions: [
      "kubectl rollout restart deployment/order-service -n prod",
      "kubectl scale deployment/order-service --replicas=5 -n prod",
    ],
    report:
      "Resolved. Rolling restart cleared the leaked heap; scaled to 5 replicas to absorb traffic. MTTR 6m03s.",
  },
  {
    pod: "auth-service",
    namespace: "prod",
    score: 0.72,
    rootCause:
      "Redis connection pool exhausted (max 100) due to a connection leak in the token-validation path.",
    blast: ["api-gateway", "session-service"],
    actions: [
      "kubectl patch deployment/auth-service -n prod --patch REDIS_MAX_CONN=200",
      "kubectl rollout restart deployment/auth-service -n prod",
    ],
    report:
      "Resolved. Raised Redis connection ceiling and restarted; zero downtime. MTTR 8m41s.",
  },
  {
    pod: "api-gateway",
    namespace: "prod",
    score: 0.91,
    rootCause:
      "Upstream gateway failure cascading to downstream services; p99 latency above 9s with a 55% error rate.",
    blast: ["order-service", "user-service", "search-service", "billing-worker"],
    actions: [
      "kubectl rollout undo deployment/api-gateway -n prod",
      "kubectl scale deployment/api-gateway --replicas=6 -n prod",
    ],
    report:
      "Resolved. Rolled back to the last good revision and scaled out; cascade contained. MTTR 11m20s.",
  },
  {
    pod: "recommendation-engine",
    namespace: "prod",
    score: 0.66,
    rootCause:
      "GPU out-of-memory during model inference — CUDA OOM triggered by a batch-size spike.",
    blast: ["product-page", "search-service"],
    actions: [
      "kubectl set env deployment/recommendation-engine BATCH_SIZE=128 -n prod",
      "kubectl rollout restart deployment/recommendation-engine -n prod",
    ],
    report:
      "Resolved. Reduced inference batch size and restarted; GPU memory back within limits. MTTR 7m55s.",
  },
  {
    pod: "user-service",
    namespace: "staging",
    score: 0.63,
    rootCause:
      "Elevated p99 latency after a deploy; cold-start contention on a newly scaled replica set.",
    blast: ["profile-service"],
    actions: [
      "kubectl rollout restart deployment/user-service -n staging",
    ],
    report:
      "Resolved. Warmed the new replicas with a rolling restart; latency normalized. MTTR 3m18s.",
  },
  {
    pod: "data-pipeline",
    namespace: "prod",
    score: 0.74,
    rootCause:
      "Egress network saturation creating backpressure on the streaming data pipeline.",
    blast: ["analytics-worker"],
    actions: [
      "kubectl scale deployment/data-pipeline --replicas=4 -n prod",
    ],
    report:
      "Resolved. Scaled consumers to drain the backlog; throughput recovered. MTTR 9m07s.",
  },
  {
    pod: "notification-svc",
    namespace: "prod",
    score: 0.68,
    rootCause:
      "High error rate with normal CPU/memory — a bad config flag rejecting valid webhook payloads.",
    blast: ["email-worker"],
    actions: [
      "kubectl rollout undo deployment/notification-svc -n prod",
    ],
    report:
      "Resolved. Rolled back the bad config revision; error rate returned to baseline. MTTR 5m44s.",
  },
];

interface DemoIncident {
  id: string;
  scenario: Scenario;
  spawnedAt: number;
  approvedAt: number | null;
  rejected: boolean;
}

const TERMINAL: IncidentStatus[] = ["resolved", "rejected", "failed"];

class DemoEngine {
  private incidents: DemoIncident[] = [];
  private started = false;
  private seq = 0;
  private timer: ReturnType<typeof setInterval> | null = null;

  private newId(): string {
    return (
      Math.random().toString(16).slice(2, 6) +
      Math.random().toString(16).slice(2, 6)
    );
  }

  private push(
    scenario: Scenario,
    spawnedAt: number,
    approvedAt: number | null = null,
    rejected = false
  ) {
    this.incidents.push({
      id: this.newId(),
      scenario,
      spawnedAt,
      approvedAt,
      rejected,
    });
    if (this.incidents.length > MAX_KEEP) {
      this.incidents = this.incidents.slice(-MAX_KEEP);
    }
  }

  /** Lazily seed + start auto-play on first use (client only). */
  start() {
    if (this.started || typeof window === "undefined") return;
    this.started = true;
    const now = Date.now();

    // Seed a varied, non-empty board on first load:
    this.push(SCENARIOS[0], now - 5 * 3600_000); // resolved ~5h ago
    this.push(SCENARIOS[2], now - 2 * 3600_000); // resolved ~2h ago
    this.push(SCENARIOS[1], now - 12_000); // awaiting approval (gate at 10s, auto at 28s)
    this.push(SCENARIOS[5], now - 6_000); // analyzing
    this.seq = 6;

    this.timer = setInterval(() => {
      const active = this.incidents.filter(
        (i) => !TERMINAL.includes(this.statusOf(i, Date.now()).status)
      ).length;
      if (active < MAX_ACTIVE) this.trigger();
    }, SPAWN_EVERY_MS);
  }

  /** Spawn a fresh incident immediately (used by auto-play and the button). */
  trigger(): void {
    this.start();
    this.push(SCENARIOS[this.seq % SCENARIOS.length], Date.now());
    this.seq++;
  }

  private statusOf(
    inc: DemoIncident,
    now: number
  ): { status: IncidentStatus; updatedAt: number } {
    const t = (now - inc.spawnedAt) / 1000;
    if (t < T_DETECT) return { status: "detected", updatedAt: inc.spawnedAt };
    if (t < T_ANALYZE_END)
      return { status: "analyzing", updatedAt: inc.spawnedAt + T_DETECT * 1000 };

    const gateAt = inc.spawnedAt + T_ANALYZE_END * 1000;
    if (inc.rejected) {
      return { status: "rejected", updatedAt: inc.approvedAt ?? gateAt };
    }
    const approvedTime = inc.approvedAt ?? inc.spawnedAt + T_AUTO_APPROVE * 1000;
    if (now < approvedTime)
      return { status: "awaiting_approval", updatedAt: gateAt };

    const execElapsed = (now - approvedTime) / 1000;
    if (execElapsed < T_EXEC)
      return { status: "executing", updatedAt: approvedTime };
    return { status: "resolved", updatedAt: approvedTime + T_EXEC * 1000 };
  }

  private toIncident(inc: DemoIncident, now: number): Incident {
    const { status, updatedAt } = this.statusOf(inc, now);
    const analyzed = status !== "detected";
    const gateReached = !["detected", "analyzing"].includes(status);
    const s = inc.scenario;
    return {
      id: inc.id,
      pod: s.pod,
      namespace: s.namespace,
      anomaly_score: s.score,
      status,
      root_cause: analyzed ? s.rootCause : null,
      blast_radius: gateReached ? s.blast : [],
      proposed_actions: gateReached ? s.actions : [],
      report:
        status === "resolved"
          ? s.report
          : status === "rejected"
          ? "Dismissed by operator — flagged as a false positive; no remediation executed."
          : null,
      created_at: new Date(inc.spawnedAt).toISOString(),
      updated_at: new Date(updatedAt).toISOString(),
    };
  }

  list(): Incident[] {
    this.start();
    const now = Date.now();
    return this.incidents
      .map((i) => this.toIncident(i, now))
      .sort(
        (a, b) =>
          new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
      );
  }

  get(id: string): Incident | null {
    this.start();
    const inc = this.incidents.find((i) => i.id === id);
    return inc ? this.toIncident(inc, Date.now()) : null;
  }

  approve(
    id: string,
    approved = true
  ): { status: string; incident_id: string } {
    const inc = this.incidents.find((i) => i.id === id);
    if (inc && inc.approvedAt == null && !inc.rejected) {
      inc.approvedAt = Date.now();
      if (!approved) inc.rejected = true;
    }
    return { status: approved ? "approved" : "rejected", incident_id: id };
  }

  health(): HealthResponse {
    const incs = this.list();
    const has = (s: IncidentStatus) => incs.some((i) => i.status === s);
    const recentlyResolved = incs.some(
      (i) =>
        i.status === "resolved" &&
        Date.now() - new Date(i.updated_at).getTime() < 60_000
    );
    const active = incs.filter((i) => !TERMINAL.includes(i.status)).length;
    return {
      status: "ok",
      pipeline: active > 0 ? "active" : "idle",
      agents: {
        sentinel: "active",
        detective: has("analyzing") ? "active" : "idle",
        oracle: has("analyzing") || has("awaiting_approval") ? "active" : "idle",
        surgeon: has("executing") ? "active" : "idle",
        scribe: recentlyResolved ? "active" : "idle",
      },
    };
  }
}

export const demo = new DemoEngine();
