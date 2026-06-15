"use client";

import { motion } from "framer-motion";
import {
  Shield,
  Search,
  Brain,
  Wrench,
  FileText,
  Hand,
  ChevronRight,
  CheckCircle,
  Loader2,
  Clock,
  AlertCircle,
  XCircle,
  RefreshCw,
} from "lucide-react";
import { useState, useCallback } from "react";
import { cn } from "@/lib/utils";
import { useApi } from "@/hooks/use-api";
import { api } from "@/lib/api-client";
import type { Incident, IncidentStatus } from "@/lib/types";

type StageStatus = "completed" | "active" | "waiting" | "idle" | "failed" | "skipped";

interface Stage {
  name: string;
  role: string;
  icon: React.ElementType;
  gradient: string;
  status: StageStatus;
  detail?: string;
}

// ── Derive the 6 pipeline stages from a real incident's status ──────────────
function buildStages(inc: Incident): Stage[] {
  const s = inc.status;

  // Ordered lifecycle rank so we can mark earlier stages complete
  const rank: Record<IncidentStatus, number> = {
    detected: 0,
    analyzing: 1,
    awaiting_approval: 2,
    executing: 3,
    resolved: 4,
    rejected: 4,
    failed: 4,
  };
  const r = rank[s] ?? 0;

  const sentinel: Stage = {
    name: "Sentinel",
    role: "Anomaly Detection",
    icon: Shield,
    gradient: "from-blue-500 to-blue-600",
    status: "completed",
    detail: `Anomaly score ${inc.anomaly_score.toFixed(3)} crossed threshold.`,
  };

  const detective: Stage = {
    name: "Detective",
    role: "Root Cause Analysis",
    icon: Search,
    gradient: "from-amber-500 to-amber-600",
    status: s === "analyzing" ? "active" : r > 1 ? "completed" : "idle",
    detail:
      inc.root_cause ??
      (s === "analyzing" ? "Running RAG query…" : "Pending analysis."),
  };

  const oracle: Stage = {
    name: "Oracle",
    role: "Blast Radius",
    icon: Brain,
    gradient: "from-purple-500 to-purple-600",
    status: r >= 2 ? "completed" : "idle",
    detail:
      inc.blast_radius.length > 1
        ? `${inc.blast_radius.length} services at risk: ${inc.blast_radius.join(", ")}`
        : inc.blast_radius.length === 1
        ? inc.blast_radius[0]
        : r >= 2
        ? "No downstream impact identified."
        : "Pending root cause.",
  };

  const hitl: Stage = {
    name: "HITL Gate",
    role: "Human Approval",
    icon: Hand,
    gradient: "from-orange-500 to-red-500",
    status:
      s === "awaiting_approval"
        ? "waiting"
        : s === "rejected"
        ? "completed"
        : r >= 3
        ? "completed"
        : "idle",
    detail:
      s === "awaiting_approval"
        ? "Awaiting operator approval."
        : s === "rejected"
        ? "Dismissed by operator."
        : r >= 3
        ? "Approved by operator."
        : "Pending previous stages.",
  };

  const surgeon: Stage = {
    name: "Surgeon",
    role: "Remediation",
    icon: Wrench,
    gradient: "from-emerald-500 to-emerald-600",
    status:
      s === "executing"
        ? "active"
        : s === "resolved"
        ? "completed"
        : s === "failed"
        ? "failed"
        : s === "rejected"
        ? "skipped"
        : "idle",
    detail:
      s === "rejected"
        ? "Skipped — incident dismissed."
        : inc.proposed_actions.length > 0
        ? `${inc.proposed_actions.length} action(s): ${inc.proposed_actions[0]}`
        : "No actions proposed.",
  };

  const scribe: Stage = {
    name: "Scribe",
    role: "Post-Mortem",
    icon: FileText,
    gradient: "from-cyan-500 to-cyan-600",
    status: inc.report
      ? "completed"
      : s === "resolved" || s === "rejected"
      ? "active"
      : "idle",
    detail: inc.report ? "Post-mortem report generated." : "Pending resolution.",
  };

  return [sentinel, detective, oracle, hitl, surgeon, scribe];
}

const statusIcon: Record<StageStatus, React.ElementType> = {
  completed: CheckCircle,
  active: Loader2,
  waiting: Clock,
  idle: AlertCircle,
  failed: XCircle,
  skipped: XCircle,
};

const statusColor: Record<StageStatus, string> = {
  completed: "text-emerald-400",
  active: "text-blue-400",
  waiting: "text-orange-400",
  idle: "text-white/20",
  failed: "text-red-400",
  skipped: "text-white/30",
};

const stageBorder: Record<StageStatus, string> = {
  completed: "border-emerald-500/20",
  active: "border-blue-500/30",
  waiting: "border-orange-500/30",
  idle: "border-white/8",
  failed: "border-red-500/30",
  skipped: "border-white/8",
};

const connectorColor: Record<StageStatus, string> = {
  completed: "text-emerald-500",
  active: "text-blue-400",
  waiting: "text-orange-400",
  idle: "text-white/10",
  failed: "text-red-400",
  skipped: "text-white/10",
};

function timeAgo(iso: string): string {
  const m = Math.floor((Date.now() - new Date(iso).getTime()) / 60_000);
  if (m < 1) return "just now";
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

function PipelineCard({
  inc,
  delay,
  onApprove,
}: {
  inc: Incident;
  delay: number;
  onApprove: (id: string) => Promise<void>;
}) {
  const [approving, setApproving] = useState(false);
  const stages = buildStages(inc);
  const activeIndex = stages.findIndex(
    (st) => st.status === "active" || st.status === "waiting"
  );

  const handleApprove = async () => {
    setApproving(true);
    try {
      await onApprove(inc.id);
    } finally {
      setApproving(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.5 }}
      className="rounded-xl border border-white/8 bg-[#0e0f11] p-6"
    >
      {/* Incident header */}
      <div className="flex items-center gap-3 mb-6">
        <div className="flex-1">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-mono font-semibold text-white">{inc.pod}</span>
            <span className="text-[10px] font-mono text-muted-foreground bg-white/5 px-1.5 py-0.5 rounded">
              {inc.namespace}
            </span>
            <span className="text-[10px] font-mono text-muted-foreground">
              #{inc.id.slice(0, 8)}
            </span>
          </div>
          <p className="text-xs text-muted-foreground mt-0.5">
            Started {timeAgo(inc.created_at)} · Anomaly score{" "}
            <span className="text-amber-400 font-mono">
              {inc.anomaly_score.toFixed(3)}
            </span>
          </p>
        </div>
        {activeIndex >= 0 && (
          <div className="flex items-center gap-1.5 text-xs text-blue-400 bg-blue-500/10 border border-blue-500/20 px-2.5 py-1 rounded-full">
            <motion.div
              className="w-1.5 h-1.5 rounded-full bg-blue-400"
              animate={{ opacity: [1, 0.3, 1] }}
              transition={{ duration: 1, repeat: Infinity }}
            />
            Stage {activeIndex + 1} / {stages.length}
          </div>
        )}
      </div>

      {/* Stages */}
      <div className="flex items-start gap-2 overflow-x-auto pb-2">
        {stages.map((stage, i) => {
          const SIcon = statusIcon[stage.status];
          const Icon = stage.icon;
          return (
            <div key={stage.name} className="flex items-start gap-2 shrink-0">
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: delay + i * 0.08, duration: 0.3 }}
                className={cn(
                  "relative rounded-xl border p-4 w-[150px] transition-all",
                  stageBorder[stage.status],
                  stage.status === "active" && "shadow-lg shadow-blue-500/10",
                  stage.status === "waiting" && "shadow-lg shadow-orange-500/10"
                )}
              >
                {stage.status === "active" && (
                  <motion.div
                    className="absolute inset-0 rounded-xl border border-blue-500/20"
                    animate={{ opacity: [0, 0.6, 0] }}
                    transition={{ duration: 1.5, repeat: Infinity }}
                  />
                )}
                <div className="relative z-10">
                  <div className="flex items-center justify-between mb-2">
                    <div
                      className={cn(
                        "w-8 h-8 rounded-lg bg-gradient-to-br flex items-center justify-center",
                        stage.gradient,
                        (stage.status === "idle" || stage.status === "skipped") &&
                          "opacity-30"
                      )}
                    >
                      <Icon className="w-4 h-4 text-white" />
                    </div>
                    <SIcon
                      className={cn(
                        "w-4 h-4",
                        statusColor[stage.status],
                        stage.status === "active" && "animate-spin"
                      )}
                    />
                  </div>
                  <p
                    className={cn(
                      "text-sm font-semibold",
                      stage.status === "idle" ? "text-white/30" : "text-white"
                    )}
                  >
                    {stage.name}
                  </p>
                  <p
                    className={cn(
                      "text-[10px] mt-0.5",
                      stage.status === "idle"
                        ? "text-white/20"
                        : "text-muted-foreground"
                    )}
                  >
                    {stage.role}
                  </p>
                  {stage.detail && stage.status !== "idle" && (
                    <p className="text-[10px] text-muted-foreground/60 mt-1.5 leading-relaxed line-clamp-3">
                      {stage.detail}
                    </p>
                  )}
                </div>
              </motion.div>

              {i < stages.length - 1 && (
                <div className="flex items-center self-center mt-4">
                  <ChevronRight
                    className={cn("w-4 h-4", connectorColor[stage.status])}
                  />
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* HITL approve CTA */}
      {inc.status === "awaiting_approval" && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: delay + 0.5 }}
          className="mt-4 p-3 rounded-lg border border-orange-500/20 bg-orange-500/5 flex items-center justify-between gap-4"
        >
          <div className="min-w-0">
            <p className="text-xs font-semibold text-orange-400">
              Human-in-the-Loop Gate
            </p>
            <p className="text-[10px] text-muted-foreground mt-0.5">
              Review proposed actions before Surgeon executes remediation.
            </p>
          </div>
          <button
            onClick={handleApprove}
            disabled={approving}
            className="text-xs font-semibold px-4 py-1.5 rounded-lg bg-orange-500 hover:bg-orange-400 disabled:opacity-50 text-white transition-colors shrink-0 flex items-center gap-2"
          >
            {approving ? (
              <>
                <Loader2 className="w-3 h-3 animate-spin" />
                Approving…
              </>
            ) : (
              "Review & Approve"
            )}
          </button>
        </motion.div>
      )}
    </motion.div>
  );
}

function SkeletonCard() {
  return (
    <div className="rounded-xl border border-white/8 bg-[#0e0f11] p-6 animate-pulse">
      <div className="h-4 w-48 bg-white/5 rounded mb-6" />
      <div className="flex gap-2">
        {[0, 1, 2, 3, 4, 5].map((i) => (
          <div key={i} className="w-[150px] h-28 bg-white/5 rounded-xl shrink-0" />
        ))}
      </div>
    </div>
  );
}

export default function PipelinePage() {
  const { data, loading, refetch } = useApi<Incident[]>(
    () => api.incidents.list(),
    [],
    { pollInterval: 4_000 }
  );

  const handleApprove = useCallback(
    async (id: string) => {
      await api.incidents.approve(id);
      refetch();
    },
    [refetch]
  );

  const incidents = data ?? [];

  // Active = anything not yet terminal. Show those first; fall back to recent.
  const active = incidents.filter(
    (i) => i.status !== "resolved" && i.status !== "failed" && i.status !== "rejected"
  );
  const recent = incidents
    .filter((i) => i.status === "resolved" || i.status === "failed")
    .sort(
      (a, b) =>
        new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()
    )
    .slice(0, 3);

  const toShow = active.length > 0 ? active : recent;

  return (
    <div className="space-y-6 max-w-[1400px] mx-auto">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="font-display text-2xl font-semibold tracking-tight">Pipeline</h2>
          <p className="text-sm text-muted-foreground mt-1">
            Live LangGraph runs — Sentinel → Detective → Oracle → HITL → Surgeon → Scribe
          </p>
        </div>
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={refetch}
          className="flex items-center gap-2 text-xs px-3 py-1.5 rounded-lg border border-white/10 hover:bg-white/5 transition-colors text-muted-foreground"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          Refresh
        </motion.button>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-5 text-xs text-muted-foreground flex-wrap">
        {(["completed", "active", "waiting", "failed", "idle"] as StageStatus[]).map(
          (st) => {
            const Icon = statusIcon[st];
            return (
              <div key={st} className="flex items-center gap-1.5">
                <Icon className={cn("w-3.5 h-3.5", statusColor[st])} />
                <span className="capitalize">{st}</span>
              </div>
            );
          }
        )}
      </div>

      {loading ? (
        <>
          <SkeletonCard />
          <SkeletonCard />
        </>
      ) : toShow.length === 0 ? (
        <div className="rounded-xl border border-white/8 bg-[#0e0f11] py-20 text-center">
          <CheckCircle className="w-10 h-10 text-emerald-500/40 mx-auto mb-3" />
          <p className="text-sm text-muted-foreground">
            No active pipeline runs. All systems nominal.
          </p>
          <p className="text-xs text-muted-foreground/60 mt-1">
            Ingest anomalous metrics to trigger a pipeline.
          </p>
        </div>
      ) : (
        <>
          {active.length === 0 && recent.length > 0 && (
            <p className="text-xs text-muted-foreground uppercase tracking-widest">
              Recently completed
            </p>
          )}
          {toShow.map((inc, i) => (
            <PipelineCard
              key={inc.id}
              inc={inc}
              delay={i * 0.1}
              onApprove={handleApprove}
            />
          ))}
        </>
      )}
    </div>
  );
}
