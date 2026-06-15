"use client";

import { motion, AnimatePresence } from "framer-motion";
import {
  AlertTriangle,
  CheckCircle,
  Clock,
  XCircle,
  ChevronRight,
  Filter,
  Search,
  RefreshCw,
  Loader2,
  Ban,
} from "lucide-react";
import { useState, useCallback } from "react";
import { cn } from "@/lib/utils";
import { useApi } from "@/hooks/use-api";
import { api } from "@/lib/api-client";
import type { Incident, IncidentStatus } from "@/lib/types";

// ──────────────────────────────────────────────
// Fallback mock (shown when backend is offline)
// ──────────────────────────────────────────────
const MOCK: Incident[] = [
  {
    id: "15460a2f",
    pod: "payment-service",
    namespace: "prod",
    anomaly_score: 0.784,
    status: "resolved",
    root_cause:
      "High error rate due to misconfiguration in environment variables. The SERVICE_TIMEOUT was set to 0ms causing all downstream calls to fail immediately.",
    blast_radius: ["checkout-service", "billing-worker", "notification-service"],
    proposed_actions: [
      "Patch ConfigMap: set SERVICE_TIMEOUT=5000",
      "Restart pod: payment-service-7f4d9",
    ],
    report:
      "Post-mortem generated. Root cause: misconfigured env var. MTTR: 14 minutes.",
    created_at: new Date(Date.now() - 7_200_000).toISOString(),
    updated_at: new Date(Date.now() - 7_000_000).toISOString(),
  },
  {
    id: "a8c3e1b7",
    pod: "order-service",
    namespace: "prod",
    anomaly_score: 0.692,
    status: "awaiting_approval",
    root_cause:
      "Memory leak in order processing loop. RSS growing at 45MB/min, OOM kill expected in ~20 minutes.",
    blast_radius: ["cart-service", "inventory-service"],
    proposed_actions: [
      "Rolling restart: order-service",
      "Scale replicas: 3→5 to absorb traffic",
    ],
    report: null,
    created_at: new Date(Date.now() - 900_000).toISOString(),
    updated_at: new Date(Date.now() - 800_000).toISOString(),
  },
  {
    id: "d4f72c9e",
    pod: "user-service",
    namespace: "staging",
    anomaly_score: 0.631,
    status: "analyzing",
    root_cause: null,
    blast_radius: [],
    proposed_actions: [],
    report: null,
    created_at: new Date(Date.now() - 120_000).toISOString(),
    updated_at: new Date(Date.now() - 60_000).toISOString(),
  },
  {
    id: "f3a91c44",
    pod: "auth-service",
    namespace: "prod",
    anomaly_score: 0.721,
    status: "resolved",
    root_cause:
      "Redis connection pool exhausted. Max connections (100) reached due to connection leak in token validation path.",
    blast_radius: ["api-gateway", "user-service", "session-service"],
    proposed_actions: [
      "Patch Deployment: set REDIS_MAX_CONN=200",
      "Deploy hotfix: fix connection leak",
    ],
    report:
      "Resolved automatically after Surgeon patched Redis connection limit. Zero downtime.",
    created_at: new Date(Date.now() - 18_000_000).toISOString(),
    updated_at: new Date(Date.now() - 17_000_000).toISOString(),
  },
  {
    id: "b2e67d11",
    pod: "recommendation-engine",
    namespace: "prod",
    anomaly_score: 0.663,
    status: "failed",
    root_cause:
      "GPU OOM during model inference. CUDA out of memory during batch size spike.",
    blast_radius: ["product-page", "search-service"],
    proposed_actions: [
      "Reduce batch size: 512→128",
      "Add HPA for GPU nodes",
    ],
    report:
      "Remediation failed: HPA target not reachable. Manual intervention required.",
    created_at: new Date(Date.now() - 86_400_000).toISOString(),
    updated_at: new Date(Date.now() - 85_000_000).toISOString(),
  },
];

// ──────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────
function timeAgo(iso: string): string {
  const m = Math.floor((Date.now() - new Date(iso).getTime()) / 60_000);
  if (m < 1) return "just now";
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

const statusConfig: Record<
  IncidentStatus,
  {
    icon: React.ElementType;
    label: string;
    color: string;
    bg: string;
    border: string;
    dot: string;
  }
> = {
  detected: {
    icon: AlertTriangle,
    label: "Detected",
    color: "text-yellow-400",
    bg: "bg-yellow-500/10",
    border: "border-yellow-500/20",
    dot: "bg-yellow-500",
  },
  analyzing: {
    icon: AlertTriangle,
    label: "Analyzing",
    color: "text-blue-400",
    bg: "bg-blue-500/10",
    border: "border-blue-500/20",
    dot: "bg-blue-500",
  },
  awaiting_approval: {
    icon: Clock,
    label: "Awaiting Approval",
    color: "text-orange-400",
    bg: "bg-orange-500/10",
    border: "border-orange-500/20",
    dot: "bg-orange-500",
  },
  executing: {
    icon: Loader2,
    label: "Executing",
    color: "text-purple-400",
    bg: "bg-purple-500/10",
    border: "border-purple-500/20",
    dot: "bg-purple-500",
  },
  resolved: {
    icon: CheckCircle,
    label: "Resolved",
    color: "text-emerald-400",
    bg: "bg-emerald-500/10",
    border: "border-emerald-500/20",
    dot: "bg-emerald-500",
  },
  rejected: {
    icon: Ban,
    label: "Dismissed",
    color: "text-zinc-400",
    bg: "bg-zinc-500/10",
    border: "border-zinc-500/20",
    dot: "bg-zinc-500",
  },
  failed: {
    icon: XCircle,
    label: "Failed",
    color: "text-red-400",
    bg: "bg-red-500/10",
    border: "border-red-500/20",
    dot: "bg-red-500",
  },
};

// ──────────────────────────────────────────────
// Score ring
// ──────────────────────────────────────────────
function ScoreRing({ score }: { score: number }) {
  const color = score > 0.7 ? "#ef4444" : score > 0.5 ? "#f59e0b" : "#10b981";
  return (
    <div className="relative w-12 h-12 shrink-0">
      <svg className="w-12 h-12 -rotate-90" viewBox="0 0 36 36">
        <path
          d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
          fill="none"
          stroke="rgba(255,255,255,0.05)"
          strokeWidth="3"
        />
        <motion.path
          d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
          fill="none"
          stroke={color}
          strokeWidth="3"
          strokeLinecap="round"
          initial={{ strokeDasharray: "0, 100" }}
          animate={{ strokeDasharray: `${score * 100}, 100` }}
          transition={{ duration: 1, ease: "easeOut" }}
        />
      </svg>
      <span className="absolute inset-0 flex items-center justify-center text-[10px] font-mono font-bold">
        {score.toFixed(2)}
      </span>
    </div>
  );
}

// ──────────────────────────────────────────────
// Incident row (expandable)
// ──────────────────────────────────────────────
function IncidentRow({
  incident,
  delay,
  onApprove,
}: {
  incident: Incident;
  delay: number;
  onApprove: (id: string) => Promise<void>;
}) {
  const [expanded, setExpanded] = useState(false);
  const [approving, setApproving] = useState(false);
  const status = statusConfig[incident.status];
  const StatusIcon = status.icon;

  const handleApprove = async (e: React.MouseEvent) => {
    e.stopPropagation();
    setApproving(true);
    try {
      await onApprove(incident.id);
    } finally {
      setApproving(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.4 }}
      className="rounded-xl border border-white/8 bg-[#0e0f11] overflow-hidden"
    >
      {/* Row header */}
      <button
        onClick={() => setExpanded((v) => !v)}
        className="w-full flex items-center gap-4 p-4 hover:bg-white/[0.03] transition-colors text-left"
      >
        <ScoreRing score={incident.anomaly_score} />

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-sm font-semibold font-mono text-white">
              {incident.pod}
            </span>
            <span className="text-[10px] text-muted-foreground bg-white/5 px-1.5 py-0.5 rounded font-mono">
              {incident.namespace}
            </span>
            <span className="text-[10px] text-muted-foreground font-mono">
              #{incident.id.slice(0, 8)}
            </span>
          </div>
          <p className="text-xs text-muted-foreground mt-0.5 truncate max-w-md">
            {incident.root_cause ?? "Analyzing…"}
          </p>
        </div>

        <div className="flex items-center gap-3 shrink-0">
          <div
            className={cn(
              "flex items-center gap-1.5 px-2.5 py-1 rounded-full border text-xs font-medium",
              status.bg,
              status.border,
              status.color
            )}
          >
            <motion.div
              className={cn("w-1.5 h-1.5 rounded-full", status.dot)}
              animate={
                incident.status === "analyzing" ||
                incident.status === "executing"
                  ? { opacity: [1, 0.3, 1] }
                  : {}
              }
              transition={{ duration: 1.2, repeat: Infinity }}
            />
            <StatusIcon className="w-3 h-3" />
            {status.label}
          </div>
          <span className="text-xs text-muted-foreground w-16 text-right">
            {timeAgo(incident.created_at)}
          </span>
          <motion.div
            animate={{ rotate: expanded ? 90 : 0 }}
            transition={{ duration: 0.2 }}
          >
            <ChevronRight className="w-4 h-4 text-muted-foreground" />
          </motion.div>
        </div>
      </button>

      {/* Expanded detail */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="overflow-hidden"
          >
            <div className="border-t border-white/8 p-4 grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Root cause */}
              <div>
                <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-widest mb-2">
                  Root Cause
                </p>
                <p className="text-xs text-white/80 leading-relaxed">
                  {incident.root_cause ?? "Analysis in progress…"}
                </p>
              </div>

              {/* Blast radius */}
              <div>
                <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-widest mb-2">
                  Blast Radius
                </p>
                {incident.blast_radius.length > 1 ? (
                  <div className="flex flex-wrap gap-1.5">
                    {incident.blast_radius.map((svc) => (
                      <span
                        key={svc}
                        className="text-[10px] font-mono px-2 py-0.5 rounded bg-red-500/10 border border-red-500/20 text-red-400"
                      >
                        {svc}
                      </span>
                    ))}
                  </div>
                ) : incident.blast_radius.length === 1 ? (
                  <p className="text-xs text-white/80 leading-relaxed">
                    {incident.blast_radius[0]}
                  </p>
                ) : (
                  <span className="text-xs text-muted-foreground">
                    {incident.status === "analyzing" ? "Assessing…" : "None identified"}
                  </span>
                )}
              </div>

              {/* Actions / report */}
              <div>
                <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-widest mb-2">
                  {incident.report ? "Post-Mortem" : "Proposed Actions"}
                </p>
                {incident.report ? (
                  <p className="text-xs text-emerald-400/80 leading-relaxed">
                    {incident.report}
                  </p>
                ) : incident.proposed_actions.length > 0 ? (
                  <div className="space-y-1">
                    {incident.proposed_actions.map((action, i) => (
                      <div
                        key={i}
                        className="text-[10px] font-mono px-2 py-1 rounded bg-white/[0.04] border border-white/8 text-white/70"
                      >
                        $ {action}
                      </div>
                    ))}
                    {incident.status === "awaiting_approval" && (
                      <button
                        onClick={handleApprove}
                        disabled={approving}
                        className="mt-2 w-full text-xs font-semibold py-1.5 rounded-lg bg-signal hover:bg-signal/90 disabled:opacity-50 text-[#0a0c05] transition-colors flex items-center justify-center gap-2"
                      >
                        {approving ? (
                          <>
                            <Loader2 className="w-3 h-3 animate-spin" />
                            Approving…
                          </>
                        ) : (
                          "Approve Remediation"
                        )}
                      </button>
                    )}
                  </div>
                ) : (
                  <span className="text-xs text-muted-foreground">
                    Pending analysis…
                  </span>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

// ──────────────────────────────────────────────
// Skeleton loader
// ──────────────────────────────────────────────
function SkeletonRow() {
  return (
    <div className="rounded-xl border border-white/8 bg-[#0e0f11] p-4 flex items-center gap-4 animate-pulse">
      <div className="w-12 h-12 rounded-full bg-white/5 shrink-0" />
      <div className="flex-1 space-y-2">
        <div className="h-3 w-40 rounded bg-white/5" />
        <div className="h-2.5 w-64 rounded bg-white/5" />
      </div>
      <div className="h-6 w-28 rounded-full bg-white/5" />
    </div>
  );
}

const FILTERS: { label: string; value: IncidentStatus | "all" }[] = [
  { label: "All", value: "all" },
  { label: "Analyzing", value: "analyzing" },
  { label: "Awaiting Approval", value: "awaiting_approval" },
  { label: "Resolved", value: "resolved" },
  { label: "Dismissed", value: "rejected" },
  { label: "Failed", value: "failed" },
];

// ──────────────────────────────────────────────
// Page
// ──────────────────────────────────────────────
export default function IncidentsPage() {
  const [filter, setFilter] = useState<IncidentStatus | "all">("all");
  const [search, setSearch] = useState("");

  const { data, loading, refetch } = useApi<Incident[]>(
    () => api.incidents.list(),
    [],
    { pollInterval: 5_000 }
  );

  const incidents = data ?? MOCK;

  const handleApprove = useCallback(
    async (id: string) => {
      await api.incidents.approve(id);
      refetch();
    },
    [refetch]
  );

  const filtered = incidents.filter((inc) => {
    const matchesFilter = filter === "all" || inc.status === filter;
    const term = search.toLowerCase();
    const matchesSearch =
      !term ||
      inc.pod.toLowerCase().includes(term) ||
      inc.id.includes(term) ||
      inc.namespace.toLowerCase().includes(term);
    return matchesFilter && matchesSearch;
  });

  const awaitingCount = incidents.filter(
    (i) => i.status === "awaiting_approval"
  ).length;

  return (
    <div className="space-y-6 max-w-[1400px] mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="font-display text-2xl font-semibold tracking-tight">Incidents</h2>
          <p className="text-sm text-muted-foreground mt-1">
            {incidents.length} total
            {awaitingCount > 0 && (
              <span className="ml-2 text-orange-400 font-medium">
                · {awaitingCount} awaiting approval
              </span>
            )}
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

      {/* Search + filters */}
      <div className="flex items-center gap-3 flex-wrap">
        <div className="flex items-center gap-2 bg-white/5 rounded-lg px-3 py-1.5 flex-1 min-w-[200px] max-w-sm">
          <Search className="w-4 h-4 text-muted-foreground shrink-0" />
          <input
            type="text"
            placeholder="Search by pod, namespace, ID…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="bg-transparent text-sm text-white placeholder:text-muted-foreground focus:outline-none w-full"
          />
        </div>
        <div className="flex items-center gap-1.5 flex-wrap">
          <Filter className="w-3.5 h-3.5 text-muted-foreground" />
          {FILTERS.map((f) => (
            <button
              key={f.value}
              onClick={() => setFilter(f.value)}
              className={cn(
                "text-xs px-3 py-1 rounded-full border transition-colors",
                filter === f.value
                  ? "bg-signal border-signal text-[#0a0c05]"
                  : "border-white/10 text-muted-foreground hover:text-white hover:bg-white/5"
              )}
            >
              {f.label}
            </button>
          ))}
        </div>
      </div>

      {/* Incident list */}
      <div className="space-y-3">
        {loading ? (
          <>
            <SkeletonRow />
            <SkeletonRow />
            <SkeletonRow />
          </>
        ) : (
          <AnimatePresence>
            {filtered.length === 0 ? (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="text-center py-16 text-muted-foreground text-sm"
              >
                No incidents match your filter.
              </motion.div>
            ) : (
              filtered.map((incident, i) => (
                <IncidentRow
                  key={incident.id}
                  incident={incident}
                  delay={i * 0.04}
                  onApprove={handleApprove}
                />
              ))
            )}
          </AnimatePresence>
        )}
      </div>
    </div>
  );
}
