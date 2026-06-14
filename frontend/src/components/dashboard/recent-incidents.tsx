"use client";

import { motion } from "framer-motion";
import { AlertTriangle, CheckCircle, Clock, XCircle, Loader2, Ban } from "lucide-react";
import { useRouter } from "next/navigation";
import { cn } from "@/lib/utils";
import { useApi } from "@/hooks/use-api";
import { api } from "@/lib/api-client";
import type { Incident, IncidentStatus } from "@/lib/types";

// Fallback mock shown when the backend is offline
const MOCK: Incident[] = [
  {
    id: "15460a2f",
    pod: "payment-service",
    namespace: "prod",
    anomaly_score: 0.784,
    status: "resolved",
    root_cause: "High error rate due to misconfiguration",
    blast_radius: [],
    proposed_actions: [],
    report: null,
    created_at: new Date(Date.now() - 7200_000).toISOString(),
    updated_at: new Date(Date.now() - 7000_000).toISOString(),
  },
  {
    id: "a8c3e1b7",
    pod: "order-service",
    namespace: "prod",
    anomaly_score: 0.692,
    status: "awaiting_approval",
    root_cause: "Memory leak causing OOM kills",
    blast_radius: [],
    proposed_actions: [],
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
];

const statusConfig: Record<
  IncidentStatus,
  {
    icon: React.ElementType;
    label: string;
    color: string;
    bg: string;
    border: string;
  }
> = {
  detected: {
    icon: AlertTriangle,
    label: "Detected",
    color: "text-yellow-400",
    bg: "bg-yellow-500/10",
    border: "border-yellow-500/20",
  },
  analyzing: {
    icon: AlertTriangle,
    label: "Analyzing",
    color: "text-blue-400",
    bg: "bg-blue-500/10",
    border: "border-blue-500/20",
  },
  awaiting_approval: {
    icon: Clock,
    label: "Awaiting Approval",
    color: "text-orange-400",
    bg: "bg-orange-500/10",
    border: "border-orange-500/20",
  },
  executing: {
    icon: Loader2,
    label: "Executing",
    color: "text-purple-400",
    bg: "bg-purple-500/10",
    border: "border-purple-500/20",
  },
  resolved: {
    icon: CheckCircle,
    label: "Resolved",
    color: "text-emerald-400",
    bg: "bg-emerald-500/10",
    border: "border-emerald-500/20",
  },
  rejected: {
    icon: Ban,
    label: "Dismissed",
    color: "text-zinc-400",
    bg: "bg-zinc-500/10",
    border: "border-zinc-500/20",
  },
  failed: {
    icon: XCircle,
    label: "Failed",
    color: "text-red-400",
    bg: "bg-red-500/10",
    border: "border-red-500/20",
  },
};

function timeAgo(iso: string): string {
  const diffMs = Date.now() - new Date(iso).getTime();
  const m = Math.floor(diffMs / 60_000);
  if (m < 1) return "just now";
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

function Skeleton() {
  return (
    <div className="space-y-3">
      {[0, 1, 2].map((i) => (
        <div key={i} className="flex items-center gap-4 p-3 rounded-lg">
          <div className="w-12 h-12 rounded-full bg-white/5 animate-pulse shrink-0" />
          <div className="flex-1 space-y-2">
            <div className="h-3 w-32 rounded bg-white/5 animate-pulse" />
            <div className="h-2.5 w-48 rounded bg-white/5 animate-pulse" />
          </div>
          <div className="h-5 w-24 rounded-full bg-white/5 animate-pulse" />
        </div>
      ))}
    </div>
  );
}

export function RecentIncidents() {
  const router = useRouter();
  const { data, loading } = useApi<Incident[]>(
    () => api.incidents.list(),
    [],
    { pollInterval: 5_000 }
  );

  const incidents = (data ?? MOCK).slice(0, 5);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.6 }}
      className="rounded-xl border border-white/8 bg-[#0e0f11] p-6"
    >
      <div className="flex items-center justify-between mb-5">
        <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-widest">
          Recent Incidents
        </h3>
        <button
          onClick={() => router.push("/incidents")}
          className="text-xs text-signal hover:text-signal/80 transition-colors"
        >
          View all
        </button>
      </div>

      {loading ? (
        <Skeleton />
      ) : (
        <div className="space-y-3">
          {incidents.map((incident, i) => {
            const status = statusConfig[incident.status];
            const StatusIcon = status.icon;

            return (
              <motion.div
                key={incident.id}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.7 + i * 0.1 }}
                whileHover={{ x: 4, transition: { duration: 0.15 } }}
                onClick={() => router.push("/incidents")}
                className="flex items-center gap-4 p-3 rounded-lg hover:bg-white/[0.03] cursor-pointer transition-colors group"
              >
                {/* Anomaly score ring */}
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
                      stroke={
                        incident.anomaly_score > 0.7
                          ? "#ef4444"
                          : incident.anomaly_score > 0.5
                          ? "#f59e0b"
                          : "#10b981"
                      }
                      strokeWidth="3"
                      strokeLinecap="round"
                      initial={{ strokeDasharray: "0, 100" }}
                      animate={{
                        strokeDasharray: `${incident.anomaly_score * 100}, 100`,
                      }}
                      transition={{ delay: 0.8 + i * 0.1, duration: 1 }}
                    />
                  </svg>
                  <span className="absolute inset-0 flex items-center justify-center text-[10px] font-mono font-bold">
                    {incident.anomaly_score.toFixed(2)}
                  </span>
                </div>

                {/* Info */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-semibold text-white font-mono">
                      {incident.pod}
                    </span>
                    <span className="text-[10px] text-muted-foreground bg-white/5 px-1.5 py-0.5 rounded">
                      {incident.namespace}
                    </span>
                  </div>
                  <p className="text-xs text-muted-foreground mt-0.5 truncate">
                    {incident.root_cause ?? "Analyzing…"}
                  </p>
                </div>

                {/* Status badge */}
                <div
                  className={cn(
                    "flex items-center gap-1.5 px-2.5 py-1 rounded-full border text-xs font-medium shrink-0",
                    status.bg,
                    status.border,
                    status.color
                  )}
                >
                  <StatusIcon className="w-3 h-3" />
                  {status.label}
                </div>

                {/* Time */}
                <span className="text-xs text-muted-foreground shrink-0">
                  {timeAgo(incident.created_at)}
                </span>
              </motion.div>
            );
          })}
        </div>
      )}
    </motion.div>
  );
}
