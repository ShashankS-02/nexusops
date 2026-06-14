"use client";

import { motion } from "framer-motion";
import {
  Shield,
  Search,
  Brain,
  Wrench,
  FileText,
  ChevronRight,
  Hand,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useApi } from "@/hooks/use-api";
import { api } from "@/lib/api-client";
import type { Incident } from "@/lib/types";

type AgentStatus = "active" | "waiting" | "idle";

interface AgentDef {
  name: string;
  role: string;
  icon: React.ElementType;
  color: string;
  glowColor: string;
}

const AGENT_DEFS: AgentDef[] = [
  { name: "Sentinel", role: "Anomaly Detection", icon: Shield, color: "from-blue-500 to-blue-600", glowColor: "bg-blue-500" },
  { name: "Detective", role: "Root Cause Analysis", icon: Search, color: "from-amber-500 to-amber-600", glowColor: "bg-amber-500" },
  { name: "Oracle", role: "Blast Radius Prediction", icon: Brain, color: "from-purple-500 to-purple-600", glowColor: "bg-purple-500" },
  { name: "HITL Gate", role: "Human Approval", icon: Hand, color: "from-orange-500 to-red-500", glowColor: "bg-orange-500" },
  { name: "Surgeon", role: "Remediation", icon: Wrench, color: "from-emerald-500 to-emerald-600", glowColor: "bg-emerald-500" },
  { name: "Scribe", role: "Post-Mortem Report", icon: FileText, color: "from-cyan-500 to-cyan-600", glowColor: "bg-cyan-500" },
];

// Derive each agent's aggregate status from the live incident set.
function deriveStatuses(incidents: Incident[]): Record<string, AgentStatus> {
  const anyAnalyzing = incidents.some((i) => i.status === "analyzing");
  const anyAwaiting = incidents.some((i) => i.status === "awaiting_approval");
  const anyExecuting = incidents.some((i) => i.status === "executing");
  const recentlyResolved = incidents.some(
    (i) =>
      i.status === "resolved" &&
      Date.now() - new Date(i.updated_at).getTime() < 60_000
  );

  return {
    Sentinel: "active", // always scoring incoming metrics
    Detective: anyAnalyzing ? "active" : "idle",
    Oracle: anyAnalyzing || anyAwaiting ? "active" : "idle",
    "HITL Gate": anyAwaiting ? "waiting" : "idle",
    Surgeon: anyExecuting ? "active" : "idle",
    Scribe: recentlyResolved ? "active" : "idle",
  };
}

const statusStyles: Record<AgentStatus, string> = {
  active: "border-emerald-500/30 shadow-emerald-500/10 shadow-lg",
  waiting: "border-orange-500/30 shadow-orange-500/10 shadow-lg",
  idle: "border-white/8",
};

const statusDot: Record<AgentStatus, string> = {
  active: "bg-emerald-500",
  waiting: "bg-orange-500",
  idle: "bg-white/20",
};

export function AgentPipeline() {
  const { data } = useApi<Incident[]>(() => api.incidents.list(), [], {
    pollInterval: 5_000,
  });
  const statuses = deriveStatuses(data ?? []);

  const activeCount = (data ?? []).filter(
    (i) => i.status !== "resolved" && i.status !== "failed" && i.status !== "rejected"
  ).length;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay: 0.3 }}
      className="rounded-xl border border-white/8 bg-[#0e0f11] p-6"
    >
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-widest">
          Agent Pipeline
        </h3>
        <span className="text-xs text-muted-foreground">
          {activeCount > 0 ? (
            <span className="text-emerald-400">{activeCount} active</span>
          ) : (
            "idle"
          )}
        </span>
      </div>

      <div className="flex items-center gap-2 overflow-x-auto pb-2">
        {AGENT_DEFS.map((agent, i) => {
          const status = statuses[agent.name] ?? "idle";
          return (
            <motion.div
              key={agent.name}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.1 * i + 0.4, duration: 0.4 }}
              className="flex items-center gap-2 shrink-0"
            >
              {/* Agent node */}
              <div
                className={cn(
                  "relative rounded-xl border p-4 min-w-[140px] transition-all",
                  statusStyles[status]
                )}
              >
                {status === "active" && (
                  <motion.div
                    className={cn(
                      "absolute inset-0 rounded-xl opacity-5 blur-xl",
                      agent.glowColor
                    )}
                    animate={{ opacity: [0.03, 0.08, 0.03] }}
                    transition={{ duration: 2, repeat: Infinity }}
                  />
                )}

                <div className="relative z-10">
                  <div className="flex items-center justify-between mb-2">
                    <div
                      className={cn(
                        "w-8 h-8 rounded-lg bg-gradient-to-br flex items-center justify-center",
                        agent.color
                      )}
                    >
                      <agent.icon className="w-4 h-4 text-white" />
                    </div>
                    <motion.div
                      className={cn("w-2 h-2 rounded-full", statusDot[status])}
                      animate={
                        status !== "idle"
                          ? { scale: [1, 1.3, 1], opacity: [1, 0.7, 1] }
                          : {}
                      }
                      transition={{ duration: 1.5, repeat: Infinity }}
                    />
                  </div>
                  <p className="text-sm font-semibold text-white">{agent.name}</p>
                  <p className="text-[11px] text-muted-foreground mt-0.5">
                    {agent.role}
                  </p>
                </div>
              </div>

              {/* Connector arrow */}
              {i < AGENT_DEFS.length - 1 && (
                <ChevronRight
                  className={cn(
                    "w-4 h-4",
                    status === "active" || status === "waiting"
                      ? "text-emerald-500"
                      : "text-white/10"
                  )}
                />
              )}
            </motion.div>
          );
        })}
      </div>
    </motion.div>
  );
}
