"use client";

import { motion } from "framer-motion";
import {
  Shield,
  Search,
  Brain,
  Wrench,
  FileText,
  CheckCircle,
  Zap,
  Clock,
  Activity,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useApi } from "@/hooks/use-api";
import { api } from "@/lib/api-client";
import type { Incident } from "@/lib/types";

type AgentStatus = "active" | "waiting" | "idle";

// ── Static agent definitions (descriptions + model-eval metrics that aren't
//    derivable from the live API) ──────────────────────────────────────────
const AGENT_DEFS = [
  {
    name: "Sentinel",
    role: "Anomaly Detection",
    description:
      "PyTorch LSTM autoencoder that scores incoming metrics for reconstruction error. Triggers an incident when the score exceeds the 0.60 threshold.",
    icon: Shield,
    gradient: "from-blue-500 to-blue-600",
    glowBorder: "border-blue-500/20",
    glowShadow: "shadow-blue-500/10",
    accentBg: "bg-blue-500/10",
    accentText: "text-blue-400",
    // model-eval stats (from training) — not API-derivable
    modelStats: { AUROC: "0.99", Threshold: "0.60" },
  },
  {
    name: "Detective",
    role: "Root Cause Analysis",
    description:
      "RAG-based agent that retrieves relevant runbooks and past incidents from Qdrant vector store, then synthesises a root-cause hypothesis using Claude.",
    icon: Search,
    gradient: "from-amber-500 to-amber-600",
    glowBorder: "border-amber-500/20",
    glowShadow: "shadow-amber-500/10",
    accentBg: "bg-amber-500/10",
    accentText: "text-amber-400",
    modelStats: { "Vector store": "Qdrant", Accuracy: "94%" },
  },
  {
    name: "Oracle",
    role: "Blast Radius Prediction",
    description:
      "Analyses service dependency graphs to predict which downstream services will be affected by an incident, and proposes remediation actions with risk levels.",
    icon: Brain,
    gradient: "from-purple-500 to-purple-600",
    glowBorder: "border-purple-500/20",
    glowShadow: "shadow-purple-500/10",
    accentBg: "bg-purple-500/10",
    accentText: "text-purple-400",
    modelStats: { Accuracy: "91%" },
  },
  {
    name: "Surgeon",
    role: "Automated Remediation",
    description:
      "Executes approved remediation actions against the Kubernetes API — rolling restarts, config patches, scaling, and rollbacks. All actions require HITL approval first.",
    icon: Wrench,
    gradient: "from-emerald-500 to-emerald-600",
    glowBorder: "border-emerald-500/20",
    glowShadow: "shadow-emerald-500/10",
    accentBg: "bg-emerald-500/10",
    accentText: "text-emerald-400",
    modelStats: { Mode: "Dry-run" },
  },
  {
    name: "Scribe",
    role: "Post-Mortem Reports",
    description:
      "Generates structured post-mortem reports in Markdown after each resolved incident, then embeds them into Qdrant so the Detective gets smarter over time.",
    icon: FileText,
    gradient: "from-cyan-500 to-cyan-600",
    glowBorder: "border-cyan-500/20",
    glowShadow: "shadow-cyan-500/10",
    accentBg: "bg-cyan-500/10",
    accentText: "text-cyan-400",
    modelStats: { "Stored in": "Qdrant" },
  },
] as const;

type AgentDef = (typeof AGENT_DEFS)[number];

function timeAgo(iso: string): string {
  const m = Math.floor((Date.now() - new Date(iso).getTime()) / 60_000);
  if (m < 1) return "just now";
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

type ActivityItem = { text: string; time: string; ok: boolean };

// ── Derive live status / stats / activity from the incident set ─────────────
function deriveStatus(name: string, incidents: Incident[]): AgentStatus {
  const anyAnalyzing = incidents.some((i) => i.status === "analyzing");
  const anyAwaiting = incidents.some((i) => i.status === "awaiting_approval");
  const anyExecuting = incidents.some((i) => i.status === "executing");
  const recentlyResolved = incidents.some(
    (i) =>
      i.status === "resolved" &&
      Date.now() - new Date(i.updated_at).getTime() < 60_000
  );
  switch (name) {
    case "Sentinel":
      return "active";
    case "Detective":
      return anyAnalyzing ? "active" : "idle";
    case "Oracle":
      return anyAnalyzing || anyAwaiting ? "active" : "idle";
    case "Surgeon":
      return anyExecuting ? "active" : anyAwaiting ? "waiting" : "idle";
    case "Scribe":
      return recentlyResolved ? "active" : "idle";
    default:
      return "idle";
  }
}

function deriveStats(
  def: AgentDef,
  incidents: Incident[]
): Record<string, string> {
  const base: Record<string, string> = { ...def.modelStats };
  const total = incidents.length;
  const active = incidents.filter(
    (i) => i.status !== "resolved" && i.status !== "failed" && i.status !== "rejected"
  ).length;
  const withRootCause = incidents.filter((i) => i.root_cause).length;
  const withBlast = incidents.filter((i) => i.blast_radius.length > 0).length;
  const awaiting = incidents.filter(
    (i) => i.status === "awaiting_approval"
  ).length;
  const resolved = incidents.filter((i) => i.status === "resolved").length;
  const withReport = incidents.filter((i) => i.report).length;

  switch (def.name) {
    case "Sentinel":
      return { "Incidents Triggered": String(total), Active: String(active), ...base };
    case "Detective":
      return { "Cases Analysed": String(withRootCause), ...base };
    case "Oracle":
      return { "Predictions Made": String(withBlast), ...base };
    case "Surgeon":
      return { Resolved: String(resolved), "Awaiting Approval": String(awaiting), ...base };
    case "Scribe":
      return { "Reports Generated": String(withReport), ...base };
    default:
      return base;
  }
}

function deriveActivity(name: string, incidents: Incident[]): ActivityItem[] {
  const recent = [...incidents]
    .sort(
      (a, b) =>
        new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()
    )
    .slice(0, 5);

  const items: ActivityItem[] = [];
  for (const inc of recent) {
    const t = timeAgo(inc.updated_at);
    switch (name) {
      case "Sentinel":
        items.push({
          text: `Triggered: ${inc.pod} (${inc.anomaly_score.toFixed(2)})`,
          time: timeAgo(inc.created_at),
          ok: false,
        });
        break;
      case "Detective":
        if (inc.root_cause)
          items.push({ text: `Root cause: ${inc.pod}`, time: t, ok: true });
        break;
      case "Oracle":
        if (inc.blast_radius.length > 0)
          items.push({
            text: `Blast radius: ${inc.blast_radius.length} svc · ${inc.pod}`,
            time: t,
            ok: true,
          });
        break;
      case "Surgeon":
        if (inc.status === "awaiting_approval")
          items.push({ text: `Awaiting approval: ${inc.pod}`, time: t, ok: true });
        else if (inc.status === "executing")
          items.push({ text: `Executing: ${inc.pod}`, time: t, ok: true });
        else if (inc.status === "resolved")
          items.push({ text: `Remediated: ${inc.pod}`, time: t, ok: true });
        else if (inc.status === "failed")
          items.push({ text: `Failed: ${inc.pod}`, time: t, ok: false });
        else if (inc.status === "rejected")
          items.push({ text: `Dismissed: ${inc.pod}`, time: t, ok: true });
        break;
      case "Scribe":
        if (inc.report)
          items.push({ text: `Report generated: ${inc.pod}`, time: t, ok: true });
        break;
    }
  }
  return items.slice(0, 4);
}

const statusStyles: Record<
  AgentStatus,
  { label: string; dot: string; text: string; bg: string; border: string }
> = {
  active: {
    label: "Active",
    dot: "bg-emerald-500",
    text: "text-emerald-400",
    bg: "bg-emerald-500/10",
    border: "border-emerald-500/20",
  },
  waiting: {
    label: "Waiting",
    dot: "bg-orange-500",
    text: "text-orange-400",
    bg: "bg-orange-500/10",
    border: "border-orange-500/20",
  },
  idle: {
    label: "Idle",
    dot: "bg-white/20",
    text: "text-muted-foreground",
    bg: "bg-white/5",
    border: "border-white/8",
  },
};

function AgentCard({
  def,
  incidents,
  delay,
}: {
  def: AgentDef;
  incidents: Incident[];
  delay: number;
}) {
  const Icon = def.icon;
  const agentStatus = deriveStatus(def.name, incidents);
  const status = statusStyles[agentStatus];
  const stats = deriveStats(def, incidents);
  const activity = deriveActivity(def.name, incidents);

  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.5, ease: "easeOut" }}
      whileHover={{ y: -2, transition: { duration: 0.2 } }}
      className={cn(
        "rounded-xl border bg-[#0e0f11] p-6 flex flex-col gap-5 transition-shadow",
        def.glowBorder,
        agentStatus === "active" && `shadow-lg ${def.glowShadow}`
      )}
    >
      {/* Header */}
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <div
            className={cn(
              "w-10 h-10 rounded-xl bg-gradient-to-br flex items-center justify-center",
              def.gradient
            )}
          >
            <Icon className="w-5 h-5 text-white" />
          </div>
          <div>
            <p className="font-semibold text-white">{def.name}</p>
            <p className="text-xs text-muted-foreground">{def.role}</p>
          </div>
        </div>
        <div
          className={cn(
            "flex items-center gap-1.5 px-2.5 py-1 rounded-full border text-xs font-medium",
            status.bg,
            status.border,
            status.text
          )}
        >
          <motion.div
            className={cn("w-1.5 h-1.5 rounded-full", status.dot)}
            animate={agentStatus !== "idle" ? { opacity: [1, 0.4, 1] } : {}}
            transition={{ duration: 1.2, repeat: Infinity }}
          />
          {status.label}
        </div>
      </div>

      {/* Description */}
      <p className="text-xs text-muted-foreground leading-relaxed">
        {def.description}
      </p>

      {/* Stats grid */}
      <div className="grid grid-cols-2 gap-2">
        {Object.entries(stats).map(([key, value]) => (
          <div key={key} className={cn("rounded-lg p-3", def.accentBg)}>
            <p className={cn("text-lg font-mono font-bold", def.accentText)}>
              {value}
            </p>
            <p className="text-[10px] text-muted-foreground mt-0.5">{key}</p>
          </div>
        ))}
      </div>

      {/* Recent activity */}
      <div>
        <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-widest mb-2">
          Recent Activity
        </p>
        {activity.length === 0 ? (
          <p className="text-xs text-muted-foreground/60">No recent activity</p>
        ) : (
          <div className="space-y-2">
            {activity.map((event, i) => (
              <div key={i} className="flex items-center gap-2 text-xs">
                {event.ok ? (
                  <CheckCircle className="w-3 h-3 text-emerald-500 shrink-0" />
                ) : (
                  <Activity className="w-3 h-3 text-amber-400 shrink-0" />
                )}
                <span className="text-white/70 flex-1 truncate">{event.text}</span>
                <span className="text-muted-foreground shrink-0">{event.time}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </motion.div>
  );
}

export default function AgentsPage() {
  const { data } = useApi<Incident[]>(() => api.incidents.list(), [], {
    pollInterval: 5_000,
  });
  const incidents = data ?? [];

  const activeCount = AGENT_DEFS.filter(
    (d) => deriveStatus(d.name, incidents) !== "idle"
  ).length;

  return (
    <div className="space-y-6 max-w-[1400px] mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="font-display text-2xl font-semibold tracking-tight">Agents</h2>
          <p className="text-sm text-muted-foreground mt-1">
            {activeCount} of {AGENT_DEFS.length} agents currently working
          </p>
        </div>
        <div className="flex items-center gap-4 text-xs text-muted-foreground">
          <div className="flex items-center gap-1.5">
            <Zap className="w-3.5 h-3.5 text-blue-400" />
            LangGraph orchestrated
          </div>
          <div className="flex items-center gap-1.5">
            <Clock className="w-3.5 h-3.5 text-purple-400" />
            PyTorch LSTM backbone
          </div>
        </div>
      </div>

      {/* Agent cards grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-5">
        {AGENT_DEFS.map((def, i) => (
          <AgentCard
            key={def.name}
            def={def}
            incidents={incidents}
            delay={i * 0.1}
          />
        ))}
      </div>
    </div>
  );
}
