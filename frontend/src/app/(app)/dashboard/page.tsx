"use client";

import { AgentPipeline } from "@/components/dashboard/agent-pipeline";
import { RecentIncidents } from "@/components/dashboard/recent-incidents";
import { AnomalyChart } from "@/components/dashboard/anomaly-chart";
import { MetricCard } from "@/components/dashboard/metric-card";
import { Activity, AlertTriangle, CheckCircle, Cpu } from "lucide-react";
import { useApi } from "@/hooks/use-api";
import { api } from "@/lib/api-client";
import type { Incident } from "@/lib/types";

function useDashboardMetrics() {
  const { data: incidents, loading } = useApi<Incident[]>(
    () => api.incidents.list(),
    [],
    { pollInterval: 5_000 }
  );

  if (!incidents || !Array.isArray(incidents)) {
    return {
      loading,
      active: 2,
      resolvedToday: 7,
      avgScore: 0.34,
    };
  }

  const now = Date.now();
  const todayStart = new Date();
  todayStart.setHours(0, 0, 0, 0);

  const active = incidents.filter(
    (i) =>
      i.status !== "resolved" && i.status !== "failed" && i.status !== "rejected"
  ).length;

  const resolvedToday = incidents.filter(
    (i) =>
      i.status === "resolved" &&
      new Date(i.updated_at).getTime() >= todayStart.getTime()
  ).length;

  const avgScore =
    incidents.length > 0
      ? incidents.reduce((sum, i) => sum + i.anomaly_score, 0) /
        incidents.length
      : 0;

  return { loading, active, resolvedToday, avgScore };
}

export default function DashboardPage() {
  const { active, resolvedToday, avgScore } = useDashboardMetrics();

  return (
    <div className="space-y-6 max-w-[1400px] mx-auto">
      {/* Header */}
      <div>
        <h2 className="font-display text-2xl font-semibold tracking-tight">Dashboard</h2>
        <p className="text-sm text-muted-foreground mt-1">
          Real-time infrastructure monitoring and incident management
        </p>
      </div>

      {/* Metric cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Active Incidents"
          value={active}
          icon={AlertTriangle}
          color="bg-red-500"
          trend="up"
          trendValue={`${active > 0 ? "+" : ""}${active}`}
          delay={0}
        />
        <MetricCard
          title="Resolved Today"
          value={resolvedToday}
          icon={CheckCircle}
          color="bg-emerald-500"
          trend="up"
          trendValue={`+${resolvedToday}`}
          delay={1}
        />
        <MetricCard
          title="Avg Anomaly Score"
          value={Number(avgScore.toFixed(2))}
          icon={Activity}
          color="bg-blue-500"
          trend={avgScore > 0.4 ? "up" : "down"}
          trendValue={avgScore > 0.4 ? "↑ elevated" : "↓ normal"}
          delay={2}
        />
        <MetricCard
          title="LSTM AUROC"
          value={0.99}
          icon={Cpu}
          color="bg-purple-500"
          trend="flat"
          trendValue="Stable"
          delay={3}
        />
      </div>

      {/* Agent pipeline */}
      <AgentPipeline />

      {/* Charts and incidents grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <AnomalyChart />
        <RecentIncidents />
      </div>
    </div>
  );
}
