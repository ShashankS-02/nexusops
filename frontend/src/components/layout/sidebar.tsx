"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion } from "framer-motion";
import {
  LayoutDashboard,
  AlertTriangle,
  GitBranch,
  Bot,
  Shield,
  Search,
  Brain,
  Wrench,
  FileText,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useApi } from "@/hooks/use-api";
import { api } from "@/lib/api-client";
import type { HealthResponse } from "@/lib/types";
import { LogoMark } from "@/components/marketing/logo-mark";

const navItems = [
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/incidents", label: "Incidents", icon: AlertTriangle },
  { href: "/pipeline", label: "Pipeline", icon: GitBranch },
  { href: "/agents", label: "Agents", icon: Bot },
];

const agentDefs = [
  { name: "Sentinel", key: "sentinel", icon: Shield, color: "text-blue-400" },
  { name: "Detective", key: "detective", icon: Search, color: "text-amber-400" },
  { name: "Oracle", key: "oracle", icon: Brain, color: "text-purple-400" },
  { name: "Surgeon", key: "surgeon", icon: Wrench, color: "text-emerald-400" },
  { name: "Scribe", key: "scribe", icon: FileText, color: "text-cyan-400" },
];

function AgentDot({ status }: { status: "active" | "idle" | "error" | undefined }) {
  if (status === "active") {
    return (
      <motion.div
        className="ml-auto w-1.5 h-1.5 rounded-full bg-emerald-500"
        animate={{ opacity: [1, 0.4, 1] }}
        transition={{ duration: 1.4, repeat: Infinity }}
      />
    );
  }
  if (status === "error") {
    return <div className="ml-auto w-1.5 h-1.5 rounded-full bg-red-500" />;
  }
  return <div className="ml-auto w-1.5 h-1.5 rounded-full bg-white/15" />;
}

export function Sidebar() {
  const pathname = usePathname();
  const { data: health } = useApi<HealthResponse>(
    () => api.health(),
    [],
    { pollInterval: 10_000 }
  );

  const sysHealthy = !health || health.status === "ok";

  return (
    <aside className="w-64 border-r border-white/8 bg-[#0b0c0e] flex flex-col">
      {/* Logo */}
      <div className="p-6 border-b border-white/8">
        <Link href="/dashboard" className="flex items-center gap-3">
          <LogoMark className="h-8 w-8" />
          <div>
            <h1 className="font-display text-lg font-semibold tracking-tight">NexusOps</h1>
            <p className="text-[10px] text-muted-foreground uppercase tracking-widest">
              Autonomous SRE
            </p>
          </div>
        </Link>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-1 overflow-y-auto">
        <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-widest mb-3 px-3">
          Overview
        </p>
        {navItems.map((item) => {
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "relative flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors",
                isActive
                  ? "text-white"
                  : "text-muted-foreground hover:text-white hover:bg-white/5"
              )}
            >
              {isActive && (
                <motion.div
                  layoutId="activeNav"
                  className="absolute inset-0 rounded-lg bg-signal/10 border border-signal/25"
                  transition={{ type: "spring", bounce: 0.2, duration: 0.4 }}
                />
              )}
              <item.icon className="w-4 h-4 relative z-10" />
              <span className="relative z-10">{item.label}</span>
            </Link>
          );
        })}

        {/* Agents section */}
        <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-widest mt-8 mb-3 px-3">
          Agents
        </p>
        {agentDefs.map((agent) => {
          const agentStatus = health?.agents?.[agent.key];
          return (
            <div
              key={agent.name}
              className="flex items-center gap-3 px-3 py-2 rounded-lg text-sm text-muted-foreground"
            >
              <agent.icon className={cn("w-4 h-4", agent.color)} />
              <span>{agent.name}</span>
              <AgentDot status={agentStatus} />
            </div>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-white/8">
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <motion.div
            className={cn(
              "w-2 h-2 rounded-full",
              sysHealthy ? "bg-emerald-500" : "bg-red-500"
            )}
            animate={sysHealthy ? { opacity: [1, 0.5, 1] } : {}}
            transition={{ duration: 2, repeat: Infinity }}
          />
          <span>{sysHealthy ? "System Healthy" : "System Degraded"}</span>
        </div>
      </div>
    </aside>
  );
}
