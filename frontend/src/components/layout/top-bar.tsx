"use client";

import { useState } from "react";
import { Zap, WifiOff, Plus, Check } from "lucide-react";
import { motion } from "framer-motion";
import { useApi } from "@/hooks/use-api";
import { api, triggerDemoIncident } from "@/lib/api-client";
import { isDemoMode } from "@/lib/demo-engine";
import type { HealthResponse } from "@/lib/types";
import { cn } from "@/lib/utils";
import { GlobalSearch } from "./global-search";
import { NotificationBell } from "./notification-bell";
import { UserMenu } from "./user-menu";

function DemoControls() {
  const [fired, setFired] = useState(false);

  function trigger() {
    triggerDemoIncident();
    setFired(true);
    setTimeout(() => setFired(false), 900);
  }

  return (
    <div className="flex items-center gap-2">
      <span
        title="Simulated data — no backend connected"
        className="hidden md:inline-flex items-center gap-1.5 rounded-full border border-signal/25 bg-signal/10 px-2.5 py-1 font-mono text-[10px] uppercase tracking-[0.16em] text-signal"
      >
        <span className="h-1.5 w-1.5 rounded-full bg-signal" />
        Demo
      </span>
      <button
        onClick={trigger}
        className="inline-flex items-center gap-1.5 rounded-lg border border-white/12 px-2.5 py-1.5 text-xs font-medium text-white/80 hover:bg-white/5 transition-colors"
      >
        {fired ? (
          <>
            <Check className="w-3.5 h-3.5 text-signal" />
            <span className="hidden sm:inline">Triggered</span>
          </>
        ) : (
          <>
            <Plus className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Trigger incident</span>
          </>
        )}
      </button>
    </div>
  );
}

function PipelineStatus() {
  const { data, loading } = useApi<HealthResponse>(
    () => api.health(),
    [],
    { pollInterval: 10_000 }
  );

  if (loading) {
    return (
      <div className="w-28 h-6 rounded-full bg-white/5 animate-pulse" />
    );
  }

  const active = data?.pipeline === "active";
  const degraded = data?.status === "degraded";

  if (degraded) {
    return (
      <div className="flex items-center gap-2 bg-white/5 border border-white/10 rounded-full px-3 py-1">
        <WifiOff className="w-3.5 h-3.5 text-muted-foreground" />
        <span className="text-xs font-medium text-muted-foreground">
          Backend offline
        </span>
      </div>
    );
  }

  return (
    <motion.div
      className={cn(
        "flex items-center gap-2 rounded-full px-3 py-1 border",
        active
          ? "bg-emerald-500/10 border-emerald-500/20"
          : "bg-white/5 border-white/10"
      )}
      animate={active ? { opacity: [0.7, 1, 0.7] } : {}}
      transition={{ duration: 2, repeat: Infinity }}
    >
      <Zap
        className={cn(
          "w-3.5 h-3.5",
          active ? "text-emerald-400" : "text-muted-foreground"
        )}
      />
      <span
        className={cn(
          "text-xs font-medium",
          active ? "text-emerald-400" : "text-muted-foreground"
        )}
      >
        {active ? "Pipeline Active" : "Pipeline Idle"}
      </span>
    </motion.div>
  );
}

export function TopBar() {
  return (
    <header className="relative z-50 h-14 border-b border-white/8 bg-[#0b0c0e]/80 backdrop-blur-xl flex items-center justify-between px-6">
      {/* Search */}
      <GlobalSearch />

      {/* Right side */}
      <div className="flex items-center gap-3">
        {isDemoMode() && <DemoControls />}
        <PipelineStatus />
        <NotificationBell />
        <UserMenu />
      </div>
    </header>
  );
}
