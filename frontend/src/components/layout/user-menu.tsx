"use client";

import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Wifi, WifiOff, FileText, Activity, Cpu } from "lucide-react";
import { useApi } from "@/hooks/use-api";
import { api } from "@/lib/api-client";
import type { HealthResponse } from "@/lib/types";
import { cn } from "@/lib/utils";

// Backend base URL for direct links (API docs, raw metrics). Local dev default.
const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

export function UserMenu() {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  const { data: health } = useApi<HealthResponse>(() => api.health(), [], {
    pollInterval: 10_000,
  });

  const connected = health?.status === "ok";
  const sentinelActive = health?.agents?.sentinel === "active";

  useEffect(() => {
    function onClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    if (open) {
      window.addEventListener("mousedown", onClick);
      return () => window.removeEventListener("mousedown", onClick);
    }
  }, [open]);

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-8 h-8 rounded-full bg-signal/15 border border-signal/30 flex items-center justify-center text-xs font-bold text-signal ring-2 ring-transparent hover:ring-white/20 transition-all"
        aria-label="Account menu"
      >
        SS
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.15 }}
            className="absolute right-0 mt-2 w-72 rounded-xl border border-white/10 bg-[#0e0f11] backdrop-blur-xl shadow-2xl shadow-black/40 overflow-hidden z-50"
          >
            {/* Identity header */}
            <div className="flex items-center gap-3 px-4 py-3 border-b border-white/8">
              <div className="w-9 h-9 rounded-full bg-signal/15 border border-signal/30 flex items-center justify-center text-xs font-bold text-signal">
                SS
              </div>
              <div className="min-w-0">
                <p className="text-sm font-semibold text-white">SRE Operator</p>
                <p className="text-[10px] text-muted-foreground">
                  Local instance · no auth
                </p>
              </div>
            </div>

            {/* System status */}
            <div className="px-4 py-3 border-b border-white/8 space-y-2">
              <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-widest">
                System
              </p>
              <div className="flex items-center justify-between text-xs">
                <span className="flex items-center gap-2 text-muted-foreground">
                  {connected ? (
                    <Wifi className="w-3.5 h-3.5 text-emerald-400" />
                  ) : (
                    <WifiOff className="w-3.5 h-3.5 text-muted-foreground" />
                  )}
                  Backend
                </span>
                <span className={connected ? "text-emerald-400" : "text-muted-foreground"}>
                  {connected ? "Connected" : "Offline"}
                </span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="flex items-center gap-2 text-muted-foreground">
                  <Cpu className="w-3.5 h-3.5 text-purple-400" />
                  Sentinel LSTM
                </span>
                <span className={sentinelActive ? "text-emerald-400" : "text-muted-foreground"}>
                  {sentinelActive ? "Loaded" : "Not loaded"}
                </span>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="flex items-center gap-2 text-muted-foreground">
                  <Activity className="w-3.5 h-3.5 text-blue-400" />
                  Pipeline
                </span>
                <span className="text-muted-foreground capitalize">
                  {health?.pipeline ?? "—"}
                </span>
              </div>
            </div>

            {/* Quick links */}
            <div className="py-1">
              <a
                href={`${BACKEND_URL}/docs`}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-3 px-4 py-2.5 text-sm text-muted-foreground hover:text-white hover:bg-white/[0.04] transition-colors"
              >
                <FileText className="w-3.5 h-3.5" />
                API Documentation
              </a>
              <a
                href={`${BACKEND_URL}/metrics`}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-3 px-4 py-2.5 text-sm text-muted-foreground hover:text-white hover:bg-white/[0.04] transition-colors"
              >
                <Activity className="w-3.5 h-3.5" />
                Prometheus Metrics
              </a>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
