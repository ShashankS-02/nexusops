"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { Bell, Clock, XCircle, CheckCircle, BellOff } from "lucide-react";
import { useApi } from "@/hooks/use-api";
import { api } from "@/lib/api-client";
import type { Incident } from "@/lib/types";
import { cn } from "@/lib/utils";

type Notification = {
  id: string;
  pod: string;
  kind: "approval" | "failed" | "resolved";
  message: string;
  at: string;
  actionable: boolean;
};

function timeAgo(iso: string): string {
  const m = Math.floor((Date.now() - new Date(iso).getTime()) / 60_000);
  if (m < 1) return "just now";
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

function buildNotifications(incidents: Incident[]): Notification[] {
  const notes: Notification[] = [];
  for (const inc of incidents) {
    if (inc.status === "awaiting_approval") {
      notes.push({
        id: inc.id,
        pod: inc.pod,
        kind: "approval",
        message: "Remediation needs your approval",
        at: inc.updated_at,
        actionable: true,
      });
    } else if (inc.status === "failed") {
      notes.push({
        id: inc.id,
        pod: inc.pod,
        kind: "failed",
        message: "Remediation failed — manual action needed",
        at: inc.updated_at,
        actionable: true,
      });
    } else if (
      inc.status === "resolved" &&
      Date.now() - new Date(inc.updated_at).getTime() < 30 * 60_000
    ) {
      notes.push({
        id: inc.id,
        pod: inc.pod,
        kind: "resolved",
        message: "Incident resolved",
        at: inc.updated_at,
        actionable: false,
      });
    }
  }
  // Actionable first, then most recent
  return notes.sort((a, b) => {
    if (a.actionable !== b.actionable) return a.actionable ? -1 : 1;
    return new Date(b.at).getTime() - new Date(a.at).getTime();
  });
}

const kindConfig = {
  approval: { icon: Clock, color: "text-orange-400", bg: "bg-orange-500/10" },
  failed: { icon: XCircle, color: "text-red-400", bg: "bg-red-500/10" },
  resolved: { icon: CheckCircle, color: "text-emerald-400", bg: "bg-emerald-500/10" },
};

export function NotificationBell() {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  const { data } = useApi<Incident[]>(() => api.incidents.list(), [], {
    pollInterval: 5_000,
  });

  const notifications = buildNotifications(data ?? []);
  const actionableCount = notifications.filter((n) => n.actionable).length;

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

  function go(n: Notification) {
    router.push("/incidents");
    setOpen(false);
  }

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        className="relative p-2 rounded-lg hover:bg-white/5 transition-colors"
        aria-label="Notifications"
      >
        <Bell className="w-4 h-4 text-muted-foreground" />
        {actionableCount > 0 && (
          <span className="absolute -top-0.5 -right-0.5 min-w-[16px] h-4 px-1 rounded-full bg-red-500 text-[9px] font-bold text-white flex items-center justify-center">
            {actionableCount}
          </span>
        )}
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.15 }}
            className="absolute right-0 mt-2 w-80 rounded-xl border border-white/10 bg-[#0e0f11] backdrop-blur-xl shadow-2xl shadow-black/40 overflow-hidden z-50"
          >
            <div className="flex items-center justify-between px-4 py-3 border-b border-white/8">
              <span className="text-sm font-semibold text-white">Notifications</span>
              {actionableCount > 0 && (
                <span className="text-[10px] text-orange-400 bg-orange-500/10 px-2 py-0.5 rounded-full">
                  {actionableCount} need action
                </span>
              )}
            </div>

            {notifications.length === 0 ? (
              <div className="py-10 text-center">
                <BellOff className="w-7 h-7 text-muted-foreground/30 mx-auto mb-2" />
                <p className="text-xs text-muted-foreground">All caught up</p>
              </div>
            ) : (
              <div className="max-h-[360px] overflow-y-auto py-1">
                {notifications.map((n) => {
                  const cfg = kindConfig[n.kind];
                  const Icon = cfg.icon;
                  return (
                    <button
                      key={`${n.kind}-${n.id}`}
                      onClick={() => go(n)}
                      className="w-full text-left px-4 py-2.5 flex items-start gap-3 hover:bg-white/[0.04] transition-colors"
                    >
                      <div
                        className={cn(
                          "w-7 h-7 rounded-lg flex items-center justify-center shrink-0 mt-0.5",
                          cfg.bg,
                          cfg.color
                        )}
                      >
                        <Icon className="w-3.5 h-3.5" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="text-sm text-white font-mono truncate">
                            {n.pod}
                          </span>
                          <span className="text-[10px] text-muted-foreground shrink-0">
                            {timeAgo(n.at)}
                          </span>
                        </div>
                        <p className="text-xs text-muted-foreground truncate">
                          {n.message}
                        </p>
                      </div>
                    </button>
                  );
                })}
              </div>
            )}

            <button
              onClick={() => {
                router.push("/incidents");
                setOpen(false);
              }}
              className="w-full border-t border-white/8 px-4 py-2.5 text-xs text-signal hover:bg-white/[0.03] transition-colors text-center"
            >
              View all incidents
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
