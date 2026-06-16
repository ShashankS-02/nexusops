"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

// Deterministic anomaly-score series (no Math.random → SSR-safe).
// Calm baseline, then a ramp past the 0.60 threshold into an incident.
const SCORES = [
  0.28, 0.31, 0.26, 0.3, 0.33, 0.27, 0.29, 0.32, 0.28, 0.34, 0.3, 0.27, 0.31,
  0.29, 0.33, 0.3, 0.28, 0.35, 0.31, 0.29, 0.32, 0.3, 0.34, 0.31, 0.28, 0.33,
  0.36, 0.4, 0.47, 0.55, 0.63, 0.71, 0.78, 0.83, 0.81, 0.84, 0.82, 0.8, 0.83,
  0.85,
];

const W = 300;
const H = 96;
const PAD = 6;
const x = (i: number) => PAD + (i * (W - PAD * 2)) / (SCORES.length - 1);
const y = (s: number) => H - 8 - s * (H - 20);
const THRESHOLD_Y = y(0.6);

const linePath = SCORES.map((s, i) => `${i === 0 ? "M" : "L"} ${x(i).toFixed(1)} ${y(s).toFixed(1)}`).join(" ");
const areaPath = `${linePath} L ${x(SCORES.length - 1)} ${H} L ${x(0)} ${H} Z`;

type NodeStatus = "idle" | "active" | "done" | "waiting";

const NODES = ["Sentinel", "Detective", "Oracle", "HITL", "Surgeon", "Scribe"];

// Step → readout line + per-node status
const STEPS: { label: string; tone: "mute" | "signal" | "amber" | "red"; status: NodeStatus[] }[] = [
  { label: "Monitoring metric stream · 5 pods", tone: "mute", status: ["active", "idle", "idle", "idle", "idle", "idle"] },
  { label: "Anomaly detected · score 0.83 ▲ 0.60", tone: "red", status: ["done", "idle", "idle", "idle", "idle", "idle"] },
  { label: "Detective · tracing root cause (RAG)", tone: "amber", status: ["done", "active", "idle", "idle", "idle", "idle"] },
  { label: "Oracle · blast radius = 3 services", tone: "amber", status: ["done", "done", "active", "idle", "idle", "idle"] },
  { label: "Awaiting human approval", tone: "amber", status: ["done", "done", "done", "waiting", "idle", "idle"] },
  { label: "Surgeon · kubectl rollout restart (dry-run)", tone: "signal", status: ["done", "done", "done", "done", "active", "idle"] },
  { label: "Resolved · MTTR 4m12s · post-mortem filed", tone: "signal", status: ["done", "done", "done", "done", "done", "done"] },
];

const toneColor = {
  mute: "text-mute",
  signal: "text-signal",
  amber: "text-amber-400",
  red: "text-red-400",
};

const nodeStyles: Record<NodeStatus, string> = {
  idle: "border-white/8 text-mute-soft",
  active: "border-amber-400/40 text-amber-300 bg-amber-400/5",
  waiting: "border-amber-400/50 text-amber-300 bg-amber-400/10",
  done: "border-signal/40 text-signal bg-signal/5",
};

export function LiveConsole() {
  const [step, setStep] = useState(0);

  useEffect(() => {
    const id = setInterval(() => setStep((s) => (s + 1) % STEPS.length), 1700);
    return () => clearInterval(id);
  }, []);

  const current = STEPS[step];
  const anomalyActive = step >= 1;

  return (
    <div className="relative w-full">
      {/* faint signal glow behind the panel */}
      <div className="absolute -inset-6 signal-glow opacity-60 blur-2xl" aria-hidden />

      <div className="relative rounded-[10px] border border-white/10 bg-ink-raised/90 backdrop-blur-sm shadow-2xl shadow-black/60 overflow-hidden">
        {/* Panel header */}
        <div className="flex items-center justify-between border-b border-white/8 px-4 py-2.5">
          <div className="flex items-center gap-2">
            <span className="relative flex h-1.5 w-1.5">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-signal opacity-60" />
              <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-signal" />
            </span>
            <span className="font-mono text-[11px] tracking-wide text-white/70">
              nexusops · sentinel
            </span>
          </div>
          <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-mute-soft">
            live
          </span>
        </div>

        {/* Anomaly chart */}
        <div className="px-4 pt-4">
          <div className="flex items-baseline justify-between">
            <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-mute-soft">
              anomaly score
            </span>
            <span
              className={cn(
                "font-mono text-sm font-semibold tabular-nums",
                anomalyActive ? "text-red-400" : "text-white/70"
              )}
            >
              {anomalyActive ? "0.83" : "0.31"}
              <span className="ml-1 text-[10px] text-mute-soft">/ thr 0.60</span>
            </span>
          </div>

          <svg viewBox={`0 0 ${W} ${H}`} className="mt-2 w-full" preserveAspectRatio="none">
            <defs>
              <linearGradient id="area" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--color-signal)" stopOpacity="0.18" />
                <stop offset="100%" stopColor="var(--color-signal)" stopOpacity="0" />
              </linearGradient>
            </defs>
            {/* threshold */}
            <line
              x1={PAD}
              y1={THRESHOLD_Y}
              x2={W - PAD}
              y2={THRESHOLD_Y}
              stroke="rgba(248,113,113,0.5)"
              strokeWidth="1"
              strokeDasharray="3 3"
            />
            <motion.path
              d={areaPath}
              fill="url(#area)"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 1 }}
            />
            <motion.path
              d={linePath}
              fill="none"
              stroke="var(--color-signal)"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 1.6, ease: "easeInOut" }}
            />
            {/* live marker at the spike */}
            <motion.circle
              cx={x(SCORES.length - 1)}
              cy={y(SCORES[SCORES.length - 1])}
              r="2.6"
              fill={anomalyActive ? "#f87171" : "var(--color-signal)"}
              animate={{ opacity: [1, 0.3, 1] }}
              transition={{ duration: 1.2, repeat: Infinity }}
            />
          </svg>
        </div>

        {/* Incident row */}
        <div className="mx-4 mt-3 flex items-center justify-between rounded-[6px] border border-white/8 bg-ink/60 px-3 py-2">
          <div className="flex items-center gap-2 min-w-0">
            <span
              className={cn(
                "h-1.5 w-1.5 rounded-full shrink-0",
                anomalyActive ? "bg-red-500" : "bg-mute-soft"
              )}
            />
            <span className="font-mono text-[11px] text-white/80 truncate">
              payment-service
            </span>
            <span className="font-mono text-[10px] text-mute-soft">prod</span>
          </div>
          <span className="font-mono text-[10px] uppercase tracking-wider text-mute-soft">
            #15460a2f
          </span>
        </div>

        {/* Pipeline mini */}
        <div className="px-4 pt-4 pb-3">
          <div className="flex items-center gap-1">
            {NODES.map((n, i) => (
              <div key={n} className="flex items-center gap-1 flex-1 last:flex-none">
                <div
                  className={cn(
                    "flex-1 rounded-[5px] border px-1.5 py-1.5 text-center transition-colors duration-300",
                    nodeStyles[current.status[i]]
                  )}
                >
                  <span className="font-mono text-[8.5px] uppercase tracking-wider">
                    {n}
                  </span>
                </div>
                {i < NODES.length - 1 && (
                  <div
                    className={cn(
                      "h-px w-2 shrink-0 transition-colors duration-300",
                      current.status[i] === "done" ? "bg-signal/50" : "bg-white/10"
                    )}
                  />
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Status readout */}
        <div className="flex items-center gap-2 border-t border-white/8 px-4 py-2.5">
          <span className="font-mono text-[10px] text-mute-soft">›</span>
          <motion.span
            key={step}
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            className={cn("font-mono text-[11px]", toneColor[current.tone])}
          >
            {current.label}
          </motion.span>
        </div>
      </div>
    </div>
  );
}
