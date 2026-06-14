"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import {
  Search,
  AlertTriangle,
  LayoutDashboard,
  GitBranch,
  Bot,
  Shield,
  Brain,
  Wrench,
  FileText,
  Hash,
  ArrowRight,
} from "lucide-react";
import { useApi } from "@/hooks/use-api";
import { api } from "@/lib/api-client";
import type { Incident } from "@/lib/types";
import { cn } from "@/lib/utils";

// ─────────────────────────────────────────────────────────────────
// Static index — pages and agents are always searchable
// ─────────────────────────────────────────────────────────────────
type StaticEntry = {
  kind: "page" | "agent";
  label: string;
  hint: string;
  href: string;
  keywords: string[];
  icon: React.ElementType;
  color: string;
};

const STATIC_ENTRIES: StaticEntry[] = [
  { kind: "page", label: "Dashboard", hint: "Overview & metrics", href: "/", keywords: ["dashboard", "home", "overview", "metrics"], icon: LayoutDashboard, color: "text-blue-400" },
  { kind: "page", label: "Incidents", hint: "All incidents list", href: "/incidents", keywords: ["incidents", "alerts", "list"], icon: AlertTriangle, color: "text-red-400" },
  { kind: "page", label: "Pipeline", hint: "Active agent runs", href: "/pipeline", keywords: ["pipeline", "langgraph", "runs", "stages"], icon: GitBranch, color: "text-purple-400" },
  { kind: "page", label: "Agents", hint: "Agent details & status", href: "/agents", keywords: ["agents", "status"], icon: Bot, color: "text-emerald-400" },
  { kind: "agent", label: "Sentinel", hint: "Anomaly detection (LSTM)", href: "/agents", keywords: ["sentinel", "anomaly", "lstm", "detection"], icon: Shield, color: "text-blue-400" },
  { kind: "agent", label: "Detective", hint: "Root cause analysis (RAG)", href: "/agents", keywords: ["detective", "root cause", "rag", "qdrant"], icon: Search, color: "text-amber-400" },
  { kind: "agent", label: "Oracle", hint: "Blast radius prediction", href: "/agents", keywords: ["oracle", "blast radius", "prediction"], icon: Brain, color: "text-purple-400" },
  { kind: "agent", label: "Surgeon", hint: "Automated remediation", href: "/agents", keywords: ["surgeon", "remediation", "fix"], icon: Wrench, color: "text-emerald-400" },
  { kind: "agent", label: "Scribe", hint: "Post-mortem reports", href: "/agents", keywords: ["scribe", "report", "post-mortem", "postmortem"], icon: FileText, color: "text-cyan-400" },
];

// ─────────────────────────────────────────────────────────────────
// Result type returned to render
// ─────────────────────────────────────────────────────────────────
type SearchResult =
  | {
      key: string;
      kind: "page" | "agent";
      label: string;
      hint: string;
      icon: React.ElementType;
      color: string;
      href: string;
    }
  | {
      key: string;
      kind: "incident";
      label: string;
      hint: string;
      icon: React.ElementType;
      color: string;
      href: string;
      score: number;
      status: string;
    };

const STATUS_COLOR: Record<string, string> = {
  resolved: "text-emerald-400",
  awaiting_approval: "text-orange-400",
  analyzing: "text-blue-400",
  executing: "text-purple-400",
  detected: "text-yellow-400",
  failed: "text-red-400",
};

// ─────────────────────────────────────────────────────────────────
// Main component
// ─────────────────────────────────────────────────────────────────
export function GlobalSearch() {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const { data: incidents } = useApi<Incident[]>(
    () => api.incidents.list(),
    [],
    { pollInterval: 10_000 }
  );

  // ── Build result list from current query ────────────────────────
  const term = query.trim().toLowerCase();
  let results: SearchResult[] = [];

  if (term.length === 0) {
    // Empty query → show the four pages as suggestions
    results = STATIC_ENTRIES.filter((e) => e.kind === "page").map((e) => ({
      key: `${e.kind}-${e.label}`,
      kind: e.kind,
      label: e.label,
      hint: e.hint,
      icon: e.icon,
      color: e.color,
      href: e.href,
    }));
  } else {
    const incidentMatches: SearchResult[] = (incidents ?? [])
      .filter((inc) => {
        return (
          inc.pod.toLowerCase().includes(term) ||
          inc.namespace.toLowerCase().includes(term) ||
          inc.id.toLowerCase().includes(term) ||
          (inc.root_cause ?? "").toLowerCase().includes(term) ||
          inc.status.toLowerCase().includes(term)
        );
      })
      .slice(0, 6)
      .map((inc) => ({
        key: `incident-${inc.id}`,
        kind: "incident" as const,
        label: inc.pod,
        hint: inc.root_cause ?? "Analysing…",
        icon: AlertTriangle,
        color: STATUS_COLOR[inc.status] ?? "text-muted-foreground",
        href: "/incidents",
        score: inc.anomaly_score,
        status: inc.status,
      }));

    const staticMatches: SearchResult[] = STATIC_ENTRIES.filter((e) => {
      return (
        e.label.toLowerCase().includes(term) ||
        e.keywords.some((k) => k.includes(term))
      );
    })
      .slice(0, 6)
      .map((e) => ({
        key: `${e.kind}-${e.label}`,
        kind: e.kind,
        label: e.label,
        hint: e.hint,
        icon: e.icon,
        color: e.color,
        href: e.href,
      }));

    results = [...incidentMatches, ...staticMatches];
  }

  // Clamp active index when results change
  useEffect(() => {
    if (activeIndex >= results.length) setActiveIndex(0);
  }, [results.length, activeIndex]);

  // ── Global "/" hotkey to focus ─────────────────────────────────
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const target = e.target as HTMLElement;
      const isTyping =
        target.tagName === "INPUT" ||
        target.tagName === "TEXTAREA" ||
        target.isContentEditable;

      if (e.key === "/" && !isTyping) {
        e.preventDefault();
        inputRef.current?.focus();
        setOpen(true);
      }
      if (e.key === "Escape") {
        setOpen(false);
        inputRef.current?.blur();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // ── Click outside to close ─────────────────────────────────────
  useEffect(() => {
    function onClick(e: MouseEvent) {
      if (
        containerRef.current &&
        !containerRef.current.contains(e.target as Node)
      ) {
        setOpen(false);
      }
    }
    if (open) {
      window.addEventListener("mousedown", onClick);
      return () => window.removeEventListener("mousedown", onClick);
    }
  }, [open]);

  // ── Navigate to a result ───────────────────────────────────────
  function go(result: SearchResult) {
    router.push(result.href);
    setOpen(false);
    setQuery("");
    inputRef.current?.blur();
  }

  // ── Arrow-key / Enter handling inside the input ────────────────
  function onInputKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveIndex((i) => Math.min(i + 1, Math.max(0, results.length - 1)));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveIndex((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      const r = results[activeIndex];
      if (r) go(r);
    }
  }

  return (
    <div ref={containerRef} className="relative">
      <div className="flex items-center gap-3 bg-white/5 rounded-lg px-3 py-1.5 w-80 border border-transparent focus-within:border-white/15 transition-colors">
        <Search className="w-4 h-4 text-muted-foreground shrink-0" />
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setOpen(true);
            setActiveIndex(0);
          }}
          onFocus={() => setOpen(true)}
          onKeyDown={onInputKeyDown}
          placeholder="Search incidents, agents, pages…"
          className="bg-transparent text-sm text-white placeholder:text-muted-foreground focus:outline-none w-full"
        />
        <kbd className="text-[10px] text-muted-foreground bg-white/5 px-1.5 py-0.5 rounded shrink-0">
          {open ? "esc" : "/"}
        </kbd>
      </div>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.15 }}
            className="absolute left-0 right-0 mt-2 rounded-xl border border-white/10 bg-[#0e0f11] backdrop-blur-xl shadow-2xl shadow-black/40 overflow-hidden z-50"
          >
            {results.length === 0 ? (
              <div className="p-6 text-center text-xs text-muted-foreground">
                No matches for{" "}
                <span className="text-white font-mono">&quot;{query}&quot;</span>
              </div>
            ) : (
              <div className="max-h-[420px] overflow-y-auto py-1">
                {term.length === 0 && (
                  <p className="px-3 py-2 text-[10px] font-semibold text-muted-foreground uppercase tracking-widest">
                    Jump to
                  </p>
                )}
                {results.map((r, i) => {
                  const Icon = r.icon;
                  const isActive = i === activeIndex;
                  return (
                    <button
                      key={r.key}
                      onMouseEnter={() => setActiveIndex(i)}
                      onClick={() => go(r)}
                      className={cn(
                        "w-full text-left px-3 py-2.5 flex items-center gap-3 transition-colors",
                        isActive ? "bg-white/[0.06]" : "hover:bg-white/[0.04]"
                      )}
                    >
                      <div
                        className={cn(
                          "w-7 h-7 rounded-lg bg-white/5 flex items-center justify-center shrink-0",
                          r.color
                        )}
                      >
                        <Icon className="w-3.5 h-3.5" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="text-sm text-white font-medium truncate">
                            {r.label}
                          </span>
                          <span className="text-[10px] text-muted-foreground uppercase tracking-wider shrink-0">
                            {r.kind}
                          </span>
                          {r.kind === "incident" && (
                            <span className="text-[10px] font-mono text-muted-foreground bg-white/5 px-1.5 py-0.5 rounded shrink-0 inline-flex items-center gap-0.5">
                              <Hash className="w-2.5 h-2.5" />
                              {r.score.toFixed(2)}
                            </span>
                          )}
                        </div>
                        <p className="text-xs text-muted-foreground truncate">
                          {r.hint}
                        </p>
                      </div>
                      <ArrowRight
                        className={cn(
                          "w-3.5 h-3.5 transition-opacity shrink-0",
                          isActive ? "opacity-100 text-white" : "opacity-0"
                        )}
                      />
                    </button>
                  );
                })}
              </div>
            )}

            {/* Footer hint */}
            <div className="border-t border-white/8 px-3 py-1.5 flex items-center gap-3 text-[10px] text-muted-foreground">
              <span className="flex items-center gap-1">
                <kbd className="bg-white/5 px-1 rounded">↑↓</kbd> navigate
              </span>
              <span className="flex items-center gap-1">
                <kbd className="bg-white/5 px-1 rounded">↵</kbd> select
              </span>
              <span className="flex items-center gap-1">
                <kbd className="bg-white/5 px-1 rounded">esc</kbd> close
              </span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
