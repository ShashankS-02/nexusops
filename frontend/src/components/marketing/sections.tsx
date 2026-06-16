"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowUpRight, Check, Pause } from "lucide-react";
import { cn } from "@/lib/utils";
import { Reveal, Eyebrow, Section } from "./primitives";

/* ──────────────────────────────────────────────────────────────
   STAT STRIP — full-bleed marquee ticker
   ────────────────────────────────────────────────────────────── */
const STATS = [
  "5 specialized agents",
  "PyTorch LSTM autoencoder",
  "AUROC 0.99",
  "anomaly threshold 0.60",
  "RAG over Qdrant",
  "LangGraph orchestration",
  "human-in-the-loop gate",
  "Kubernetes remediation",
  "self-improving memory",
  "auto-filed post-mortems",
];

export function StatStrip() {
  return (
    <div className="relative border-y border-white/8 bg-ink-raised/40 overflow-hidden">
      <div className="flex w-max marquee">
        {[0, 1].map((dup) => (
          <div key={dup} className="flex items-center" aria-hidden={dup === 1}>
            {STATS.map((s) => (
              <div key={s} className="flex items-center">
                <span className="px-6 py-3.5 font-mono text-[11px] uppercase tracking-[0.18em] text-mute whitespace-nowrap">
                  {s}
                </span>
                <span className="text-signal/50 text-[8px]">◆</span>
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

/* ──────────────────────────────────────────────────────────────
   THESIS — the why
   ────────────────────────────────────────────────────────────── */
export function Thesis() {
  return (
    <Section className="grid gap-12 lg:grid-cols-[0.9fr_1.1fr] lg:gap-20">
      <Reveal>
        <Eyebrow index="//">Why NexusOps</Eyebrow>
        <h2 className="mt-6 font-display text-3xl font-semibold leading-[1.1] tracking-[-0.02em] text-white text-balance sm:text-[2.6rem]">
          Alerts don&apos;t fix themselves. On-call shouldn&apos;t have to either.
        </h2>
      </Reveal>

      <Reveal delay={0.1} className="space-y-5">
        <p className="text-[15px] leading-relaxed text-mute">
          A modern cluster fires hundreds of alerts a day. Most are noise; a few
          are real. By the time a human pieces together what broke, reads the
          runbook, and types the fix, minutes of downtime have already passed.
        </p>
        <p className="text-[15px] leading-relaxed text-mute">
          NexusOps collapses that loop. It scores anomalies in real time,
          reasons about the root cause from past incidents, predicts what else
          will break, and prepares the exact remediation — then hands you a
          single decision: <span className="text-white">approve, or don&apos;t.</span>
        </p>
        <div className="grid grid-cols-3 gap-4 pt-2">
          {[
            ["≈ 0", "manual triage steps"],
            ["1", "human decision per incident"],
            ["minutes", "target MTTR"],
          ].map(([big, small]) => (
            <div key={small} className="border-l border-white/10 pl-4">
              <div className="font-display text-2xl font-semibold text-signal">
                {big}
              </div>
              <div className="mt-1 font-mono text-[10px] uppercase tracking-wider text-mute-soft">
                {small}
              </div>
            </div>
          ))}
        </div>
      </Reveal>
    </Section>
  );
}

/* ──────────────────────────────────────────────────────────────
   LIFECYCLE — incident timeline
   ────────────────────────────────────────────────────────────── */
const LIFECYCLE = [
  {
    n: "01",
    agent: "Sentinel",
    title: "Detect & score",
    status: "detected",
    body: "Metrics stream into FastAPI. Sentinel runs each 30-point window through a PyTorch LSTM autoencoder. A reconstruction score above 0.60 opens an incident.",
  },
  {
    n: "02",
    agent: "Detective",
    title: "Diagnose root cause",
    status: "analyzing",
    body: "The alert is embedded and matched against past incidents in Qdrant. The LLM reasons over those precedents to produce a root cause and a confidence score.",
  },
  {
    n: "03",
    agent: "Oracle",
    title: "Predict & plan",
    status: "analyzing",
    body: "Oracle estimates the blast radius across dependent services and drafts one to three remediation actions, each with a kubectl command and a risk level.",
  },
  {
    n: "04",
    agent: "HITL Gate",
    title: "Pause for approval",
    status: "awaiting_approval",
    body: "The LangGraph pipeline halts before any action. State is checkpointed to disk. A human reviews the plan and approves or rejects — minutes or hours later.",
  },
  {
    n: "05",
    agent: "Surgeon",
    title: "Remediate",
    status: "executing",
    body: "On approval, Surgeon executes the actions against the Kubernetes API (dry-run in this build), stopping on the first failure to avoid chaining risky changes.",
  },
  {
    n: "06",
    agent: "Scribe",
    title: "Document & learn",
    status: "resolved",
    body: "Scribe writes a Markdown post-mortem and embeds it back into Qdrant — so the next time a similar incident fires, the Detective already knows the answer.",
  },
];

const statusPill: Record<string, string> = {
  detected: "text-yellow-400 border-yellow-400/25 bg-yellow-400/5",
  analyzing: "text-blue-400 border-blue-400/25 bg-blue-400/5",
  awaiting_approval: "text-amber-400 border-amber-400/25 bg-amber-400/5",
  executing: "text-purple-400 border-purple-400/25 bg-purple-400/5",
  resolved: "text-signal border-signal/30 bg-signal/5",
};

export function Lifecycle() {
  return (
    <Section id="lifecycle">
      <Reveal>
        <Eyebrow index="01">Incident Lifecycle</Eyebrow>
        <h2 className="mt-6 max-w-2xl font-display text-3xl font-semibold leading-[1.1] tracking-[-0.02em] text-white text-balance sm:text-[2.6rem]">
          From signal to resolution — autonomously, end to end.
        </h2>
      </Reveal>

      <div className="relative mt-14">
        {/* vertical rail */}
        <div className="absolute left-[7px] top-3 bottom-3 w-px bg-gradient-to-b from-signal/40 via-white/10 to-white/5" />
        <div className="space-y-2">
          {LIFECYCLE.map((step, i) => (
            <Reveal key={step.n} delay={i * 0.05}>
              <div className="relative grid grid-cols-[auto_1fr] gap-x-6 rounded-[8px] py-5 pl-0 pr-4 transition-colors hover:bg-white/[0.02]">
                {/* node */}
                <div className="relative z-10 mt-1.5 flex h-[15px] w-[15px] items-center justify-center">
                  <span className="h-[15px] w-[15px] rounded-full border border-signal/50 bg-ink" />
                  <span className="absolute h-1.5 w-1.5 rounded-full bg-signal" />
                </div>

                <div className="grid gap-x-8 gap-y-3 md:grid-cols-[200px_1fr] md:items-baseline">
                  <div className="flex items-center gap-3">
                    <span className="font-mono text-[12px] text-mute-soft">
                      {step.n}
                    </span>
                    <div>
                      <div className="font-display text-base font-semibold text-white">
                        {step.title}
                      </div>
                      <div className="font-mono text-[10px] uppercase tracking-wider text-mute-soft">
                        {step.agent}
                      </div>
                    </div>
                  </div>
                  <div>
                    <p className="max-w-2xl text-[14px] leading-relaxed text-mute">
                      {step.body}
                    </p>
                    <span
                      className={cn(
                        "mt-3 inline-block rounded-[4px] border px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider",
                        statusPill[step.status]
                      )}
                    >
                      {step.status.replace("_", " ")}
                    </span>
                  </div>
                </div>
              </div>
            </Reveal>
          ))}
        </div>
      </div>
    </Section>
  );
}

/* ──────────────────────────────────────────────────────────────
   AGENTS — the five
   ────────────────────────────────────────────────────────────── */
const AGENTS = [
  {
    n: "01",
    name: "Sentinel",
    role: "Anomaly Detection",
    dot: "bg-blue-400",
    body: "Scores every 30-point metric window through a PyTorch LSTM autoencoder, fused with a log classifier (0.7 / 0.3). Opens an incident above the 0.60 threshold.",
    io: "metric window → anomaly_score, AnomalyAlert",
    tags: ["PyTorch", "LSTM autoencoder"],
  },
  {
    n: "02",
    name: "Detective",
    role: "Root Cause Analysis",
    dot: "bg-amber-400",
    body: "Embeds the alert, retrieves the top-3 most similar past incidents from Qdrant, and reasons a root cause with an LLM — getting sharper as the corpus grows.",
    io: "alert → root_cause, confidence",
    tags: ["RAG", "Qdrant", "LLM"],
  },
  {
    n: "03",
    name: "Oracle",
    role: "Blast Radius & Planning",
    dot: "bg-purple-400",
    body: "Predicts which downstream services are at risk and drafts one to three remediation actions, each with a target, a kubectl command, and a risk level.",
    io: "root cause → blast_radius, proposed_actions",
    tags: ["LLM", "dependency reasoning"],
  },
  {
    n: "04",
    name: "Surgeon",
    role: "Remediation",
    dot: "bg-emerald-400",
    body: "The only agent that touches infrastructure. Executes approved actions via kubectl (dry-run in this build) and stops on the first failure. Never runs unapproved.",
    io: "approval + actions → executed_actions",
    tags: ["kubectl", "HITL-gated"],
  },
  {
    n: "05",
    name: "Scribe",
    role: "Post-Mortem & Memory",
    dot: "bg-cyan-400",
    body: "Writes a structured Markdown post-mortem, then embeds it into Qdrant so the Detective inherits the lesson. This closes the self-improvement loop.",
    io: "full incident → report, vector",
    tags: ["LLM", "embeddings", "Qdrant"],
  },
];

export function Agents() {
  return (
    <Section id="agents">
      <Reveal>
        <Eyebrow index="02">The Agents</Eyebrow>
        <h2 className="mt-6 max-w-2xl font-display text-3xl font-semibold leading-[1.1] tracking-[-0.02em] text-white text-balance sm:text-[2.6rem]">
          Five agents, one objective — orchestrated by LangGraph.
        </h2>
      </Reveal>

      <div className="mt-14 divide-y divide-white/8 border-y border-white/8">
        {AGENTS.map((a, i) => (
          <Reveal key={a.name} delay={i * 0.04}>
            <div className="group grid gap-x-10 gap-y-4 py-7 md:grid-cols-[280px_1fr] transition-colors hover:bg-white/[0.015]">
              {/* identity */}
              <div className="flex items-start gap-4">
                <span className="font-mono text-[12px] text-mute-soft pt-1.5">
                  {a.n}
                </span>
                <div>
                  <div className="flex items-center gap-2.5">
                    <span className={cn("h-2 w-2 rounded-full", a.dot)} />
                    <h3 className="font-display text-xl font-semibold text-white">
                      {a.name}
                    </h3>
                  </div>
                  <p className="mt-1 font-mono text-[10px] uppercase tracking-[0.16em] text-mute-soft">
                    {a.role}
                  </p>
                </div>
              </div>

              {/* details */}
              <div>
                <p className="max-w-2xl text-[14.5px] leading-relaxed text-mute">
                  {a.body}
                </p>
                <div className="mt-4 flex flex-wrap items-center gap-x-5 gap-y-3">
                  <code className="rounded-[4px] border border-white/8 bg-ink px-2.5 py-1 font-mono text-[11px] text-white/70">
                    {a.io}
                  </code>
                  <div className="flex flex-wrap gap-1.5">
                    {a.tags.map((t) => (
                      <span
                        key={t}
                        className="font-mono text-[10px] uppercase tracking-wider text-mute-soft"
                      >
                        {t}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </Reveal>
        ))}
      </div>
    </Section>
  );
}

/* ──────────────────────────────────────────────────────────────
   ARCHITECTURE — layered system diagram
   ────────────────────────────────────────────────────────────── */
function ArchBand({
  label,
  children,
  accent,
}: {
  label: string;
  children: React.ReactNode;
  accent?: boolean;
}) {
  return (
    <div
      className={cn(
        "grid grid-cols-1 gap-3 rounded-[8px] border bg-ink-raised/40 p-4 sm:grid-cols-[140px_1fr] sm:items-center",
        accent ? "border-signal/25" : "border-white/8"
      )}
    >
      <span
        className={cn(
          "font-mono text-[10px] uppercase tracking-[0.18em]",
          accent ? "text-signal" : "text-mute-soft"
        )}
      >
        {label}
      </span>
      <div className="flex flex-wrap items-center gap-2">{children}</div>
    </div>
  );
}

function Chip({ children, dim }: { children: React.ReactNode; dim?: boolean }) {
  return (
    <span
      className={cn(
        "rounded-[5px] border px-2.5 py-1.5 font-mono text-[11px]",
        dim
          ? "border-white/8 text-mute"
          : "border-white/12 bg-ink text-white/80"
      )}
    >
      {children}
    </span>
  );
}

function Connector({ note }: { note?: string }) {
  return (
    <div className="flex items-center gap-3 py-2 pl-2 sm:pl-[60px]">
      <div className="h-6 w-px bg-white/12" />
      {note && (
        <span className="font-mono text-[10px] text-mute-soft">{note}</span>
      )}
    </div>
  );
}

export function Architecture() {
  return (
    <Section id="architecture">
      <Reveal>
        <Eyebrow index="03">Architecture</Eyebrow>
        <h2 className="mt-6 max-w-2xl font-display text-3xl font-semibold leading-[1.1] tracking-[-0.02em] text-white text-balance sm:text-[2.6rem]">
          A detection model, an agent graph, and a memory.
        </h2>
      </Reveal>

      <Reveal delay={0.1} className="mt-12">
        <div className="rounded-[12px] border border-white/8 bg-grid-sm p-4 sm:p-6">
          <ArchBand label="Signal sources">
            <Chip dim>Prometheus</Chip>
            <Chip dim>K3s metrics</Chip>
            <Chip dim>Metric simulator</Chip>
          </ArchBand>
          <Connector note="POST /api/v1/metrics/ingest" />

          <ArchBand label="Gateway">
            <Chip>FastAPI</Chip>
            <Chip dim>/metrics/ingest</Chip>
            <Chip dim>/incidents</Chip>
            <Chip dim>/approve</Chip>
          </ArchBand>
          <Connector note="30-point window" />

          <ArchBand label="Detection" accent>
            <Chip>Sentinel</Chip>
            <span className="font-mono text-[11px] text-mute">
              PyTorch LSTM autoencoder · score &gt; 0.60 → incident
            </span>
          </ArchBand>
          <Connector note="initial state · thread_id = incident_id" />

          <ArchBand label="Supervisor" accent>
            <div className="flex flex-wrap items-center gap-1.5">
              {["Detective", "Oracle"].map((n) => (
                <Chip key={n}>{n}</Chip>
              ))}
              <span className="flex items-center gap-1 rounded-[5px] border border-amber-400/30 bg-amber-400/5 px-2.5 py-1.5 font-mono text-[11px] text-amber-300">
                <Pause className="h-3 w-3" /> HITL
              </span>
              {["Surgeon", "Scribe"].map((n) => (
                <Chip key={n}>{n}</Chip>
              ))}
            </div>
          </ArchBand>

          <div className="mt-3 grid gap-3 sm:grid-cols-2">
            <ArchBand label="Memory">
              <Chip dim>Qdrant vectors</Chip>
              <Chip dim>SQLite checkpoints</Chip>
            </ArchBand>
            <ArchBand label="Surface">
              <Chip dim>Next.js console</Chip>
              <Chip dim>live polling</Chip>
            </ArchBand>
          </div>

          <p className="mt-5 font-mono text-[10.5px] leading-relaxed text-mute-soft">
            LangGraph orchestrates the four agent nodes over one shared state
            object. The graph pauses at a static interrupt before Surgeon and
            checkpoints to SQLite, so an incident can wait for human approval
            and resume exactly where it left off.
          </p>
        </div>
      </Reveal>
    </Section>
  );
}

/* ──────────────────────────────────────────────────────────────
   RAG LOOP — self improvement
   ────────────────────────────────────────────────────────────── */
const LOOP = [
  { k: "Resolve", v: "An incident is diagnosed and closed." },
  { k: "Embed", v: "Scribe writes a post-mortem and stores its vector in Qdrant." },
  { k: "Retrieve", v: "The next similar alert pulls that incident back as context." },
  { k: "Improve", v: "Detective reasons from precedent — faster, more accurate." },
];

export function RagLoop() {
  return (
    <Section id="rag">
      <Reveal>
        <Eyebrow index="04">Self-Improving</Eyebrow>
        <h2 className="mt-6 max-w-2xl font-display text-3xl font-semibold leading-[1.1] tracking-[-0.02em] text-white text-balance sm:text-[2.6rem]">
          Every incident makes the next one easier to solve.
        </h2>
        <p className="mt-5 max-w-xl text-[15px] leading-relaxed text-mute">
          The knowledge base starts empty and fills itself. There is no corpus
          to curate — the system writes its own, one resolution at a time. This
          is progressive knowledge accumulation, and it compounds.
        </p>
      </Reveal>

      <Reveal delay={0.1} className="mt-12">
        <div className="grid gap-px overflow-hidden rounded-[10px] border border-white/8 bg-white/8 sm:grid-cols-4">
          {LOOP.map((s, i) => (
            <div key={s.k} className="relative bg-ink-raised/60 p-5">
              <div className="flex items-center justify-between">
                <span className="font-mono text-[11px] uppercase tracking-[0.16em] text-signal">
                  {s.k}
                </span>
                <span className="font-mono text-[11px] text-mute-soft">
                  {String(i + 1).padStart(2, "0")}
                </span>
              </div>
              <p className="mt-3 text-[13px] leading-relaxed text-mute">{s.v}</p>
              {i === LOOP.length - 1 && (
                <span className="absolute right-4 top-5 font-mono text-[11px] text-signal">
                  ↻
                </span>
              )}
            </div>
          ))}
        </div>
        <p className="mt-4 text-center font-mono text-[10.5px] uppercase tracking-[0.18em] text-mute-soft">
          Scribe → Qdrant → Detective → repeat
        </p>
      </Reveal>
    </Section>
  );
}

/* ──────────────────────────────────────────────────────────────
   HITL — human in the loop
   ────────────────────────────────────────────────────────────── */
export function Hitl() {
  return (
    <Section id="hitl" className="grid gap-12 lg:grid-cols-[1.05fr_0.95fr] lg:items-center lg:gap-20">
      <Reveal>
        <Eyebrow index="05">Safety</Eyebrow>
        <h2 className="mt-6 font-display text-3xl font-semibold leading-[1.1] tracking-[-0.02em] text-white text-balance sm:text-[2.6rem]">
          Nothing reaches production without a human.
        </h2>
        <p className="mt-5 max-w-lg text-[15px] leading-relaxed text-mute">
          Autonomy without a brake is a liability. NexusOps reasons and plans on
          its own, but the Surgeon — the only agent that can change
          infrastructure — sits behind a hard gate. The graph literally stops
          and persists its state until you decide.
        </p>
        <ul className="mt-7 space-y-3">
          {[
            "Static interrupt before any remediation runs",
            "Full plan, blast radius, and commands shown before approval",
            "Dry-run execution mode — validate without applying",
            "Reject path is first-class: dismiss noise, keep the audit trail",
          ].map((t) => (
            <li key={t} className="flex items-start gap-3">
              <Check className="mt-0.5 h-4 w-4 shrink-0 text-signal" />
              <span className="text-[14px] text-mute">{t}</span>
            </li>
          ))}
        </ul>
      </Reveal>

      {/* approval card mock */}
      <Reveal delay={0.1}>
        <div className="rounded-[10px] border border-amber-400/20 bg-amber-400/[0.03] p-5">
          <div className="flex items-center justify-between">
            <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-amber-300">
              Awaiting approval
            </span>
            <span className="font-mono text-[10px] text-mute-soft">#a8c3e1b7</span>
          </div>
          <div className="mt-3 flex items-center gap-2">
            <span className="font-mono text-sm text-white">order-service</span>
            <span className="rounded bg-white/5 px-1.5 py-0.5 font-mono text-[10px] text-mute-soft">
              prod
            </span>
          </div>
          <p className="mt-2 text-[13px] leading-relaxed text-mute">
            Memory leak in the order-processing loop. OOM kill expected in ~20
            minutes. Two downstream services at risk.
          </p>
          <div className="mt-4 space-y-1.5">
            {[
              "kubectl rollout restart deploy/order-service -n prod",
              "kubectl scale deploy/order-service --replicas=5 -n prod",
            ].map((c) => (
              <code
                key={c}
                className="block rounded-[4px] border border-white/8 bg-ink px-2.5 py-1.5 font-mono text-[10.5px] text-white/70"
              >
                $ {c}
              </code>
            ))}
          </div>
          <div className="mt-4 flex gap-2">
            <span className="flex-1 rounded-[5px] bg-signal py-2 text-center font-mono text-[11px] font-semibold uppercase tracking-wider text-[#0a0c05]">
              Approve
            </span>
            <span className="rounded-[5px] border border-white/12 px-4 py-2 text-center font-mono text-[11px] font-semibold uppercase tracking-wider text-mute">
              Dismiss
            </span>
          </div>
        </div>
      </Reveal>
    </Section>
  );
}

/* ──────────────────────────────────────────────────────────────
   STACK
   ────────────────────────────────────────────────────────────── */
const STACK: { cat: string; items: string[] }[] = [
  { cat: "Detection", items: ["PyTorch — LSTM autoencoder", "TensorFlow — log classifier", "NumPy"] },
  { cat: "Agents", items: ["LangGraph — orchestration", "LangChain — LLM plumbing", "OpenAI / Groq"] },
  { cat: "Memory", items: ["Qdrant — vector store", "sentence-transformers", "all-MiniLM-L6-v2 · 384d"] },
  { cat: "Backend", items: ["FastAPI", "Pydantic", "SQLite — graph checkpoints"] },
  { cat: "Frontend", items: ["Next.js 16", "Tailwind CSS", "Framer Motion · Recharts"] },
  { cat: "Infra", items: ["Kubernetes (K3s)", "Prometheus", "Docker Compose"] },
];

export function Stack() {
  return (
    <Section id="stack">
      <Reveal>
        <Eyebrow index="06">Stack</Eyebrow>
        <h2 className="mt-6 max-w-2xl font-display text-3xl font-semibold leading-[1.1] tracking-[-0.02em] text-white text-balance sm:text-[2.6rem]">
          Built on a serious, boring-on-purpose stack.
        </h2>
      </Reveal>

      <div className="mt-12 grid gap-px overflow-hidden rounded-[10px] border border-white/8 bg-white/8 sm:grid-cols-2 lg:grid-cols-3">
        {STACK.map((s, i) => (
          <Reveal key={s.cat} delay={(i % 3) * 0.05}>
            <div className="h-full bg-ink-raised/60 p-5">
              <span className="font-mono text-[10px] uppercase tracking-[0.2em] text-signal">
                {s.cat}
              </span>
              <ul className="mt-3 space-y-1.5">
                {s.items.map((it) => (
                  <li key={it} className="font-mono text-[12px] text-mute">
                    {it}
                  </li>
                ))}
              </ul>
            </div>
          </Reveal>
        ))}
      </div>
    </Section>
  );
}

/* ──────────────────────────────────────────────────────────────
   CTA
   ────────────────────────────────────────────────────────────── */
export function CTA() {
  return (
    <Section className="py-24 sm:py-32">
      <div className="relative overflow-hidden rounded-[16px] border border-white/10 bg-ink-raised/50 px-6 py-16 text-center sm:px-12 sm:py-20">
        <div className="pointer-events-none absolute inset-0 bg-grid-sm mask-fade-edges opacity-50" />
        <div className="pointer-events-none absolute left-1/2 top-0 h-64 w-[520px] -translate-x-1/2 signal-glow blur-3xl opacity-60" />
        <div className="relative">
          <motion.h2
            initial={{ opacity: 0, y: 12 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="mx-auto max-w-2xl font-display text-3xl font-semibold leading-[1.08] tracking-[-0.02em] text-white text-balance sm:text-[2.9rem]"
          >
            Watch an incident detect, diagnose, and heal — live.
          </motion.h2>
          <p className="mx-auto mt-5 max-w-md text-[15px] leading-relaxed text-mute">
            Open the console, trigger an anomaly, and walk the full pipeline
            from signal to post-mortem.
          </p>
          <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
            <Link
              href="/dashboard"
              className="group inline-flex items-center gap-1.5 rounded-[6px] bg-signal px-6 py-3.5 font-mono text-[12.5px] font-semibold uppercase tracking-wider text-[#0a0c05] transition-transform hover:-translate-y-0.5"
            >
              Launch the console
              <ArrowUpRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
            </Link>
            <Link
              href="/incidents"
              className="inline-flex items-center gap-1.5 rounded-[6px] border border-white/12 px-6 py-3.5 font-mono text-[12.5px] font-semibold uppercase tracking-wider text-white/80 transition-colors hover:bg-white/[0.04]"
            >
              Browse incidents
            </Link>
          </div>
        </div>
      </div>
    </Section>
  );
}
