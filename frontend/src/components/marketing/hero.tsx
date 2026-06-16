"use client";

import Link from "next/link";
import { motion, useSpring } from "framer-motion";
import { ArrowUpRight, ArrowDown } from "lucide-react";
import { LiveConsole } from "./live-console";
import { SensorField } from "./sensor-field";

export function Hero() {
  // Subtle depth parallax on the console, driven by cursor position
  const px = useSpring(0, { stiffness: 70, damping: 22 });
  const py = useSpring(0, { stiffness: 70, damping: 22 });

  function handlePointer(e: React.PointerEvent<HTMLElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const nx = (e.clientX - rect.left) / rect.width - 0.5;
    const ny = (e.clientY - rect.top) / rect.height - 0.5;
    px.set(-nx * 16);
    py.set(-ny * 14);
  }

  return (
    <section
      id="overview"
      onPointerMove={handlePointer}
      className="relative overflow-hidden"
    >
      {/* interactive sensor field + asymmetric glow */}
      <SensorField className="pointer-events-none absolute inset-0 z-0 h-full w-full mask-fade-edges" />
      <div
        className="pointer-events-none absolute -top-40 right-[-10%] z-0 h-[480px] w-[480px] rounded-full signal-glow blur-3xl opacity-50"
        aria-hidden
      />

      <div className="relative z-[2] mx-auto max-w-[1200px] px-5 sm:px-8 pt-32 pb-20 lg:pt-40 lg:pb-28">
        <div className="grid items-center gap-14 lg:grid-cols-[1.04fr_0.96fr]">
          {/* Left: copy */}
          <div>
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.02] px-3 py-1"
            >
              <span className="relative flex h-1.5 w-1.5">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-signal opacity-60" />
                <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-signal" />
              </span>
              <span className="font-mono text-[10.5px] uppercase tracking-[0.22em] text-mute">
                Autonomous AI SRE · v0.1
              </span>
            </motion.div>

            <motion.h1
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.05 }}
              className="mt-6 font-display text-[2.6rem] leading-[1.04] font-semibold tracking-[-0.025em] text-white text-balance sm:text-6xl lg:text-[4.1rem]"
            >
              An SRE that never sleeps —
              <br className="hidden sm:block" />{" "}
              and never acts{" "}
              <span className="relative whitespace-nowrap text-signal">
                without approval
                <svg
                  className="absolute -bottom-1 left-0 w-full"
                  viewBox="0 0 200 8"
                  fill="none"
                  preserveAspectRatio="none"
                >
                  <path
                    d="M1 5.5 Q 50 1.5 100 4 T 199 3"
                    stroke="var(--color-signal)"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    opacity="0.7"
                  />
                </svg>
              </span>
              .
            </motion.h1>

            <motion.p
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.12 }}
              className="mt-6 max-w-xl text-[15px] leading-relaxed text-mute sm:text-base"
            >
              NexusOps runs a five-agent LangGraph pipeline over your live
              metrics — detecting anomalies with a PyTorch LSTM, tracing root
              cause with RAG, predicting blast radius, and remediating
              Kubernetes incidents behind a human-in-the-loop gate.
            </motion.p>

            <motion.div
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.18 }}
              className="mt-8 flex flex-wrap items-center gap-3"
            >
              <Link
                href="/dashboard"
                className="group inline-flex items-center gap-1.5 rounded-[6px] bg-signal px-5 py-3 font-mono text-[12.5px] font-semibold uppercase tracking-wider text-[#0a0c05] transition-transform hover:-translate-y-0.5"
              >
                Launch the console
                <ArrowUpRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
              </Link>
              <a
                href="#lifecycle"
                className="group inline-flex items-center gap-1.5 rounded-[6px] border border-white/12 px-5 py-3 font-mono text-[12.5px] font-semibold uppercase tracking-wider text-white/80 transition-colors hover:bg-white/[0.04] hover:text-white"
              >
                See how it works
                <ArrowDown className="h-4 w-4 transition-transform group-hover:translate-y-0.5" />
              </a>
            </motion.div>

            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.6, delay: 0.28 }}
              className="mt-10 flex flex-wrap items-center gap-x-5 gap-y-2 font-mono text-[11px] uppercase tracking-wider text-mute-soft"
            >
              <span>5 specialized agents</span>
              <span className="text-white/15">/</span>
              <span>LSTM AUROC 0.99</span>
              <span className="text-white/15">/</span>
              <span>anomaly threshold 0.60</span>
            </motion.div>
          </div>

          {/* Right: live console (parallax wrapper) */}
          <motion.div style={{ x: px, y: py }} className="will-change-transform">
            <motion.div
              initial={{ opacity: 0, y: 20, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{ duration: 0.7, delay: 0.15, ease: "easeOut" }}
            >
              <LiveConsole />
            </motion.div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
