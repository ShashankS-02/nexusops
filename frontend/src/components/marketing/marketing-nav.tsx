"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { motion, AnimatePresence } from "framer-motion";
import { ArrowUpRight, Menu, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { LogoMark } from "./logo-mark";

const LINKS = [
  { label: "Overview", href: "#overview" },
  { label: "Lifecycle", href: "#lifecycle" },
  { label: "Agents", href: "#agents" },
  { label: "Architecture", href: "#architecture" },
  { label: "Stack", href: "#stack" },
];

export function MarketingNav() {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 12);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <header
      className={cn(
        "fixed top-0 inset-x-0 z-50 transition-colors duration-300",
        scrolled
          ? "bg-ink/80 backdrop-blur-xl border-b border-white/8"
          : "bg-transparent border-b border-transparent"
      )}
    >
      <div className="mx-auto max-w-[1200px] px-5 sm:px-8">
        <div className="flex h-16 items-center justify-between">
          {/* Brand */}
          <Link href="/" className="flex items-center gap-2.5 group">
            <LogoMark />
            <div className="flex items-baseline gap-2">
              <span className="font-display text-[17px] font-semibold tracking-tight text-white">
                NexusOps
              </span>
              <span className="hidden sm:inline font-mono text-[10px] uppercase tracking-[0.2em] text-mute">
                /SRE
              </span>
            </div>
          </Link>

          {/* Center links */}
          <nav className="hidden md:flex items-center gap-1">
            {LINKS.map((l) => (
              <a
                key={l.href}
                href={l.href}
                className="px-3 py-2 font-mono text-[12px] uppercase tracking-wider text-mute hover:text-white transition-colors"
              >
                {l.label}
              </a>
            ))}
          </nav>

          {/* CTA */}
          <div className="flex items-center gap-2">
            <Link
              href="/dashboard"
              className="group hidden sm:inline-flex items-center gap-1.5 rounded-[5px] bg-signal px-3.5 py-2 font-mono text-[12px] font-semibold uppercase tracking-wider text-[#0a0c05] transition-transform hover:-translate-y-0.5"
            >
              Launch Console
              <ArrowUpRight className="h-3.5 w-3.5 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
            </Link>
            <button
              onClick={() => setOpen((v) => !v)}
              className="md:hidden p-2 text-mute hover:text-white"
              aria-label="Menu"
            >
              {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile menu */}
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="md:hidden overflow-hidden border-t border-white/8 bg-ink/95 backdrop-blur-xl"
          >
            <div className="px-5 py-4 flex flex-col gap-1">
              {LINKS.map((l) => (
                <a
                  key={l.href}
                  href={l.href}
                  onClick={() => setOpen(false)}
                  className="py-2.5 font-mono text-sm uppercase tracking-wider text-mute hover:text-white"
                >
                  {l.label}
                </a>
              ))}
              <Link
                href="/dashboard"
                className="mt-2 inline-flex items-center justify-center gap-1.5 rounded-[5px] bg-signal px-3.5 py-2.5 font-mono text-[12px] font-semibold uppercase tracking-wider text-[#0a0c05]"
              >
                Launch Console <ArrowUpRight className="h-3.5 w-3.5" />
              </Link>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  );
}
