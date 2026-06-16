import Link from "next/link";
import { LogoMark } from "./logo-mark";

const COLUMNS: { title: string; links: { label: string; href: string }[] }[] = [
  {
    title: "Product",
    links: [
      { label: "Overview", href: "#overview" },
      { label: "Incident Lifecycle", href: "#lifecycle" },
      { label: "The Agents", href: "#agents" },
      { label: "Launch Console", href: "/dashboard" },
    ],
  },
  {
    title: "Engineering",
    links: [
      { label: "Architecture", href: "#architecture" },
      { label: "Self-improving RAG", href: "#rag" },
      { label: "Human-in-the-loop", href: "#hitl" },
      { label: "Tech Stack", href: "#stack" },
    ],
  },
  {
    title: "Console",
    links: [
      { label: "Dashboard", href: "/dashboard" },
      { label: "Incidents", href: "/incidents" },
      { label: "Pipeline", href: "/pipeline" },
      { label: "Agents", href: "/agents" },
    ],
  },
];

export function MarketingFooter() {
  return (
    <footer className="relative border-t border-white/8 bg-ink">
      <div className="mx-auto max-w-[1200px] px-5 sm:px-8 py-16">
        <div className="grid gap-12 lg:grid-cols-[1.4fr_1fr_1fr_1fr]">
          {/* Brand block */}
          <div>
            <div className="flex items-center gap-2.5">
              <LogoMark />
              <span className="font-display text-[17px] font-semibold tracking-tight text-white">
                NexusOps
              </span>
            </div>
            <p className="mt-4 max-w-xs text-sm leading-relaxed text-mute">
              An autonomous Site Reliability Engineer. Five agents that detect,
              diagnose, and remediate infrastructure incidents — supervised by a
              human, improving with every resolution.
            </p>
            <div className="mt-5 inline-flex items-center gap-2 rounded-[5px] border border-white/8 px-3 py-1.5">
              <span className="relative flex h-1.5 w-1.5">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-signal opacity-60" />
                <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-signal" />
              </span>
              <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-mute">
                All systems nominal
              </span>
            </div>
          </div>

          {COLUMNS.map((col) => (
            <div key={col.title}>
              <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-mute-soft">
                {col.title}
              </p>
              <ul className="mt-4 space-y-2.5">
                {col.links.map((l) => (
                  <li key={l.label}>
                    <Link
                      href={l.href}
                      className="text-sm text-mute hover:text-white transition-colors"
                    >
                      {l.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <div className="mt-14 flex flex-col gap-4 border-t border-white/8 pt-6 sm:flex-row sm:items-center sm:justify-between">
          <p className="font-mono text-[11px] tracking-wide text-mute-soft">
            © {new Date().getFullYear()} NexusOps · Autonomous AI SRE
          </p>
          <p className="font-mono text-[11px] tracking-wide text-mute-soft">
            FastAPI · LangGraph · LangChain · PyTorch · Qdrant · Next.js
          </p>
        </div>
      </div>
    </footer>
  );
}
