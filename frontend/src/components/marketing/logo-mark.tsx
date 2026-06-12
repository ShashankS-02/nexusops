import { cn } from "@/lib/utils";

/**
 * NexusOps mark — a hexagonal sentinel node with a live signal core.
 * Hairline geometry + a single lime pulse. Intentionally not a generic
 * rounded-gradient app icon.
 */
export function LogoMark({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      className={cn("h-6 w-6", className)}
      aria-hidden="true"
    >
      <path
        d="M12 1.6 21 6.8V17.2L12 22.4 3 17.2V6.8L12 1.6Z"
        stroke="currentColor"
        strokeWidth="1.1"
        strokeLinejoin="round"
        className="text-white/40"
      />
      <path
        d="M12 6.4 16.5 9v6L12 17.6 7.5 15V9L12 6.4Z"
        stroke="var(--color-signal)"
        strokeWidth="1.1"
        strokeLinejoin="round"
        opacity="0.5"
      />
      <circle cx="12" cy="12" r="2.1" fill="var(--color-signal)" />
    </svg>
  );
}
