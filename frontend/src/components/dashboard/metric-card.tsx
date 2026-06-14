"use client";

import { motion, useSpring, useTransform, useMotionValueEvent } from "framer-motion";
import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";
import type { LucideIcon } from "lucide-react";

const iconBgMap: Record<string, string> = {
  "bg-red-500": "bg-red-500/10",
  "bg-emerald-500": "bg-emerald-500/10",
  "bg-blue-500": "bg-blue-500/10",
  "bg-purple-500": "bg-purple-500/10",
  "bg-amber-500": "bg-amber-500/10",
  "bg-cyan-500": "bg-cyan-500/10",
};

const iconTextMap: Record<string, string> = {
  "bg-red-500": "text-red-500",
  "bg-emerald-500": "text-emerald-500",
  "bg-blue-500": "text-blue-500",
  "bg-purple-500": "text-purple-500",
  "bg-amber-500": "text-amber-500",
  "bg-cyan-500": "text-cyan-500",
};

interface MetricCardProps {
  title: string;
  value: number;
  suffix?: string;
  trend?: "up" | "down" | "flat";
  trendValue?: string;
  icon: LucideIcon;
  color: string;
  delay?: number;
}

function AnimatedNumber({ value, suffix = "" }: { value: number; suffix?: string }) {
  const spring = useSpring(0, { bounce: 0, duration: 1500 });
  const [displayValue, setDisplayValue] = useState("0");

  useMotionValueEvent(spring, "change", (latest) => {
    setDisplayValue(latest % 1 === 0 ? latest.toFixed(0) : latest.toFixed(2));
  });

  useEffect(() => {
    spring.set(value);
  }, [spring, value]);

  return (
    <span className="font-mono text-3xl font-bold tracking-tight">
      {displayValue}
      {suffix && <span className="text-lg text-muted-foreground ml-1">{suffix}</span>}
    </span>
  );
}

export function MetricCard({
  title,
  value,
  suffix,
  trend = "flat",
  trendValue,
  icon: Icon,
  color,
  delay = 0,
}: MetricCardProps) {
  const trendIcon = {
    up: TrendingUp,
    down: TrendingDown,
    flat: Minus,
  }[trend];
  const TrendIcon = trendIcon;

  const trendColor = {
    up: "text-emerald-400",
    down: "text-red-400",
    flat: "text-muted-foreground",
  }[trend];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: delay * 0.1, duration: 0.5, ease: "easeOut" }}
      whileHover={{ scale: 1.02, transition: { duration: 0.2 } }}
      className="relative overflow-hidden rounded-xl border border-white/8 bg-[#0e0f11] p-5"
    >
      {/* Gradient glow */}
      <div
        className={cn(
          "absolute top-0 right-0 w-32 h-32 rounded-full blur-3xl opacity-10",
          color
        )}
      />

      <div className="relative z-10">
        <div className="flex items-center justify-between mb-4">
          <span className="text-sm text-muted-foreground font-medium">
            {title}
          </span>
          <div
            className={cn(
              "w-8 h-8 rounded-lg flex items-center justify-center",
              iconBgMap[color] ?? "bg-white/10"
            )}
          >
            <Icon className={cn("w-4 h-4", iconTextMap[color] ?? "text-white")} />
          </div>
        </div>

        <AnimatedNumber value={value} suffix={suffix} />

        {trendValue && (
          <div className="flex items-center gap-1.5 mt-3">
            <TrendIcon className={cn("w-3.5 h-3.5", trendColor)} />
            <span className={cn("text-xs font-medium", trendColor)}>
              {trendValue}
            </span>
            <span className="text-xs text-muted-foreground">vs last hour</span>
          </div>
        )}
      </div>
    </motion.div>
  );
}
