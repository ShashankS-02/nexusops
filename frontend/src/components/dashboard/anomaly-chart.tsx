"use client";

import { motion } from "framer-motion";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

// Simulated anomaly score data over time
const data = Array.from({ length: 30 }, (_, i) => {
  const base = Math.sin(i * 0.3) * 0.1 + 0.15;
  const spike = i >= 22 && i <= 26 ? (i - 22) * 0.15 : 0;
  return {
    time: `${String(i).padStart(2, "0")}:00`,
    score: Math.min(1, Math.max(0, base + spike + Math.random() * 0.05)),
    threshold: 0.6,
  };
});

function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  const score = payload[0]?.value;
  const isAnomaly = score > 0.6;
  return (
    <div className="bg-[#15171a] border border-white/10 rounded-lg px-3 py-2 shadow-xl">
      <p className="text-[10px] text-muted-foreground">{label}</p>
      <p
        className={`text-sm font-mono font-bold ${
          isAnomaly ? "text-red-400" : "text-emerald-400"
        }`}
      >
        Score: {score?.toFixed(4)}
      </p>
      {isAnomaly && (
        <p className="text-[10px] text-red-400 mt-0.5">Above threshold</p>
      )}
    </div>
  );
}

export function AnomalyChart() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.5 }}
      className="rounded-xl border border-white/8 bg-[#0e0f11] p-6"
    >
      <div className="flex items-center justify-between mb-5">
        <div>
          <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-widest">
            Anomaly Score Timeline
          </h3>
          <p className="text-xs text-muted-foreground mt-1">
            LSTM Autoencoder reconstruction error (threshold: 0.60)
          </p>
        </div>
        <div className="flex items-center gap-4 text-xs">
          <div className="flex items-center gap-1.5">
            <div className="w-2.5 h-2.5 rounded-full bg-signal" />
            <span className="text-muted-foreground">Score</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-2.5 h-0.5 bg-red-500/60" />
            <span className="text-muted-foreground">Threshold</span>
          </div>
        </div>
      </div>

      <div className="h-[200px]">
        <ResponsiveContainer width="100%" height={200}>
          <AreaChart data={data}>
            <defs>
              <linearGradient id="scoreGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#c6f24e" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#c6f24e" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="rgba(255,255,255,0.03)"
            />
            <XAxis
              dataKey="time"
              stroke="rgba(255,255,255,0.1)"
              tick={{ fill: "rgba(255,255,255,0.3)", fontSize: 10 }}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              domain={[0, 1]}
              stroke="rgba(255,255,255,0.1)"
              tick={{ fill: "rgba(255,255,255,0.3)", fontSize: 10 }}
              tickLine={false}
              axisLine={false}
            />
            <Tooltip content={<CustomTooltip />} />
            <Area
              type="monotone"
              dataKey="threshold"
              stroke="#ef4444"
              strokeWidth={1}
              strokeDasharray="4 4"
              fill="none"
              dot={false}
            />
            <Area
              type="monotone"
              dataKey="score"
              stroke="#c6f24e"
              strokeWidth={2}
              fill="url(#scoreGradient)"
              dot={false}
              animationDuration={2000}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </motion.div>
  );
}
