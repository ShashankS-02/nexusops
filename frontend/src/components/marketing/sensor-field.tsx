"use client";

import { useEffect, useRef } from "react";

/**
 * Interactive "sensor field" — a grid of monitoring nodes. The cursor acts as
 * a probe: nearby nodes light up lime, hairlines connect to it, and a crosshair
 * reticle tracks the pointer. When the pointer leaves, a virtual probe drifts
 * so the field still feels alive (and works with no pointer on touch devices).
 *
 * Deliberately NOT the generic "spotlight that follows the cursor" — this is
 * themed to anomaly detection: you are inspecting a live system.
 */
export function SensorField({ className }: { className?: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvasEl = canvasRef.current;
    const parentEl = canvasEl?.parentElement;
    const ctx = canvasEl?.getContext("2d");
    if (!canvasEl || !parentEl || !ctx) return;

    // Non-null aliases so the render loop's closures type-check cleanly.
    const canvas: HTMLCanvasElement = canvasEl;
    const parent: HTMLElement = parentEl;
    const c: CanvasRenderingContext2D = ctx;

    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);

    const SPACING = 34;
    const RADIUS = 150; // influence radius
    const LINK = 118; // connector radius
    const LIME = "198,242,78";

    let w = 0;
    let h = 0;
    let raf = 0;
    let t = 0;
    let running = true;
    const mouse = { x: -9999, y: -9999, active: false };

    function resize() {
      const rect = parent.getBoundingClientRect();
      w = rect.width;
      h = rect.height;
      canvas.width = Math.max(1, Math.floor(w * dpr));
      canvas.height = Math.max(1, Math.floor(h * dpr));
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
      c.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    resize();

    const ro = new ResizeObserver(resize);
    ro.observe(parent);

    function onMove(e: PointerEvent) {
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      if (x >= 0 && x <= w && y >= 0 && y <= h) {
        mouse.x = x;
        mouse.y = y;
        mouse.active = true;
      } else {
        mouse.active = false;
      }
    }
    window.addEventListener("pointermove", onMove, { passive: true });

    function drawStatic() {
      c.clearRect(0, 0, w, h);
      for (let x = SPACING / 2; x < w; x += SPACING) {
        for (let y = SPACING / 2; y < h; y += SPACING) {
          c.beginPath();
          c.arc(x, y, 0.7, 0, Math.PI * 2);
          c.fillStyle = "rgba(255,255,255,0.06)";
          c.fill();
        }
      }
    }

    function frame() {
      raf = 0;
      if (!running) return;
      t += 0.016;
      c.clearRect(0, 0, w, h);

      // Real probe when hovering, drifting virtual probe otherwise
      let mx = mouse.x;
      let my = mouse.y;
      if (!mouse.active) {
        mx = w * (0.5 + 0.34 * Math.cos(t * 0.22));
        my = h * (0.46 + 0.3 * Math.sin(t * 0.29));
      }

      for (let x = SPACING / 2; x < w; x += SPACING) {
        for (let y = SPACING / 2; y < h; y += SPACING) {
          const dx = x - mx;
          const dy = y - my;
          const d = Math.hypot(dx, dy);
          const f = Math.max(0, 1 - d / RADIUS);

          if (f > 0.02) {
            c.beginPath();
            c.arc(x, y, 0.8 + f * 2.1, 0, Math.PI * 2);
            c.fillStyle = `rgba(${LIME},${0.12 + f * 0.72})`;
            c.fill();
          } else {
            c.beginPath();
            c.arc(x, y, 0.7, 0, Math.PI * 2);
            c.fillStyle = "rgba(255,255,255,0.05)";
            c.fill();
          }

          if (d < LINK) {
            c.beginPath();
            c.moveTo(mx, my);
            c.lineTo(x, y);
            c.strokeStyle = `rgba(${LIME},${(1 - d / LINK) * 0.28})`;
            c.lineWidth = 0.6;
            c.stroke();
          }
        }
      }

      // Crosshair reticle at the probe
      c.strokeStyle = `rgba(${LIME},0.7)`;
      c.lineWidth = 1;
      c.beginPath();
      c.arc(mx, my, 5, 0, Math.PI * 2);
      c.stroke();
      c.strokeStyle = `rgba(${LIME},0.45)`;
      c.beginPath();
      c.moveTo(mx - 11, my);
      c.lineTo(mx - 6, my);
      c.moveTo(mx + 6, my);
      c.lineTo(mx + 11, my);
      c.moveTo(mx, my - 11);
      c.lineTo(mx, my - 6);
      c.moveTo(mx, my + 6);
      c.lineTo(mx, my + 11);
      c.stroke();

      raf = requestAnimationFrame(frame);
    }

    // Pause the loop when the hero scrolls out of view
    const io = new IntersectionObserver(
      ([entry]) => {
        running = entry.isIntersecting;
        if (running && !reduce && !raf) raf = requestAnimationFrame(frame);
      },
      { threshold: 0 }
    );
    io.observe(parent);

    if (reduce) {
      drawStatic();
    } else {
      raf = requestAnimationFrame(frame);
    }

    return () => {
      if (raf) cancelAnimationFrame(raf);
      ro.disconnect();
      io.disconnect();
      window.removeEventListener("pointermove", onMove);
    };
  }, []);

  return <canvas ref={canvasRef} className={className} aria-hidden />;
}
