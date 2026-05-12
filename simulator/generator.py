"""
Synthetic Metrics & Log Generator — Simulator

Generates realistic Kubernetes pod metrics with:
  - Normal baseline behavior
  - Injected anomaly scenarios (CPU spike, memory leak, cascading failure)

Usage:
    python -m simulator.generator              # stream forever
    python -m simulator.generator --scenario cpu_spike --duration 60
"""
from __future__ import annotations

import asyncio
import json
import math
import random
import time
import uuid
from datetime import datetime, timezone
from typing import Generator

import httpx
import typer
from rich.console import Console
from rich.live import Live
from rich.table import Table

from nexusops.models import LogEntry, MetricPoint, Severity

console = Console()
app = typer.Typer()

# ── Pod Definitions ────────────────────────────────────────────────────────────

PODS = [
    {"name": "payment-service-7d4f9b-xk2pq", "namespace": "prod"},
    {"name": "auth-service-6c8b7d-mn3rs", "namespace": "prod"},
    {"name": "inventory-api-5f6e8c-jt7uv", "namespace": "prod"},
    {"name": "notification-svc-9a2d1e-yw8op", "namespace": "prod"},
    {"name": "data-pipeline-8b3f5a-rz9lm", "namespace": "staging"},
]

# ── Baseline Normal Distributions ─────────────────────────────────────────────

BASELINE = {
    "cpu_usage_percent": (25.0, 8.0),        # mean, std
    "memory_usage_percent": (45.0, 10.0),
    "request_latency_ms": (120.0, 30.0),
    "error_rate_percent": (0.5, 0.3),
    "network_io_kbps": (2000.0, 400.0),
}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def generate_normal_metric(pod: dict) -> MetricPoint:
    """Generates a single normal (non-anomalous) metric reading."""
    return MetricPoint(
        timestamp=datetime.now(timezone.utc),
        pod_name=pod["name"],
        namespace=pod["namespace"],
        cpu_usage_percent=_clamp(
            random.gauss(*BASELINE["cpu_usage_percent"]), 0, 100
        ),
        memory_usage_percent=_clamp(
            random.gauss(*BASELINE["memory_usage_percent"]), 0, 100
        ),
        request_latency_ms=_clamp(
            random.gauss(*BASELINE["request_latency_ms"]), 0, 10000
        ),
        error_rate_percent=_clamp(
            random.gauss(*BASELINE["error_rate_percent"]), 0, 100
        ),
        network_io_kbps=_clamp(
            random.gauss(*BASELINE["network_io_kbps"]), 0, 100000
        ),
    )


# ── Anomaly Scenarios ─────────────────────────────────────────────────────────

def generate_cpu_spike(pod: dict, intensity: float = 1.0) -> MetricPoint:
    """Simulates a CPU spike — typical of a runaway process or traffic surge."""
    base = generate_normal_metric(pod)
    base.cpu_usage_percent = _clamp(
        80.0 + (intensity * 15.0) + random.gauss(0, 2.0), 0, 100
    )
    base.request_latency_ms = _clamp(
        base.request_latency_ms * (1 + intensity * 3), 0, 10000
    )
    return base


def generate_memory_leak(pod: dict, elapsed_seconds: float) -> MetricPoint:
    """Simulates a slow memory leak — memory climbs linearly over time."""
    base = generate_normal_metric(pod)
    # Memory grows ~1% per 10 seconds
    leak_amount = (elapsed_seconds / 10.0) * 1.0
    base.memory_usage_percent = _clamp(
        45.0 + leak_amount + random.gauss(0, 1.0), 0, 100
    )
    return base


def generate_cascading_failure(pod: dict, failure_progress: float) -> MetricPoint:
    """Simulates a cascading failure — latency + errors explode together."""
    base = generate_normal_metric(pod)
    # failure_progress: 0.0 (start) → 1.0 (full failure)
    sigmoid = 1 / (1 + math.exp(-10 * (failure_progress - 0.5)))
    base.request_latency_ms = _clamp(120 + sigmoid * 4880, 0, 10000)
    base.error_rate_percent = _clamp(0.5 + sigmoid * 49.5, 0, 100)
    base.cpu_usage_percent = _clamp(
        base.cpu_usage_percent + sigmoid * 40, 0, 100
    )
    return base


# ── Log Templates ─────────────────────────────────────────────────────────────

NORMAL_LOG_TEMPLATES = [
    ("INFO", "Request processed successfully in {latency}ms"),
    ("INFO", "Health check passed"),
    ("INFO", "Cache hit ratio: {ratio}%"),
    ("INFO", "Scheduled job completed: metrics-aggregation"),
    ("INFO", "Connected to database pool [active={conn} idle={idle}]"),
]

ANOMALY_LOG_TEMPLATES = [
    ("ERROR", "Connection timeout after {latency}ms — retrying ({attempt}/3)"),
    ("ERROR", "Out of memory: Killed process {pid}"),
    ("CRITICAL", "Circuit breaker OPEN — downstream {service} unreachable"),
    ("ERROR", "Database pool exhausted: waiting for connection"),
    ("WARN", "CPU throttling detected — requests queued: {queue}"),
    ("CRITICAL", "Panic: runtime error: index out of range [{idx}]"),
]


def generate_log(pod: dict, anomaly: bool = False) -> LogEntry:
    """Generates a structured log entry."""
    if anomaly:
        level, template = random.choice(ANOMALY_LOG_TEMPLATES)
        message = template.format(
            latency=random.randint(5000, 30000),
            pid=random.randint(1000, 9999),
            service=random.choice(["postgres", "redis", "kafka"]),
            queue=random.randint(100, 1000),
            idx=random.randint(0, 99),
            attempt=random.randint(1, 3),
        )
    else:
        level, template = random.choice(NORMAL_LOG_TEMPLATES)
        message = template.format(
            latency=random.randint(80, 200),
            ratio=random.randint(70, 95),
            conn=random.randint(5, 20),
            idle=random.randint(5, 30),
        )

    return LogEntry(
        timestamp=datetime.now(timezone.utc),
        pod_name=pod["name"],
        namespace=pod["namespace"],
        level=level,
        message=message,
        trace_id=str(uuid.uuid4())[:16],
    )


# ── Streaming Generator ────────────────────────────────────────────────────────

async def stream_metrics(
    scenario: str = "normal",
    duration_seconds: int = 300,
    interval_seconds: float = 2.0,
    api_url: str = "http://localhost:8000",
    dry_run: bool = True,
) -> None:
    """
    Streams simulated metrics to the FastAPI gateway.
    
    Scenarios:
      - normal           : baseline operation
      - cpu_spike        : sudden CPU surge on payment-service
      - memory_leak      : slow memory growth on data-pipeline
      - cascading_failure: latency + error rate explosion on auth-service
    """
    target_pod = PODS[0]  # default: payment-service

    if scenario == "memory_leak":
        target_pod = PODS[4]  # data-pipeline
    elif scenario == "cascading_failure":
        target_pod = PODS[1]  # auth-service

    start_time = time.time()
    iteration = 0

    console.print(
        f"\n[bold cyan]NexusOps Metric Simulator[/bold cyan]\n"
        f"  Scenario : [yellow]{scenario}[/yellow]\n"
        f"  Duration : {duration_seconds}s\n"
        f"  Target   : {target_pod['name']}\n"
        f"  Dry Run  : {dry_run}\n"
    )

    async with httpx.AsyncClient(timeout=5.0) as client:
        while True:
            elapsed = time.time() - start_time
            if elapsed > duration_seconds:
                break

            metrics_batch: list[MetricPoint] = []

            for pod in PODS:
                is_target = pod["name"] == target_pod["name"]

                if scenario == "cpu_spike" and is_target:
                    intensity = min(1.0, elapsed / 30.0)  # ramp up over 30s
                    metric = generate_cpu_spike(pod, intensity)
                elif scenario == "memory_leak" and is_target:
                    metric = generate_memory_leak(pod, elapsed)
                elif scenario == "cascading_failure" and is_target:
                    progress = min(1.0, elapsed / 60.0)
                    metric = generate_cascading_failure(pod, progress)
                else:
                    metric = generate_normal_metric(pod)

                metrics_batch.append(metric)

            # Post to API (or print in dry-run mode)
            if not dry_run:
                for m in metrics_batch:
                    try:
                        await client.post(
                            f"{api_url}/api/v1/metrics/ingest",
                            content=m.model_dump_json(),
                            headers={"Content-Type": "application/json"},
                        )
                    except Exception as e:
                        console.print(f"[red]API error: {e}[/red]")
            else:
                # Pretty-print a summary table
                table = Table(
                    title=f"[Iteration {iteration + 1}] t={elapsed:.1f}s | Scenario: {scenario}",
                    show_header=True,
                )
                table.add_column("Pod", style="cyan", no_wrap=True)
                table.add_column("CPU%", justify="right")
                table.add_column("MEM%", justify="right")
                table.add_column("Latency(ms)", justify="right")
                table.add_column("Error%", justify="right")

                for m in metrics_batch:
                    short_name = m.pod_name.split("-")[0] + "-" + m.pod_name.split("-")[1]
                    cpu_str = f"[red]{m.cpu_usage_percent:.1f}[/red]" if m.cpu_usage_percent > 75 else f"{m.cpu_usage_percent:.1f}"
                    mem_str = f"[red]{m.memory_usage_percent:.1f}[/red]" if m.memory_usage_percent > 80 else f"{m.memory_usage_percent:.1f}"
                    lat_str = f"[red]{m.request_latency_ms:.0f}[/red]" if m.request_latency_ms > 1000 else f"{m.request_latency_ms:.0f}"
                    err_str = f"[red]{m.error_rate_percent:.2f}[/red]" if m.error_rate_percent > 5 else f"{m.error_rate_percent:.2f}"
                    table.add_row(short_name, cpu_str, mem_str, lat_str, err_str)

                console.print(table)

            iteration += 1
            await asyncio.sleep(interval_seconds)

    console.print("\n[green]✓ Simulation complete.[/green]\n")


# ── CLI Entrypoint ─────────────────────────────────────────────────────────────

@app.command()
def run(
    scenario: str = typer.Option("normal", help="Scenario: normal | cpu_spike | memory_leak | cascading_failure"),
    duration: int = typer.Option(120, help="Duration in seconds"),
    interval: float = typer.Option(2.0, help="Interval between metric batches"),
    api_url: str = typer.Option("http://localhost:8000", help="NexusOps API URL"),
    dry_run: bool = typer.Option(True, help="Print metrics instead of posting to API"),
):
    """Run the NexusOps synthetic metrics generator."""
    asyncio.run(
        stream_metrics(
            scenario=scenario,
            duration_seconds=duration,
            interval_seconds=interval,
            api_url=api_url,
            dry_run=dry_run,
        )
    )


if __name__ == "__main__":
    app()
