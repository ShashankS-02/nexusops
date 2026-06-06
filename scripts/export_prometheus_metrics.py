"""
Export real Prometheus metrics from a K3s cluster for LSTM training.

Queries Prometheus for pod-level CPU, memory, network I/O, and request
metrics, then writes them to CSV in the same 5-feature format the LSTM
autoencoder expects:
    cpu_usage_percent, memory_usage_percent, request_latency_ms,
    error_rate_percent, network_io_kbps

Usage:
    python -m scripts.export_prometheus_metrics
    python -m scripts.export_prometheus_metrics --hours 2 --step 15
"""

from __future__ import annotations

import csv
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
import typer
from rich.console import Console
from rich.progress import Progress

console = Console()
app = typer.Typer()

PROMETHEUS_URL = "http://localhost:9090"
OUTPUT_DIR = Path("data/prometheus_export")


def query_range(query: str, start: float, end: float, step: str = "15s") -> list[dict]:
    """Execute a Prometheus range query and return time-series results."""
    resp = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query_range",
        params={"query": query, "start": start, "end": end, "step": step},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if data["status"] != "success":
        console.print(f"[red]Query failed: {query}[/red]")
        return []
    return data["data"]["result"]


def extract_pod_series(results: list[dict], pod_filter: str = "") -> dict[str, list[tuple[float, float]]]:
    """Group time-series values by pod name."""
    pod_data: dict[str, list[tuple[float, float]]] = {}
    for series in results:
        pod = series["metric"].get("pod", series["metric"].get("container", "unknown"))
        if pod_filter and pod_filter not in pod:
            continue
        values = [(float(ts), float(val)) for ts, val in series["values"]]
        pod_data[pod] = values
    return pod_data


@app.command()
def export(
    hours: float = typer.Option(1.0, help="Hours of history to export"),
    step: str = typer.Option("15s", help="Query resolution step"),
    namespace: str = typer.Option("prod", help="Kubernetes namespace"),
    output: str = typer.Option("", help="Output CSV path (auto-generated if empty)"),
):
    """Export Prometheus metrics to CSV for LSTM training."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    end_ts = time.time()
    start_ts = end_ts - (hours * 3600)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output) if output else OUTPUT_DIR / f"metrics_{namespace}_{timestamp}.csv"

    console.print(
        f"\n[bold cyan]Prometheus Metrics Export[/bold cyan]\n"
        f"  Namespace : {namespace}\n"
        f"  Time range: {hours}h (step={step})\n"
        f"  Prometheus: {PROMETHEUS_URL}\n"
    )

    # ── Prometheus Queries ──────────────────────────────────────────────
    # These queries match the 5 features the LSTM expects
    queries = {
        "cpu_usage_percent": (
            f'100 * rate(container_cpu_usage_seconds_total{{namespace="{namespace}"}}[1m])'
        ),
        "memory_usage_percent": (
            f'100 * container_memory_working_set_bytes{{namespace="{namespace}"}}'
            f' / container_spec_memory_limit_bytes{{namespace="{namespace}"}} > 0'
        ),
        "network_receive_kbps": (
            f'rate(container_network_receive_bytes_total{{namespace="{namespace}"}}[1m]) / 1024'
        ),
        "network_transmit_kbps": (
            f'rate(container_network_transmit_bytes_total{{namespace="{namespace}"}}[1m]) / 1024'
        ),
    }

    all_metrics: dict[str, dict[str, list[tuple[float, float]]]] = {}

    with Progress(console=console) as progress:
        task = progress.add_task("Querying Prometheus...", total=len(queries))
        for metric_name, query in queries.items():
            results = query_range(query, start_ts, end_ts, step)
            all_metrics[metric_name] = extract_pod_series(results)
            progress.advance(task)

    # ── Merge into unified time-series per pod ──────────────────────────
    # Collect all pods that have CPU data
    pods = set()
    for metric_data in all_metrics.values():
        pods.update(metric_data.keys())

    if not pods:
        console.print("[yellow]No metrics found. Is the load generator running? Wait a few minutes and retry.[/yellow]")
        raise typer.Exit(1)

    console.print(f"  Found {len(pods)} pod(s): {', '.join(sorted(pods))}\n")

    # Build a timestamp index from CPU (most complete series)
    rows = []
    for pod in sorted(pods):
        cpu_series = all_metrics.get("cpu_usage_percent", {}).get(pod, [])
        mem_series = all_metrics.get("memory_usage_percent", {}).get(pod, [])
        net_rx_series = all_metrics.get("network_receive_kbps", {}).get(pod, [])
        net_tx_series = all_metrics.get("network_transmit_kbps", {}).get(pod, [])

        # Index by timestamp for fast lookup
        mem_map = {ts: val for ts, val in mem_series}
        net_rx_map = {ts: val for ts, val in net_rx_series}
        net_tx_map = {ts: val for ts, val in net_tx_series}

        for ts, cpu_val in cpu_series:
            mem_val = mem_map.get(ts, 0.0)
            net_rx = net_rx_map.get(ts, 0.0)
            net_tx = net_tx_map.get(ts, 0.0)
            network_io = net_rx + net_tx

            # We don't have real latency/error metrics from nginx,
            # so we derive synthetic but realistic proxies:
            # - latency correlates with CPU load (higher CPU → slower responses)
            # - error rate stays near zero for normal, spikes when CPU > 80%
            latency_ms = max(5.0, cpu_val * 50 + (mem_val * 10))
            error_rate = 0.1 if cpu_val < 80 else min(25.0, (cpu_val - 80) * 1.2)

            rows.append({
                "timestamp": datetime.fromtimestamp(ts).isoformat(),
                "pod_name": pod,
                "namespace": namespace,
                "cpu_usage_percent": round(cpu_val, 4),
                "memory_usage_percent": round(mem_val, 4),
                "request_latency_ms": round(latency_ms, 2),
                "error_rate_percent": round(error_rate, 4),
                "network_io_kbps": round(network_io, 4),
            })

    # ── Write CSV ───────────────────────────────────────────────────────
    fieldnames = [
        "timestamp", "pod_name", "namespace",
        "cpu_usage_percent", "memory_usage_percent",
        "request_latency_ms", "error_rate_percent", "network_io_kbps",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    console.print(
        f"[bold green]✓ Exported {len(rows)} data points[/bold green]\n"
        f"  File: [dim]{output_path}[/dim]\n"
        f"  Pods: {len(pods)}\n"
        f"  Ready for: [cyan]python -m ml.pytorch.train --data-source {output_path}[/cyan]\n"
    )


if __name__ == "__main__":
    app()
