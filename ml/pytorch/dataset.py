"""
PyTorch Dataset for Metric Time-Series Windows
================================================

Handles:
  - Synthetic data generation (for Phase 1 training without real k8s data)
  - Real Prometheus CSV data loading (Phase 4 — K3s cluster metrics)
  - Sliding window creation from raw metric sequences
  - Feature normalization (StandardScaler per feature)
  - Train/val split

Usage:
    python -m ml.pytorch.dataset --generate --output data/metrics_normal.npy
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from simulator.generator import (
    PODS,
    generate_cascading_failure,
    generate_cpu_spike,
    generate_memory_leak,
    generate_normal_metric,
)

# ── Feature Extraction ────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "cpu_usage_percent",
    "memory_usage_percent",
    "request_latency_ms",
    "error_rate_percent",
    "network_io_kbps",
]

# Normalization constants (approximate max values for each feature)
FEATURE_SCALES = [100.0, 100.0, 10000.0, 100.0, 100000.0]


def metric_to_vector(metric) -> np.ndarray:
    """Convert a MetricPoint to a normalized numpy feature vector."""
    raw = np.array(
        [
            metric.cpu_usage_percent,
            metric.memory_usage_percent,
            metric.request_latency_ms,
            metric.error_rate_percent,
            metric.network_io_kbps,
        ],
        dtype=np.float32,
    )
    scales = np.array(FEATURE_SCALES, dtype=np.float32)
    return raw / scales  # scale to [0, 1]


# ── Synthetic Dataset Generation ──────────────────────────────────────────────


def generate_synthetic_dataset(
    n_normal_sequences: int = 5000,
    n_anomaly_sequences: int = 500,  # used for evaluation only, not training
    seq_len: int = 30,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic training and evaluation datasets.

    Returns:
        train_data:  (n_normal, seq_len, n_features) — normal sequences for training
        eval_normal: (n_normal//5, seq_len, n_features) — held-out normal sequences
        eval_anomaly:(n_anomaly, seq_len, n_features) — anomalous sequences for eval
    """
    random.seed(seed)
    np.random.seed(seed)

    pod = PODS[0]

    # ── Normal sequences ─────────────────────────────────────────────────────
    normal_seqs = []
    for _ in range(n_normal_sequences + n_normal_sequences // 5):
        seq = [metric_to_vector(generate_normal_metric(pod)) for _ in range(seq_len)]
        normal_seqs.append(seq)

    normal_array = np.array(normal_seqs, dtype=np.float32)
    train_data = normal_array[:n_normal_sequences]
    eval_normal = normal_array[n_normal_sequences:]

    # ── Anomalous sequences ──────────────────────────────────────────────────
    anomaly_seqs = []
    anomaly_per_type = n_anomaly_sequences // 3

    # CPU spikes
    for _ in range(anomaly_per_type):
        intensity = random.uniform(0.5, 1.0)
        seq = [metric_to_vector(generate_cpu_spike(pod, intensity)) for _ in range(seq_len)]
        anomaly_seqs.append(seq)

    # Memory leaks
    for _i in range(anomaly_per_type):
        elapsed = random.uniform(60, 600)
        seq = [metric_to_vector(generate_memory_leak(pod, elapsed + j * 2)) for j in range(seq_len)]
        anomaly_seqs.append(seq)

    # Cascading failures
    for _ in range(n_anomaly_sequences - 2 * anomaly_per_type):
        progress = random.uniform(0.3, 1.0)
        seq = [metric_to_vector(generate_cascading_failure(pod, progress)) for _ in range(seq_len)]
        anomaly_seqs.append(seq)

    eval_anomaly = np.array(anomaly_seqs, dtype=np.float32)

    return train_data, eval_normal, eval_anomaly


# ── Real Prometheus Data Loading ──────────────────────────────────────────────


def load_prometheus_dataset(
    csv_path: str | Path,
    seq_len: int = 30,
    anomaly_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load real Prometheus metrics from CSV and create sliding windows.

    The CSV is assumed to contain normal operational data. Anomaly sequences
    are synthesized by injecting realistic perturbations (scaled spikes,
    shifted distributions) into copies of real windows — this lets us
    evaluate the LSTM on data whose normal baseline comes from production.

    Returns:
        train_data:   (n_train, seq_len, 5) — normal windows for training
        eval_normal:  (n_eval, seq_len, 5)  — held-out normal windows
        eval_anomaly: (n_anom, seq_len, 5)  — perturbed windows for eval
    """
    np.random.seed(seed)
    random.seed(seed)

    df = pd.read_csv(csv_path)
    df = df.sort_values(["pod_name", "timestamp"])

    # Extract the 5 features in the order the LSTM expects
    feature_cols = FEATURE_NAMES
    all_windows: list[np.ndarray] = []

    for _pod, group in df.groupby("pod_name"):
        values = group[feature_cols].values.astype(np.float32)
        # Normalize using the same scales as synthetic data
        scales = np.array(FEATURE_SCALES, dtype=np.float32)
        values = values / scales

        # Create sliding windows with 50% overlap
        stride = max(1, seq_len // 2)
        for start in range(0, len(values) - seq_len + 1, stride):
            window = values[start : start + seq_len]
            all_windows.append(window)

    if not all_windows:
        raise ValueError(
            f"No valid windows created from {csv_path}. Need at least {seq_len} rows per pod."
        )

    windows = np.array(all_windows, dtype=np.float32)
    np.random.shuffle(windows)

    # Split: 80% train, 10% eval_normal, 10% base for anomaly synthesis
    n_total = len(windows)
    n_train = int(n_total * 0.80)
    n_eval = int(n_total * 0.10)

    train_data = windows[:n_train]
    eval_normal = windows[n_train : n_train + n_eval]
    anomaly_base = windows[n_train + n_eval :]

    # Synthesize anomalies from real data by injecting perturbations
    n_anomaly = max(len(anomaly_base), int(n_total * anomaly_fraction))
    anomaly_seqs = []
    for i in range(n_anomaly):
        base = anomaly_base[i % len(anomaly_base)].copy()
        perturbation = np.random.choice(["spike", "drift", "plateau"])

        if perturbation == "spike":
            # Sudden CPU/memory spike in the last third of the window
            spike_start = seq_len * 2 // 3
            base[spike_start:, 0] = np.clip(base[spike_start:, 0] * 3.0 + 0.3, 0, 1)  # CPU
            base[spike_start:, 1] = np.clip(base[spike_start:, 1] * 2.5 + 0.2, 0, 1)  # Memory
            base[spike_start:, 2] = np.clip(base[spike_start:, 2] * 4.0, 0, 1)  # Latency
            base[spike_start:, 3] = np.clip(base[spike_start:, 3] + 0.15, 0, 1)  # Error rate

        elif perturbation == "drift":
            # Gradual increase across the whole window (memory leak pattern)
            ramp = np.linspace(1.0, 3.0, seq_len).reshape(-1, 1)
            base[:, 1] = np.clip(base[:, 1] * ramp.flatten(), 0, 1)
            base[:, 2] = np.clip(base[:, 2] * ramp.flatten() * 1.5, 0, 1)

        else:  # plateau
            # All metrics flatline at high values (resource exhaustion)
            base[seq_len // 2 :, 0] = np.random.uniform(0.85, 0.98)
            base[seq_len // 2 :, 1] = np.random.uniform(0.80, 0.95)
            base[seq_len // 2 :, 3] = np.random.uniform(0.10, 0.25)

        anomaly_seqs.append(base)

    eval_anomaly = np.array(anomaly_seqs, dtype=np.float32)

    return train_data, eval_normal, eval_anomaly


# ── PyTorch Dataset Class ─────────────────────────────────────────────────────


class MetricWindowDataset(Dataset):
    """
    PyTorch Dataset wrapping a numpy array of metric windows.

    Args:
        data: (n_samples, seq_len, n_features) numpy array
        labels: optional (n_samples,) array — 0=normal, 1=anomaly (for eval)
    """

    def __init__(self, data: np.ndarray, labels: np.ndarray | None = None) -> None:
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        return self.data[idx]


def get_dataloaders(
    batch_size: int = 64,
    seq_len: int = 30,
    val_split: float = 0.1,
    num_workers: int = 0,
    seed: int = 42,
    data_source: str | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, eval_normal_loader, eval_anomaly_loader).

    Args:
        data_source: Path to a Prometheus CSV file. If None, uses synthetic data.
    """
    if data_source and Path(data_source).exists():
        train_data, eval_normal, eval_anomaly = load_prometheus_dataset(
            csv_path=data_source,
            seq_len=seq_len,
            seed=seed,
        )
    else:
        train_data, eval_normal, eval_anomaly = generate_synthetic_dataset(
            seq_len=seq_len,
            seed=seed,
        )

    # Split train into train + val
    n_val = int(len(train_data) * val_split)
    val_data = train_data[:n_val]
    train_data = train_data[n_val:]

    train_ds = MetricWindowDataset(train_data)
    val_ds = MetricWindowDataset(val_data)
    eval_normal_ds = MetricWindowDataset(eval_normal, labels=np.zeros(len(eval_normal), dtype=int))
    eval_anomaly_ds = MetricWindowDataset(
        eval_anomaly, labels=np.ones(len(eval_anomaly), dtype=int)
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    eval_normal_loader = DataLoader(
        eval_normal_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    eval_anomaly_loader = DataLoader(
        eval_anomaly_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, eval_normal_loader, eval_anomaly_loader
