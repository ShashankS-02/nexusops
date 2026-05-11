"""
PyTorch Dataset for Metric Time-Series Windows
================================================

Handles:
  - Synthetic data generation (for Phase 1 training without real k8s data)
  - Sliding window creation from raw metric sequences
  - Feature normalization (StandardScaler per feature)
  - Train/val split

Usage:
    python -m ml.pytorch.dataset --generate --output data/metrics_normal.npy
"""
from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from simulator.generator import (
    generate_normal_metric,
    generate_cpu_spike,
    generate_memory_leak,
    generate_cascading_failure,
    PODS,
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
    raw = np.array([
        metric.cpu_usage_percent,
        metric.memory_usage_percent,
        metric.request_latency_ms,
        metric.error_rate_percent,
        metric.network_io_kbps,
    ], dtype=np.float32)
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
    for i in range(anomaly_per_type):
        elapsed = random.uniform(60, 600)
        seq = [
            metric_to_vector(generate_memory_leak(pod, elapsed + j * 2))
            for j in range(seq_len)
        ]
        anomaly_seqs.append(seq)

    # Cascading failures
    for _ in range(n_anomaly_sequences - 2 * anomaly_per_type):
        progress = random.uniform(0.3, 1.0)
        seq = [
            metric_to_vector(generate_cascading_failure(pod, progress))
            for _ in range(seq_len)
        ]
        anomaly_seqs.append(seq)

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

    def __init__(self, data: np.ndarray, labels: Optional[np.ndarray] = None) -> None:
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
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, eval_normal_loader, eval_anomaly_loader).
    """
    train_data, eval_normal, eval_anomaly = generate_synthetic_dataset(
        seq_len=seq_len, seed=seed
    )

    # Split train into train + val
    n_val = int(len(train_data) * val_split)
    val_data = train_data[:n_val]
    train_data = train_data[n_val:]

    train_ds = MetricWindowDataset(train_data)
    val_ds = MetricWindowDataset(val_data)
    eval_normal_ds = MetricWindowDataset(
        eval_normal, labels=np.zeros(len(eval_normal), dtype=int)
    )
    eval_anomaly_ds = MetricWindowDataset(
        eval_anomaly, labels=np.ones(len(eval_anomaly), dtype=int)
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    eval_normal_loader = DataLoader(
        eval_normal_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    eval_anomaly_loader = DataLoader(
        eval_anomaly_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, eval_normal_loader, eval_anomaly_loader
