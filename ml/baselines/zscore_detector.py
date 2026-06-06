"""
Z-Score Baseline Anomaly Detector
===================================

A simple statistical baseline that flags data points where any feature
exceeds N standard deviations from the training mean. Used to benchmark
the LSTM autoencoder against a non-learned approach.

Usage:
    python -m ml.baselines.zscore_detector
    python -m ml.baselines.zscore_detector --data-source data/prometheus_export/metrics_combined.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.table import Table
from sklearn.metrics import roc_auc_score

from ml.pytorch.dataset import (
    FEATURE_NAMES,
    FEATURE_SCALES,
    generate_synthetic_dataset,
    load_prometheus_dataset,
)

console = Console()
app = typer.Typer()


class ZScoreDetector:
    """
    Anomaly detector using per-feature z-scores.

    Training: compute mean and std from normal data.
    Inference: flag samples where max z-score across features exceeds threshold.
    """

    def __init__(self, z_threshold: float = 3.0) -> None:
        self.z_threshold = z_threshold
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, train_data: np.ndarray) -> None:
        """
        Compute per-feature mean and std from training data.

        Args:
            train_data: (n_samples, seq_len, n_features)
        """
        # Flatten to (n_samples * seq_len, n_features)
        flat = train_data.reshape(-1, train_data.shape[-1])
        self.mean = flat.mean(axis=0)
        self.std = flat.std(axis=0) + 1e-9  # avoid division by zero

    def score(self, data: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores as max z-score across features per window.

        Args:
            data: (n_samples, seq_len, n_features)
        Returns:
            scores: (n_samples,) — higher = more anomalous
        """
        if self.mean is None:
            raise RuntimeError("Call fit() before score()")

        # Z-score per time step per feature
        z = np.abs((data - self.mean) / self.std)
        # Max z-score across all features and time steps per window
        max_z = z.reshape(len(data), -1).max(axis=1)
        # Normalize to [0, 1] via sigmoid-like transform
        scores = 1.0 / (1.0 + np.exp(-0.5 * (max_z - self.z_threshold)))
        return scores

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Binary predictions: 1 = anomaly, 0 = normal."""
        scores = self.score(data)
        return (scores > 0.5).astype(int)


def evaluate_detector(
    name: str,
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute AUROC, precision, recall, F1 for a detector."""
    auroc = roc_auc_score(labels, scores)
    preds = (scores > threshold).astype(int)
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        "name": name,
        "auroc": float(auroc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


@app.command()
def benchmark(
    data_source: str = typer.Option("", help="Path to Prometheus CSV (empty = synthetic)"),
    seq_len: int = typer.Option(30, help="Sequence window length"),
    z_threshold: float = typer.Option(3.0, help="Z-score threshold for baseline"),
    lstm_model_path: str = typer.Option(
        "ml/pytorch/saved_models/lstm_autoencoder_best.pt",
        help="Path to trained LSTM model",
    ),
):
    """Benchmark z-score baseline against the LSTM autoencoder."""
    import torch
    from ml.pytorch.model import LSTMAutoencoder

    console.print("\n[bold cyan]NexusOps — Anomaly Detection Benchmark[/bold cyan]\n")

    # ── Load data ───────────────────────────────────────────────────────
    if data_source and Path(data_source).exists():
        console.print(f"  Data source: [green]Prometheus CSV[/green] ({data_source})")
        train_data, eval_normal, eval_anomaly = load_prometheus_dataset(
            csv_path=data_source, seq_len=seq_len,
        )
    else:
        console.print("  Data source: [yellow]Synthetic[/yellow]")
        train_data, eval_normal, eval_anomaly = generate_synthetic_dataset(seq_len=seq_len)

    console.print(
        f"  Train: {len(train_data)} windows | "
        f"Eval normal: {len(eval_normal)} | "
        f"Eval anomaly: {len(eval_anomaly)}\n"
    )

    # Build labels
    eval_data = np.concatenate([eval_normal, eval_anomaly])
    labels = np.concatenate([
        np.zeros(len(eval_normal)),
        np.ones(len(eval_anomaly)),
    ])

    results = []

    # ── Z-Score Baseline ────────────────────────────────────────────────
    console.print("[dim]Running z-score baseline...[/dim]")
    zscore = ZScoreDetector(z_threshold=z_threshold)
    zscore.fit(train_data)
    zscore_scores = zscore.score(eval_data)
    results.append(evaluate_detector("Z-Score Baseline", zscore_scores, labels))

    # ── Static Threshold Baseline ───────────────────────────────────────
    console.print("[dim]Running static threshold baseline...[/dim]")
    # Flag as anomaly if any feature exceeds 80% of its scale
    static_scores = (eval_data.reshape(len(eval_data), -1).max(axis=1) > 0.8).astype(float)
    # Add noise to avoid degenerate AUROC
    static_scores = static_scores + np.random.normal(0, 0.05, len(static_scores))
    static_scores = np.clip(static_scores, 0, 1)
    results.append(evaluate_detector("Static Threshold", static_scores, labels))

    # ── LSTM Autoencoder ────────────────────────────────────────────────
    if Path(lstm_model_path).exists():
        console.print("[dim]Running LSTM autoencoder...[/dim]")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = LSTMAutoencoder.load(lstm_model_path, device=device)
        eval_tensor = torch.tensor(eval_data, dtype=torch.float32).to(device)

        # Score in batches to avoid OOM
        lstm_scores = []
        batch_size = 64
        for i in range(0, len(eval_tensor), batch_size):
            batch = eval_tensor[i : i + batch_size]
            scores = model.anomaly_score(batch).cpu().numpy()
            lstm_scores.extend(scores)
        lstm_scores = np.array(lstm_scores)

        results.append(evaluate_detector("LSTM Autoencoder", lstm_scores, labels, threshold=0.6))
    else:
        console.print(f"[yellow]⚠ LSTM model not found at {lstm_model_path} — skipping[/yellow]")

    # ── Results Table ───────────────────────────────────────────────────
    table = Table(title="Anomaly Detection Benchmark", show_lines=True)
    table.add_column("Detector", style="bold")
    table.add_column("AUROC", justify="center")
    table.add_column("Precision", justify="center")
    table.add_column("Recall", justify="center")
    table.add_column("F1", justify="center")

    for r in results:
        auroc_color = "green" if r["auroc"] > 0.95 else "yellow" if r["auroc"] > 0.85 else "red"
        table.add_row(
            r["name"],
            f"[{auroc_color}]{r['auroc']:.4f}[/{auroc_color}]",
            f"{r['precision']:.4f}",
            f"{r['recall']:.4f}",
            f"{r['f1']:.4f}",
        )

    console.print()
    console.print(table)
    console.print()

    # ── Determine winner ────────────────────────────────────────────────
    best = max(results, key=lambda x: x["auroc"])
    console.print(
        f"[bold green]Winner: {best['name']}[/bold green] "
        f"(AUROC {best['auroc']:.4f})\n"
    )


if __name__ == "__main__":
    app()
