"""
PyTorch LSTM Autoencoder Training Script
=========================================

Trains the anomaly detection model on synthetic normal metric data.
Logs all experiments to MLflow (local instance).

Usage:
    python -m ml.pytorch.train
    python -m ml.pytorch.train --epochs 30 --hidden-size 128 --lr 0.001

What you'll learn:
  - PyTorch training loop structure
  - Early stopping
  - MLflow experiment tracking
  - Model checkpointing
  - Evaluation with precision/recall/AUROC
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.optim as optim
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from sklearn.metrics import roc_auc_score, precision_recall_curve

from nexusops.config import settings
from ml.pytorch.model import LSTMAutoencoder
from ml.pytorch.dataset import get_dataloaders

console = Console()
app = typer.Typer()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_DIR = Path("ml/pytorch/saved_models")


def train_one_epoch(
    model: LSTMAutoencoder,
    loader,
    optimizer: optim.Optimizer,
) -> float:
    """Run one full training epoch. Returns average training loss."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        # batch shape: (batch_size, seq_len, n_features)
        x = batch.to(DEVICE)
        optimizer.zero_grad()
        reconstruction = model(x)
        loss = model.reconstruction_loss(x, reconstruction)
        loss.backward()
        # Gradient clipping — important for LSTMs to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model: LSTMAutoencoder, loader) -> float:
    """Compute average reconstruction loss on a dataset (no gradients)."""
    model.eval()
    total_loss = 0.0
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            x = batch[0].to(DEVICE)
        else:
            x = batch.to(DEVICE)
        reconstruction = model(x)
        loss = model.reconstruction_loss(x, reconstruction)
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def compute_auroc(
    model: LSTMAutoencoder,
    normal_loader,
    anomaly_loader,
    threshold: float,
) -> dict:
    """
    Compute AUROC and find optimal classification threshold.
    
    Returns a dict with: auroc, precision, recall, f1, threshold
    """
    model.eval()
    all_scores = []
    all_labels = []

    # Collect scores and labels from both loaders
    for loader, label in [(normal_loader, 0), (anomaly_loader, 1)]:
        for batch in loader:
            x, _ = batch
            x = x.to(DEVICE)
            scores = model.anomaly_score(x).cpu().numpy()
            all_scores.extend(scores)
            all_labels.extend([label] * len(scores))

    scores = np.array(all_scores)
    labels = np.array(all_labels)

    auroc = roc_auc_score(labels, scores)
    preds = (scores > threshold).astype(int)
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        "auroc": float(auroc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": threshold,
    }


@app.command()
def train(
    epochs: int = typer.Option(20, help="Number of training epochs"),
    hidden_size: int = typer.Option(64, help="LSTM hidden state size"),
    num_layers: int = typer.Option(2, help="Number of LSTM layers"),
    seq_len: int = typer.Option(30, help="Sequence window length (time steps)"),
    batch_size: int = typer.Option(64, help="Training batch size"),
    lr: float = typer.Option(1e-3, help="Learning rate"),
    dropout: float = typer.Option(0.2, help="Dropout rate"),
    patience: int = typer.Option(5, help="Early stopping patience (epochs)"),
    anomaly_threshold: float = typer.Option(0.85, help="Anomaly score threshold"),
):
    """Train the LSTM Autoencoder anomaly detection model."""

    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    console.print(
        f"\n[bold cyan]NexusOps — PyTorch LSTM Autoencoder Training[/bold cyan]\n"
        f"  Device      : [yellow]{DEVICE}[/yellow]\n"
        f"  Hidden size : {hidden_size}\n"
        f"  Num layers  : {num_layers}\n"
        f"  Seq length  : {seq_len}\n"
        f"  Batch size  : {batch_size}\n"
        f"  Epochs      : {epochs}\n"
        f"  LR          : {lr}\n"
    )

    # ── Data ────────────────────────────────────────────────────────────────
    console.print("[dim]Generating synthetic dataset...[/dim]")
    train_loader, val_loader, eval_normal_loader, eval_anomaly_loader = get_dataloaders(
        batch_size=batch_size,
        seq_len=seq_len,
    )
    console.print(
        f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}\n"
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = LSTMAutoencoder(
        input_size=5,
        hidden_size=hidden_size,
        num_layers=num_layers,
        seq_len=seq_len,
        dropout=dropout,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"  Model parameters: [green]{total_params:,}[/green]\n")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # Use local file tracking — no Docker server needed, no version conflicts.
    # View results with: mlflow ui --port 5001
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("nexusops-anomaly-detection")

    with mlflow.start_run(run_name=f"lstm_ae_h{hidden_size}_l{num_layers}"):
        # Log hyperparameters
        mlflow.log_params({
            "model_type": "LSTMAutoencoder",
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "learning_rate": lr,
            "dropout": dropout,
            "device": str(DEVICE),
            "total_params": total_params,
        })

        # ── Training Loop ─────────────────────────────────────────────────────
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        best_model_path = MODEL_SAVE_DIR / "lstm_autoencoder_best.pt"

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[cyan]{task.completed}/{task.total}[/cyan]"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Training...", total=epochs)

            for epoch in range(1, epochs + 1):
                train_loss = train_one_epoch(model, train_loader, optimizer)
                val_loss = evaluate(model, val_loader)
                scheduler.step(val_loss)

                # Log to MLflow
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                }, step=epoch)

                progress.update(
                    task,
                    advance=1,
                    description=f"Epoch {epoch:3d}/{epochs} | "
                                f"Train: {train_loss:.6f} | "
                                f"Val: {val_loss:.6f} | "
                                f"LR: {optimizer.param_groups[0]['lr']:.2e}",
                )

                # ── Early Stopping + Checkpointing ────────────────────────────
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    model.save(str(best_model_path))
                    mlflow.log_metric("best_val_loss", best_val_loss, step=epoch)
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        console.print(
                            f"\n[yellow]⚠ Early stopping at epoch {epoch} "
                            f"(no improvement for {patience} epochs)[/yellow]"
                        )
                        break

        # ── Evaluation ────────────────────────────────────────────────────────
        console.print("\n[bold]Evaluating anomaly detection performance...[/bold]")

        # Load best model
        best_model = LSTMAutoencoder.load(str(best_model_path), device=str(DEVICE))

        metrics = compute_auroc(
            best_model,
            eval_normal_loader,
            eval_anomaly_loader,
            threshold=anomaly_threshold,
        )

        mlflow.log_metrics({
            "eval_auroc": metrics["auroc"],
            "eval_precision": metrics["precision"],
            "eval_recall": metrics["recall"],
            "eval_f1": metrics["f1"],
        })

        # Log model artifact
        mlflow.pytorch.log_model(best_model, name="lstm_autoencoder")

        console.print(
            f"\n[bold green]✓ Training Complete![/bold green]\n"
            f"\n  [bold]Evaluation Results[/bold]\n"
            f"  ─────────────────────────────────────\n"
            f"  AUROC     : [cyan]{metrics['auroc']:.4f}[/cyan]\n"
            f"  Precision : [cyan]{metrics['precision']:.4f}[/cyan]\n"
            f"  Recall    : [cyan]{metrics['recall']:.4f}[/cyan]\n"
            f"  F1 Score  : [cyan]{metrics['f1']:.4f}[/cyan]\n"
            f"  Threshold : {metrics['threshold']:.2f}\n"
            f"\n  Model saved: [dim]{best_model_path}[/dim]\n"
            f"  MLflow UI : [link={settings.MLFLOW_TRACKING_URI}]{settings.MLFLOW_TRACKING_URI}[/link]\n"
        )


if __name__ == "__main__":
    app()
