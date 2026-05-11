"""
TensorFlow Log Classifier — Dataset Generation & Training
==========================================================

Generates synthetic labeled log windows and trains the TF classifier.
Tracks experiments with MLflow and saves the model in SavedModel format.

Usage:
    python -m ml.tensorflow.train
    python -m ml.tensorflow.train --epochs 15 --model-type dense
"""
from __future__ import annotations

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
from pathlib import Path
from typing import Literal

import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf

# Force CPU training — macOS Metal GPU plugin crashes on string operations (TextVectorization)
# Since the dataset is small, CPU training will be extremely fast anyway.
try:
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass

import typer
from rich.console import Console

from nexusops.config import settings
from simulator.generator import generate_log, PODS
from ml.tensorflow.model import (
    build_log_classifier, 
    build_log_classifier_with_lstm, 
    NUM_CLASSES, 
    VOCAB_SIZE
)

console = Console()
app = typer.Typer()

MODEL_SAVE_DIR = Path("ml/tensorflow/saved_models")
LOG_WINDOW_SIZE = 50  # number of log entries per window


# ── Dataset Generation ─────────────────────────────────────────────────────────

def generate_log_dataset(
    n_normal: int = 3000,
    n_warning: int = 1500,
    n_critical: int = 1500,
    window_size: int = LOG_WINDOW_SIZE,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate labeled log windows.

    Labels:
      0 → NORMAL   (all INFO logs)
      1 → WARNING  (mix of INFO and WARN/ERROR)
      2 → CRITICAL (majority ERROR/CRITICAL logs)

    Returns:
        X_train, X_val, y_train, y_val as numpy arrays of strings
    """
    random.seed(seed)
    np.random.seed(seed)

    pod = PODS[0]
    texts, labels = [], []

    # Class 0: NORMAL
    for _ in range(n_normal):
        logs = [generate_log(pod, anomaly=False) for _ in range(window_size)]
        window_text = " [SEP] ".join(f"{l.level}: {l.message}" for l in logs)
        texts.append(window_text)
        labels.append(0)

    # Class 1: WARNING (30% anomalous logs mixed in)
    for _ in range(n_warning):
        logs = []
        for _ in range(window_size):
            is_anomaly = random.random() < 0.3
            logs.append(generate_log(pod, anomaly=is_anomaly))
        window_text = " [SEP] ".join(f"{l.level}: {l.message}" for l in logs)
        texts.append(window_text)
        labels.append(1)

    # Class 2: CRITICAL (80%+ anomalous logs)
    for _ in range(n_critical):
        logs = []
        for _ in range(window_size):
            is_anomaly = random.random() < 0.85
            logs.append(generate_log(pod, anomaly=is_anomaly))
        window_text = " [SEP] ".join(f"{l.level}: {l.message}" for l in logs)
        texts.append(window_text)
        labels.append(2)

    # Shuffle
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)

    texts = np.array(texts, dtype=object)
    labels = np.array(labels, dtype=np.int32)

    # Train/val split (80/20)
    n_val = int(len(texts) * 0.2)
    return texts[n_val:], texts[:n_val], labels[n_val:], labels[:n_val]


# ── Training ───────────────────────────────────────────────────────────────────

@app.command()
def train(
    epochs: int = typer.Option(10, help="Number of training epochs"),
    batch_size: int = typer.Option(64, help="Batch size"),
    model_type: str = typer.Option("dense", help="Model type: dense | lstm"),
    dropout_rate: float = typer.Option(0.3, help="Dropout rate"),
):
    """Train the TensorFlow log pattern classifier."""

    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    console.print(
        f"\n[bold cyan]NexusOps — TensorFlow Log Classifier Training[/bold cyan]\n"
        f"  Model type : [yellow]{model_type}[/yellow]\n"
        f"  Epochs     : {epochs}\n"
        f"  Batch size : {batch_size}\n"
        f"  TF version : [dim]{tf.__version__}[/dim]\n"
    )

    # ── Generate Data ────────────────────────────────────────────────────────
    console.print("[dim]Generating synthetic log dataset...[/dim]")
    X_train, X_val, y_train, y_val = generate_log_dataset()
    console.print(
        f"  Train samples: {len(X_train)} | Val samples: {len(X_val)}\n"
        f"  Class distribution: {np.bincount(y_train)}\n"
    )

    # Convert to explicit TensorFlow string tensors (1D array of strings)
    X_train_tf = tf.constant(list(X_train), dtype=tf.string)
    X_val_tf = tf.constant(list(X_val), dtype=tf.string)

    # ── Build & Adapt Model ──────────────────────────────────────────────────
    console.print("[dim]Adapting text vectorizer...[/dim]")
    
    if model_type == "dense":
        text_vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=VOCAB_SIZE, 
            output_mode="tf_idf", 
            name="text_vectorization"
        )
        text_vectorizer.adapt(X_train_tf)
        model = build_log_classifier(text_vectorizer, dropout_rate=dropout_rate)
    else:
        text_vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=VOCAB_SIZE, 
            output_mode="int", 
            output_sequence_length=200, 
            name="text_vec_lstm"
        )
        text_vectorizer.adapt(X_train_tf)
        model = build_log_classifier_with_lstm(text_vectorizer, dropout_rate=dropout_rate)

    model.summary(print_fn=lambda x: console.print(f"  [dim]{x}[/dim]"))

    # Use local file tracking — no Docker server needed, no version conflicts.
    # View results with: mlflow ui --port 5001
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("nexusops-log-classification")

    with mlflow.start_run(run_name=f"tf_{model_type}_e{epochs}"):
        mlflow.log_params({
            "model_type": model_type,
            "epochs": epochs,
            "batch_size": batch_size,
            "dropout_rate": dropout_rate,
            "tf_version": tf.__version__,
            "num_classes": NUM_CLASSES,
        })

        # ── Keras Callbacks ──────────────────────────────────────────────────
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=3,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                verbose=1,
            ),
            # MLflow callback for auto-logging each epoch
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: mlflow.log_metrics(
                    {
                        "train_loss": logs.get("loss", 0),
                        "train_accuracy": logs.get("accuracy", 0),
                        "val_loss": logs.get("val_loss", 0),
                        "val_accuracy": logs.get("val_accuracy", 0),
                    },
                    step=epoch,
                )
            ),
        ]

        # ── Training ─────────────────────────────────────────────────────────
        history = model.fit(
            X_train_tf,
            y_train,
            validation_data=(X_val_tf, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        # ── Save Model ────────────────────────────────────────────────────────
        save_path = MODEL_SAVE_DIR / f"log_classifier_{model_type}.keras"
        model.save(str(save_path))
        mlflow.tensorflow.log_model(model, name=f"tf_log_classifier_{model_type}")

        # Log final metrics
        final_val_acc = max(history.history.get("val_accuracy", [0]))
        mlflow.log_metric("final_val_accuracy", final_val_acc)

        console.print(
            f"\n[bold green]✓ TensorFlow Training Complete![/bold green]\n"
            f"\n  [bold]Results[/bold]\n"
            f"  ─────────────────────────────────────\n"
            f"  Best Val Accuracy : [cyan]{final_val_acc:.4f}[/cyan]\n"
            f"  Model saved       : [dim]{save_path}[/dim]\n"
            f"  MLflow UI         : {settings.MLFLOW_TRACKING_URI}\n"
        )


if __name__ == "__main__":
    app()
