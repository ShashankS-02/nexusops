"""
Sentinel Agent — Anomaly Detection Engine
==========================================

The Sentinel Agent is the "eyes" of NexusOps. It:

1. Receives a rolling window of MetricPoints
2. Runs them through the PyTorch LSTM Autoencoder
3. Also classifies the associated logs through the TF classifier
4. Fuses both signals into a unified anomaly score
5. If the score exceeds threshold, it constructs an AnomalyAlert
   which is handed to the Supervisor to trigger the full agent pipeline

Fusion Strategy:
  final_score = 0.7 * pytorch_score + 0.3 * tf_score

  We weight PyTorch higher because the LSTM captures temporal patterns
  in the numerical metrics that are more reliable than keyword-based
  log classification for detecting subtle anomalies.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from nexusops.config import settings
from nexusops.models import MetricPoint
from ml.pytorch.dataset import metric_to_vector


class SentinelDetector:
    """
    Fused anomaly detector combining PyTorch LSTM and TensorFlow log classifier.

    Lazy-loads models on first call if paths exist.
    Falls back to heuristic scoring if models aren't trained yet.
    """

    PYTORCH_WEIGHT = 0.7
    TF_WEIGHT = 0.3

    def __init__(self) -> None:
        self._pytorch_model = None
        self._tf_model = None
        self._load_models()

    def _load_models(self) -> None:
        """Attempt to load both models. Silently skip if not trained yet."""
        # ── PyTorch LSTM ─────────────────────────────────────────────────────
        pytorch_path = Path(settings.PYTORCH_MODEL_PATH)
        if pytorch_path.exists():
            try:
                import torch
                from ml.pytorch.model import LSTMAutoencoder
                self._pytorch_model = LSTMAutoencoder.load(str(pytorch_path))
                self._pytorch_model.eval()
                print(f"  ✓ PyTorch model loaded from {pytorch_path}")
            except Exception as e:
                print(f"  ⚠ PyTorch model failed to load: {e}")

        # ── TensorFlow Classifier ────────────────────────────────────────────
        tf_path = Path(settings.TF_MODEL_PATH)
        if tf_path.exists():
            try:
                import tensorflow as tf
                os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
                self._tf_model = tf.keras.models.load_model(str(tf_path))
                print(f"  ✓ TensorFlow model loaded from {tf_path}")
            except Exception as e:
                print(f"  ⚠ TensorFlow model failed to load: {e}")

    @property
    def models_available(self) -> dict:
        return {
            "pytorch": self._pytorch_model is not None,
            "tensorflow": self._tf_model is not None,
        }

    def score(self, window: list[MetricPoint]) -> float:
        """
        Compute a fused anomaly score [0.0, 1.0] for a metric window.

        Args:
            window: list of 30 MetricPoints (the most recent sliding window)

        Returns:
            float: 0.0 = perfectly normal, 1.0 = definitely anomalous
        """
        pytorch_score = self._pytorch_score(window)
        tf_score = self._tf_score_from_metrics(window)

        # Fuse
        if self._pytorch_model and self._tf_model:
            fused = self.PYTORCH_WEIGHT * pytorch_score + self.TF_WEIGHT * tf_score
        elif self._pytorch_model:
            fused = pytorch_score
        elif self._tf_model:
            fused = tf_score
        else:
            # No models loaded — fall back to rule-based heuristic
            fused = self._heuristic_score(window)

        fused = float(np.clip(fused, 0.0, 1.0))
        
        # Debug log if there is any interesting activity
        if fused > 0.1:
            print(f"[\U0001f50e Sentinel] {window[-1].pod_name} | PyTorch: {pytorch_score:.3f} | TF: {tf_score:.3f} | Fused: {fused:.3f} | Threshold: {settings.SENTINEL_ANOMALY_THRESHOLD}")
            
        return fused

    def _pytorch_score(self, window: list[MetricPoint]) -> float:
        """Run the PyTorch LSTM autoencoder on the metric window."""
        if not self._pytorch_model:
            return 0.0

        import torch

        # Convert window to tensor: (1, seq_len, n_features)
        vectors = np.array([metric_to_vector(m) for m in window], dtype=np.float32)
        x = torch.tensor(vectors).unsqueeze(0)  # (1, 30, 5)

        with torch.no_grad():
            score = self._pytorch_model.anomaly_score(x)

        return float(score.item())

    def _tf_score_from_metrics(self, window: list[MetricPoint]) -> float:
        """
        Convert the last metrics to a log-like text representation
        and score with the TF classifier.

        In Phase 2, this will use real log entries instead.
        """
        if not self._tf_model:
            return 0.0

        import tensorflow as tf

        # Synthesize a pseudo-log string from metric values
        last = window[-1]
        pseudo_log = (
            f"ERROR: CPU at {last.cpu_usage_percent:.1f}% "
            f"WARN: latency {last.request_latency_ms:.0f}ms "
            f"ERROR: error rate {last.error_rate_percent:.2f}% "
            f"INFO: memory {last.memory_usage_percent:.1f}%"
        )
        x = tf.constant([[pseudo_log]], dtype=tf.string)

        try:
            probs = self._tf_model(x, training=False).numpy()[0]
            # probs[0]=NORMAL, probs[1]=WARNING, probs[2]=CRITICAL
            # Map to anomaly score: 0*normal + 0.5*warning + 1.0*critical
            score = 0.0 * probs[0] + 0.5 * probs[1] + 1.0 * probs[2]
            return float(score)
        except Exception:
            return 0.0

    def _heuristic_score(self, window: list[MetricPoint]) -> float:
        """
        Rule-based fallback anomaly scoring when models aren't trained.
        Uses simple thresholds — good for sanity testing Phase 1.
        """
        last = window[-1]
        score = 0.0

        if last.cpu_usage_percent > 85:
            score += 0.4
        elif last.cpu_usage_percent > 70:
            score += 0.2

        if last.memory_usage_percent > 90:
            score += 0.3
        elif last.memory_usage_percent > 75:
            score += 0.1

        if last.request_latency_ms > 2000:
            score += 0.2
        elif last.request_latency_ms > 500:
            score += 0.1

        if last.error_rate_percent > 10:
            score += 0.3
        elif last.error_rate_percent > 2:
            score += 0.1

        return min(score, 1.0)
