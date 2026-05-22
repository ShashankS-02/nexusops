"""
Sentinel — TensorFlow Log Severity Classifier
==============================================

Wraps the trained TF Dense log classifier for inference inside the
Detective node. Classifies a window of log messages into:
  NORMAL  (0) — healthy traffic
  WARNING (1) — degraded but not failed
  CRITICAL(2) — active failure / OOM / circuit-breaker open

Input:  list of log message strings (the recent_logs window)
Output: dict with predicted_severity and per-class probabilities
"""
from __future__ import annotations

from nexusops.config import settings

CLASSES = ["NORMAL", "WARNING", "CRITICAL"]


class LogClassifier:
    """
    Loads the saved Keras Dense model and scores a log window.

    Raises on __init__ if the model file is missing — Detective catches
    this and continues without log classification (non-fatal).
    """

    def __init__(self) -> None:
        import tensorflow as tf  # deferred — heavy import

        # Force CPU — Metal GPU plugin miscomputes TextVectorization on macOS
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass

        self.model = tf.keras.models.load_model(settings.TF_MODEL_PATH)

    def classify(self, log_messages: list[str]) -> dict:
        """
        Classify a window of log messages.

        Args:
            log_messages: list of raw log strings (any length).

        Returns:
            {
              "predicted_severity": "NORMAL" | "WARNING" | "CRITICAL",
              "probabilities": {"NORMAL": 0.9, "WARNING": 0.08, "CRITICAL": 0.02}
            }
        """
        # Join window into a single string — matches training format
        text = " ".join(log_messages)

        # TextVectorization requires tf.string tensors — numpy str dtype is incompatible
        import tensorflow as tf
        import numpy as np
        probs = self.model.predict(tf.constant([text]), verbose=0)[0]
        predicted = CLASSES[int(np.argmax(probs))]

        return {
            "predicted_severity": predicted,
            "probabilities": {
                "NORMAL":   float(probs[0]),
                "WARNING":  float(probs[1]),
                "CRITICAL": float(probs[2]),
            },
        }
