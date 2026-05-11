"""
TensorFlow Log Pattern Classifier
====================================

Classifies a sequence of log messages into severity categories:
  0 → NORMAL    (mostly INFO logs, system running fine)
  1 → WARNING   (scattered WARN/ERROR logs, degraded performance)
  2 → CRITICAL  (ERROR/CRITICAL floods, system failing)

Architecture:
  Log messages (text) → TF-IDF features → Dense Neural Network → Softmax

Why TensorFlow here (vs PyTorch for metrics)?
  - Demonstrates you can use BOTH frameworks
  - Shows understanding of Keras functional API
  - TF model is saved in SavedModel format (production standard)

This complements the PyTorch LSTM which handles numerical time-series.
Together, they form a hybrid detection pipeline:
  Metrics (PyTorch LSTM) + Logs (TF Classifier) → Sentinel Agent fuses both signals
"""
from __future__ import annotations

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress TF INFO/WARNING noise

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ── Constants ──────────────────────────────────────────────────────────────────

NUM_CLASSES = 3         # NORMAL, WARNING, CRITICAL
VOCAB_SIZE = 2000       # TF-IDF vocabulary size
SEQUENCE_LENGTH = 50    # Number of log entries per window
EMBEDDING_DIM = 32


# ── Model Architecture ─────────────────────────────────────────────────────────

def build_log_classifier(
    text_vectorizer: layers.TextVectorization,
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = 0.3,
) -> keras.Model:
    """
    Build and compile the Dense log classification model.
    
    Takes an already-adapted TextVectorization layer so shapes are fully defined.
    This creates an end-to-end model that accepts raw strings.
    """
    inputs = keras.Input(shape=(), dtype=tf.string, name="log_window_text")

    x = text_vectorizer(inputs)

    x = layers.Dense(256, activation="relu", name="dense_1")(x)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)

    x = layers.Dense(128, activation="relu", name="dense_2")(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Dropout(dropout_rate, name="dropout_2")(x)

    x = layers.Dense(64, activation="relu", name="dense_3")(x)
    x = layers.Dropout(dropout_rate / 2, name="dropout_3")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="NexusOps_LogClassifier")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.SparseCategoricalAccuracy(name="top1_acc"),
            keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2_acc"),
        ],
    )

    return model


def build_log_classifier_with_lstm(
    text_vectorizer: layers.TextVectorization,
    embedding_dim: int = EMBEDDING_DIM,
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = 0.3,
) -> keras.Model:
    """
    Alternative LSTM-based log classifier for comparison.
    Takes an already-adapted TextVectorization layer.
    """
    inputs = keras.Input(shape=(), dtype=tf.string, name="log_text_input")

    x = text_vectorizer(inputs)
    # text_vectorizer must have output_mode="int" for this model
    vocab_size = text_vectorizer.vocabulary_size()
    
    x = layers.Embedding(vocab_size, embedding_dim, name="embedding")(x)
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True), name="bilstm_1"
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(32), name="bilstm_2"
    )(x)
    x = layers.Dense(64, activation="relu", name="dense")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="NexusOps_LogClassifier_LSTM")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
