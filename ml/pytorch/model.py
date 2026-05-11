"""
PyTorch LSTM Autoencoder for Time-Series Anomaly Detection
===========================================================

Architecture:
  Input  → LSTM Encoder → Bottleneck → LSTM Decoder → Reconstructed Output

Training objective:
  Minimize reconstruction error (MSE) on NORMAL metric sequences.
  At inference time, sequences with high reconstruction error are anomalies.

This is an UNSUPERVISED approach — no anomaly labels needed during training.
The model learns what "normal" looks like, then flags deviations.

Features (5 per time step):
  [cpu_usage_percent, memory_usage_percent, request_latency_ms,
   error_rate_percent, network_io_kbps]
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class LSTMEncoder(nn.Module):
    """
    Encodes a metric sequence into a compact latent representation.
    
    Args:
        input_size:  Number of features per time step (default: 5)
        hidden_size: LSTM hidden state dimension
        num_layers:  Number of stacked LSTM layers
        dropout:     Dropout between LSTM layers (only if num_layers > 1)
    """

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,          # input shape: (batch, seq_len, features)
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            output: (batch_size, seq_len, hidden_size) — all hidden states
            (h_n, c_n): final hidden and cell states
        """
        output, (h_n, c_n) = self.lstm(x)
        return output, (h_n, c_n)


class LSTMDecoder(nn.Module):
    """
    Decodes the latent representation back to the original sequence.
    Uses the encoder's final hidden state as the initial decoder state.
    """

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=hidden_size,    # decoder input is encoder hidden states
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Project back to original feature space
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(
        self,
        encoder_output: Tensor,
        encoder_hidden: tuple[Tensor, Tensor],
    ) -> Tensor:
        """
        Args:
            encoder_output: (batch_size, seq_len, hidden_size)
            encoder_hidden: (h_n, c_n) from the encoder
        Returns:
            reconstruction: (batch_size, seq_len, input_size)
        """
        decoder_output, _ = self.lstm(encoder_output, encoder_hidden)
        reconstruction = self.output_layer(decoder_output)
        return reconstruction


class LSTMAutoencoder(nn.Module):
    """
    Full LSTM Autoencoder = Encoder + Decoder.

    During training:  forward() → compare reconstruction to input → MSE loss
    During inference: compute per-sample reconstruction error → anomaly score

    Example:
        model = LSTMAutoencoder(input_size=5, hidden_size=64, seq_len=30)
        x = torch.randn(32, 30, 5)   # batch of 32, window of 30 time steps
        reconstruction = model(x)
        loss = model.reconstruction_loss(x, reconstruction)
    """

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 64,
        num_layers: int = 2,
        seq_len: int = 30,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len

        self.encoder = LSTMEncoder(input_size, hidden_size, num_layers, dropout)
        self.decoder = LSTMDecoder(input_size, hidden_size, num_layers, dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Full autoencoder forward pass.
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            reconstruction: (batch_size, seq_len, input_size)
        """
        encoder_output, encoder_hidden = self.encoder(x)
        reconstruction = self.decoder(encoder_output, encoder_hidden)
        return reconstruction

    def reconstruction_loss(self, original: Tensor, reconstruction: Tensor) -> Tensor:
        """Mean Squared Error between original and reconstructed sequences."""
        return nn.functional.mse_loss(reconstruction, original, reduction="mean")

    @torch.no_grad()
    def anomaly_score(self, x: Tensor) -> Tensor:
        """
        Compute per-sample anomaly scores (0.0 → normal, 1.0 → anomalous).

        The score is the MSE reconstruction error for each sample, normalized
        via a sigmoid so it maps to [0, 1].

        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            scores: (batch_size,) — one score per sample
        """
        self.eval()
        reconstruction = self.forward(x)
        # Per-sample MSE: average over seq_len and features
        per_sample_mse = ((reconstruction - x) ** 2).mean(dim=[1, 2])
        # Sigmoid normalization (scale factor of 10 works well empirically)
        scores = torch.sigmoid(per_sample_mse * 10)
        return scores

    def save(self, path: str) -> None:
        """Save model weights and config."""
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "config": {
                    "input_size": self.input_size,
                    "hidden_size": self.hidden_size,
                    "seq_len": self.seq_len,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "LSTMAutoencoder":
        """Load a saved model."""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint["config"]
        model = cls(**config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model
