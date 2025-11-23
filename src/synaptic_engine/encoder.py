"""Lightweight multi-modal encoder for synthetic EEG and biometrics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class EncoderConfig:
    eeg_channels: int = 8
    conv_kernel: int = 9
    conv_stride: int = 2
    latent_dim: int = 32
    attention_heads: int = 4
    seed: int | None = 0


class MultiModalEncoder:
    def __init__(self, config: EncoderConfig):
        self.config = config
        rng = np.random.default_rng(config.seed)
        self.conv_filters = rng.normal(scale=0.1, size=(config.eeg_channels, config.conv_kernel))
        self.projection = rng.normal(scale=0.2, size=(config.eeg_channels, config.latent_dim))
        self.bio_projection = rng.normal(scale=0.2, size=(config.latent_dim,))
        self.attn_query = rng.normal(scale=0.1, size=(config.attention_heads, config.latent_dim))

    def _conv1d(self, eeg_window: np.ndarray) -> np.ndarray:
        channels, length = eeg_window.shape
        kernel = self.config.conv_kernel
        stride = self.config.conv_stride
        output_length = (length - kernel) // stride + 1
        convolved = np.zeros((channels, output_length))
        for c in range(channels):
            for i in range(output_length):
                segment = eeg_window[c, i * stride : i * stride + kernel]
                convolved[c, i] = np.dot(segment, self.conv_filters[c])
        return np.tanh(convolved)

    def _temporal_attention(self, conv_out: np.ndarray) -> np.ndarray:
        # conv_out: (channels, steps)
        heads = self.config.attention_heads
        steps = conv_out.shape[1]
        head_embeddings = []
        for h in range(heads):
            query = self.attn_query[h]
            # project conv_out to latent_dim via projection per channel
            projected = conv_out.T @ self.projection  # (steps, latent_dim)
            scores = projected @ query
            weights = np.exp(scores - scores.max())
            weights /= weights.sum() + 1e-6
            head_embeddings.append((weights[:, None] * projected).sum(axis=0))
        return np.mean(head_embeddings, axis=0)

    def encode(self, eeg_window: np.ndarray, biometrics: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        conv_out = self._conv1d(eeg_window)
        attn_embedding = self._temporal_attention(conv_out)

        if biometrics is not None:
            bio_vector = biometrics @ self.bio_projection
            fused = attn_embedding + 0.3 * bio_vector
        else:
            fused = attn_embedding

        return {
            "latent": fused.astype(np.float32),
            "conv": conv_out.astype(np.float32),
            "attn": attn_embedding.astype(np.float32),
        }
