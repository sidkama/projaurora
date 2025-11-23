"""Synthetic data generation for EEG-like and biometric streams."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class SyntheticConfig:
    sample_rate: int = 256
    window_seconds: float = 2.0
    n_channels: int = 8
    biometric_dim: int = 3  # e.g., HRV, EDA, respiration
    n_classes: int = 4
    noise_std: float = 0.15
    seed: int | None = 13

    @property
    def window_size(self) -> int:
        return int(self.sample_rate * self.window_seconds)


def _erp_waveform(length: int, peak: int, width: int = 12, amplitude: float = 1.0) -> np.ndarray:
    t = np.arange(length)
    return amplitude * np.exp(-0.5 * ((t - peak) / width) ** 2)


def _sinusoid(length: int, freq: float, phase: float = 0.0, amplitude: float = 1.0) -> np.ndarray:
    t = np.linspace(0, 1, length)
    return amplitude * np.sin(2 * math.pi * freq * t + phase)


def generate_batch(batch_size: int, config: SyntheticConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.seed)
    eeg = rng.normal(scale=config.noise_std, size=(batch_size, config.n_channels, config.window_size))
    biometrics = rng.normal(scale=config.noise_std, size=(batch_size, config.biometric_dim))
    labels = rng.integers(0, config.n_classes, size=batch_size)

    for i, label in enumerate(labels):
        peak = rng.integers(config.window_size // 4, 3 * config.window_size // 4)
        freq = 6 + 2 * label
        phase = rng.uniform(0, math.pi)
        amplitude = 1 + 0.1 * label

        waveform = _erp_waveform(config.window_size, peak, amplitude=amplitude)
        carrier = _sinusoid(config.window_size, freq=freq, phase=phase, amplitude=0.6)

        eeg[i, 0] += waveform  # event-related peak on channel 0
        eeg[i, 1] += carrier   # rhythmic component on channel 1
        eeg[i, 2] += 0.6 * waveform + 0.5 * carrier

        biometrics[i, 0] += 0.2 * label  # HRV-like trend
        biometrics[i, 1] += rng.normal(0.1 * label)
        biometrics[i, 2] += rng.normal(0.05 * (label + 1))

    return eeg.astype(np.float32), biometrics.astype(np.float32), labels.astype(int)


def generate_stream(num_windows: int, config: SyntheticConfig) -> List[Dict[str, np.ndarray]]:
    eeg_batch, bio_batch, labels = generate_batch(num_windows, config)
    stream = []
    for i in range(num_windows):
        stream.append({
            "eeg": eeg_batch[i],
            "biometric": bio_batch[i],
            "label": labels[i],
        })
    return stream
