"""Visualization helpers for hierarchy and latent trajectories."""
from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from .hierarchy import ClusterNode


def plot_hierarchy(node: ClusterNode, ax=None, x=0, y=0, level_height=1.5):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.scatter([x], [y], s=80, color="tab:blue")
    ax.text(x, y + 0.08, node.id, ha="center", fontsize=8)
    if not node.children:
        return x, x

    span = 0
    child_positions: List[float] = []
    for child in node.children:
        left, right = plot_hierarchy(child, ax, x + span, y - level_height, level_height)
        child_center = (left + right) / 2
        child_positions.append(child_center)
        ax.plot([x, child_center], [y, y - level_height], color="gray", linewidth=1)
        span = right - x + 1
    return min(child_positions), max(child_positions)


def plot_latent_trajectory(latents: List[np.ndarray]):
    latent_arr = np.stack(latents)
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(latent_arr)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(coords[:, 0], coords[:, 1], marker="o", markersize=3, linewidth=1)
    ax.set_title("Latent trajectory (PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    return fig, ax


def plot_temporal_labels(labels: List[str], smoothed: List[str]):
    t = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.plot(t, labels, label="raw", linewidth=1, marker="o")
    ax.plot(t, smoothed, label="smoothed", linewidth=2, marker="o")
    ax.set_xlabel("time step")
    ax.set_ylabel("state")
    ax.legend()
    ax.set_title("Temporal coherence")
    return fig, ax
