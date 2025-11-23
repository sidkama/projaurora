"""Hierarchical clustering, pruning, and inference utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from sklearn.cluster import KMeans


@dataclass
class HierarchyConfig:
    branch_k: int = 6
    max_depth: int = 3
    prune_threshold: float = 0.05
    min_size: int = 5
    temporal_window: int = 5


@dataclass
class ClusterNode:
    id: str
    depth: int
    centroid: np.ndarray
    children: List["ClusterNode"] = field(default_factory=list)
    size: int = 0
    posterior: float = 1.0

    def path(self) -> List[str]:
        return [self.id]


def _kmeans(points: np.ndarray, k: int) -> List[np.ndarray]:
    model = KMeans(n_clusters=k, n_init="auto", random_state=0)
    model.fit(points)
    centroids = model.cluster_centers_
    labels = model.labels_
    partitions: List[np.ndarray] = []
    for i in range(k):
        partitions.append(points[labels == i])
    return centroids, partitions


def build_hierarchy(latents: np.ndarray, config: HierarchyConfig, depth: int = 0, prefix: str = "C") -> ClusterNode:
    centroid = latents.mean(axis=0)
    node = ClusterNode(id=f"{prefix}{depth}", depth=depth, centroid=centroid, size=len(latents))
    if depth >= config.max_depth - 1 or len(latents) < config.min_size:
        return node

    centroids, partitions = _kmeans(latents, config.branch_k)
    posteriors = np.array([len(p) for p in partitions], dtype=float)
    posteriors = posteriors / (posteriors.sum() + 1e-6)

    for i, (child_centroid, pts, posterior) in enumerate(zip(centroids, partitions, posteriors)):
        if len(pts) == 0:
            continue
        child_prefix = f"{node.id}-{i}"
        child_node = build_hierarchy(pts, config, depth + 1, prefix=child_prefix)
        child_node.centroid = child_centroid
        child_node.size = len(pts)
        child_node.posterior = float(posterior)
        node.children.append(child_node)
    return node


def prune(node: ClusterNode, threshold: float) -> Optional[ClusterNode]:
    if node.posterior < threshold:
        return None
    pruned_children = []
    for child in node.children:
        pruned_child = prune(child, threshold)
        if pruned_child is not None:
            pruned_children.append(pruned_child)
    node.children = pruned_children
    return node


def infer_state(latent: np.ndarray, node: ClusterNode) -> Dict[str, float]:
    scores = {}
    stack = [node]
    while stack:
        current = stack.pop()
        distance = float(np.linalg.norm(latent - current.centroid))
        scores[current.id] = distance
        stack.extend(current.children)
    # Convert distances to pseudo-posteriors
    inv = np.exp(-np.array(list(scores.values())))
    inv = inv / (inv.sum() + 1e-6)
    return {node_id: float(p) for node_id, p in zip(scores.keys(), inv)}


def temporal_smooth(labels: List[str], window: int) -> List[str]:
    if window <= 1:
        return labels
    smoothed: List[str] = []
    for i in range(len(labels)):
        start = max(0, i - window + 1)
        window_labels = labels[start : i + 1]
        counts: Dict[str, int] = {}
        for lbl in window_labels:
            counts[lbl] = counts.get(lbl, 0) + 1
        dominant = max(counts, key=counts.get)
        smoothed.append(dominant)
    return smoothed
