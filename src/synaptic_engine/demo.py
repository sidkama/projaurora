"""End-to-end demo pipeline demonstrating hierarchical clustering, pruning, and smoothing."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .agents import SpoonOSOrchestrator
from .encoder import EncoderConfig
from .hierarchy import HierarchyConfig
from .synthetic import SyntheticConfig
from .visualize import plot_hierarchy, plot_latent_trajectory, plot_temporal_labels


def run_demo(output_dir: Path, num_windows: int = 128):
    synth_cfg = SyntheticConfig()
    enc_cfg = EncoderConfig()
    hier_cfg = HierarchyConfig()

    orchestrator = SpoonOSOrchestrator(synth_cfg, enc_cfg, hier_cfg)
    artifacts = orchestrator.run_pipeline(num_windows)

    output_dir.mkdir(parents=True, exist_ok=True)

    fig1, _ = plot_latent_trajectory(artifacts.latents)
    fig1.savefig(output_dir / "latent_trajectory.png", dpi=200, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    plot_hierarchy(artifacts.pruned_hierarchy, ax=ax2)
    ax2.set_title("Pruned hierarchy")
    fig2.savefig(output_dir / "hierarchy.png", dpi=200, bbox_inches="tight")
    plt.close(fig2)

    fig3, _ = plot_temporal_labels(artifacts.raw_labels, artifacts.smoothed_labels)
    fig3.savefig(output_dir / "temporal_coherence.png", dpi=200, bbox_inches="tight")
    plt.close(fig3)

    reductions = []
    stack = [(artifacts.hierarchy, 1.0)]
    while stack:
        node, parent_mass = stack.pop()
        if node.children:
            child_mass = sum(child.posterior for child in node.children)
            reduction = (len(node.children) or 1) / max(child_mass, 1e-6)
            reductions.append(reduction)
        for child in node.children:
            stack.append((child, node.posterior))

    print("Pipeline complete\n---")
    print(f"Latents: {len(artifacts.latents)} windows")
    print(f"Hierarchy depth: {hier_cfg.max_depth}")
    print(f"Average branch reduction (~search-space): {np.mean(reductions):.2f}x")
    mismatch = sum(r != s for r, s in zip(artifacts.raw_labels, artifacts.smoothed_labels))
    stability_gain = 1 - mismatch / max(len(artifacts.raw_labels), 1)
    print(f"Temporal smoothing stability: {stability_gain * 100:.1f}%")
    print(f"Saved plots to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run Synaptic Engine v2 demo pipeline")
    parser.add_argument("--output", type=Path, default=Path("artifacts"), help="Directory to store plots")
    parser.add_argument("--windows", type=int, default=128, help="Number of windows to simulate")
    args = parser.parse_args()
    run_demo(args.output, args.windows)


if __name__ == "__main__":
    main()
