"""SpoonOS-oriented agent scaffolding for the Synaptic Engine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .encoder import EncoderConfig, MultiModalEncoder
from .hierarchy import HierarchyConfig, ClusterNode, build_hierarchy, infer_state, prune, temporal_smooth
from .synthetic import SyntheticConfig, generate_stream


@dataclass
class AgentArtifacts:
    latents: List[np.ndarray]
    hierarchy: ClusterNode
    pruned_hierarchy: ClusterNode
    raw_labels: List[str]
    smoothed_labels: List[str]


class AcquisitionAgent:
    def __init__(self, config: SyntheticConfig, encoder: MultiModalEncoder):
        self.config = config
        self.encoder = encoder

    def run(self, num_windows: int) -> Dict[str, List[np.ndarray]]:
        stream = generate_stream(num_windows, self.config)
        latents = []
        for sample in stream:
            enc = self.encoder.encode(sample["eeg"], sample["biometric"])
            latents.append(enc["latent"])
        return {"latents": latents, "labels": [s["label"] for s in stream]}


class ManifoldAgent:
    def __init__(self, config: HierarchyConfig):
        self.config = config

    def build_and_prune(self, latents: List[np.ndarray]) -> Dict[str, ClusterNode]:
        latent_arr = np.stack(latents)
        hierarchy = build_hierarchy(latent_arr, self.config)
        pruned = prune(hierarchy, self.config.prune_threshold)
        return {"hierarchy": hierarchy, "pruned": pruned}


class DecoderAgent:
    def __init__(self):
        pass

    def decode(self, latents: List[np.ndarray], hierarchy: ClusterNode) -> List[str]:
        labels: List[str] = []
        for latent in latents:
            posterior = infer_state(latent, hierarchy)
            labels.append(max(posterior, key=posterior.get))
        return labels


class NeuroscientistAgent:
    def __init__(self, window: int):
        self.window = window

    def analyze_temporal_coherence(self, labels: List[str]) -> Dict[str, List[str]]:
        smoothed = temporal_smooth(labels, self.window)
        return {"raw": labels, "smoothed": smoothed}


class SpoonOSOrchestrator:
    def __init__(self, synth_cfg: SyntheticConfig, enc_cfg: EncoderConfig, hier_cfg: HierarchyConfig):
        self.encoder = MultiModalEncoder(enc_cfg)
        self.acquisition = AcquisitionAgent(synth_cfg, self.encoder)
        self.manifold = ManifoldAgent(hier_cfg)
        self.decoder = DecoderAgent()
        self.neuroscientist = NeuroscientistAgent(hier_cfg.temporal_window)

    def run_pipeline(self, num_windows: int) -> AgentArtifacts:
        acquisition_out = self.acquisition.run(num_windows)
        manifold_out = self.manifold.build_and_prune(acquisition_out["latents"])
        decoded = self.decoder.decode(acquisition_out["latents"], manifold_out["pruned"])
        temporal = self.neuroscientist.analyze_temporal_coherence(decoded)

        return AgentArtifacts(
            latents=acquisition_out["latents"],
            hierarchy=manifold_out["hierarchy"],
            pruned_hierarchy=manifold_out["pruned"],
            raw_labels=temporal["raw"],
            smoothed_labels=temporal["smoothed"],
        )
