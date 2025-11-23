# Aurora Synaptic Engine v2 (SpoonOS-Native AI4Science System)

## Purpose and Framing
Synaptic Engine v2 repositions the original TreeHacks concept as a reproducible AI4Science research platform. It focuses on mapping the latent geometry of cognitive activity from multi-modal neural and biometric streams, pairing hierarchical manifold learning with agentic experimentation on SpoonOS. The system targets scientific discovery and hypothesis testing rather than unverifiable BCI performance claims.

## Core Principles
- **Scientific credibility**: Demonstrate clustering, pruning, and temporal coherence with synthetic or recorded signals; avoid unvalidated throughput claims.
- **AI4Science alignment**: Provide workflows for hypothesis generation, manifold analysis, and experiment comparison across tasks or subjects.
- **Agentic backbone**: Four SpoonOS agents orchestrate acquisition, manifold refinement, decoding, and scientist-facing reporting.
- **Buildable in 48–72 hours**: Minimal viable encoder, hierarchical clustering stack, and visualization with clear input→output loops.

## System Architecture
```
Raw EEG-like + biometric streams ──► Multi-Modal Encoder ──► z_t latent
                                      (1D conv → transformer → fusion)
         │
         ▼
Hierarchical Engine
- coarse clusters (k≈5–10)
- recursive subclusters (k≈5–10 each)
- posterior-based pruning
- temporal smoothing (moving window / HMM)
         │
         ▼
State posterior ──► DecoderAgent (text/image/task outputs)
         │
         ▼
SpoonOS Agent Layer
1) AcquisitionAgent
2) ManifoldAgent
3) DecoderAgent
4) NeuroscientistAgent
         │
         ▼
Scientist UI: manifold viz, trajectories, pruning curves, temporal coherence
```

### Multi-Modal Encoder (MVP)
- Inputs: synthetic EEG (sinusoid/ERP mixtures), biometric channels (HRV, EDA), optional text or CLIP embeddings.
- Architecture: lightweight 1D conv stack → transformer encoder → mean pooling; attention-based fusion for optional modalities.
- Outputs: latent vector `z_t ∈ ℝ^D` per window.

### Hierarchical Engine
- **Level 1** coarse clustering (k≈5–10) on latent batches.
- **Level 2+** recursive subclustering per parent cluster (k≈5–10 each) with on-demand refinement.
- **Pruning**: drop subtrees whose posterior mass falls below threshold; logarithmic reduction of search space (target 7–8× per layer in simulation).
- **Temporal coherence**: moving-window smoothing or lightweight HMM to stabilize labels across timesteps; report stability gains (e.g., >40% consistency in tests).

### SpoonOS Agent Roles
- **AcquisitionAgent**: streams real or synthetic data; calls `/encode` to produce latents.
- **ManifoldAgent**: updates hierarchy, triggers subclustering, executes pruning, tracks temporal metrics.
- **DecoderAgent**: infers cognitive state posterior, emits text/image/task-level outputs, exposes uncertainty.
- **NeuroscientistAgent**: designs experiments, compares manifolds across tasks/subjects, surfaces hypotheses and interpretability notes.

## Input→Output Demo Loop
1. Raw window from synthetic EEG + biometrics.
2. Encoder produces latent `z_t`.
3. ManifoldAgent clusters and prunes hierarchy.
4. Temporal smoother refines state; DecoderAgent reports current cognitive hypothesis.
5. UI renders dendrogram, latent trajectory (UMAP/t-SNE), pruning curve, and temporal coherence plot with live updates.

## Scientific Workflows
- **Hypothesis testing**: NeuroscientistAgent suggests contrasts (task A vs B), runs clustering per condition, and reports structural differences.
- **Ablations**: evaluate impact of pruning thresholds, temporal window sizes, and fusion strategies on stability and search-space reduction.
- **Cross-subject comparison**: align manifolds via Procrustes/CCA on shared stimuli, reporting cluster correspondence scores.

## Evaluation Metrics
- Search-space reduction per layer (target 7–8×).
- Temporal consistency improvement from smoothing (report % gain over raw labels).
- Cluster purity/entropy on synthetic labeled datasets.
- Latent trajectory smoothness (e.g., velocity/curvature stats) to evidence manifold structure.

## Neo/Web3 (Optional)
- Log experiment hypotheses and cluster prototypes on-chain for provenance.
- Record agent-to-agent decisions as verifiable events aligned with SpoonOS spec.

## Implementation Plan (48–72 Hours)
1. **Data + Encoder**: generate synthetic EEG/biometric streams; implement 1D conv + transformer encoder with fusion.
2. **Hierarchy Core**: coarse k-means/HDBSCAN, recursive subclustering, posterior-based pruning, and temporal smoothing.
3. **Agent Layer**: four SpoonOS-compliant scripts wrapping encode/cluster/infer/report endpoints.
4. **Visualization**: dendrogram, latent UMAP/t-SNE trajectory, pruning and temporal coherence plots; sidebar for decoded state + uncertainty.
5. **Experiments**: run ablations on pruning thresholds and smoothing windows; capture metrics and screenshots for presentation.

## Deliverables
- Live demo showing streaming data → embeddings → cluster hierarchy → pruned/temporal-stabilized decoding.
- Scientist UI with manifold visualization and interpretability overlays.
- Technical report (PDF) covering architecture, pruning/temporal equations, and ablation results.
- SpoonOS agent configs and scripts for reproducibility.
- GitHub repo documenting workflows and commands for judges.
