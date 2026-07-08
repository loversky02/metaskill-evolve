"""Configuration for MetaSkill-Evolve.

Hyperparameters mirror arXiv:2607.05297 (*MetaSkill-Evolve: Recursive
Self-Improvement of LLM Agents via Two-Timescale Meta-Skill Evolution*).
``faithful()`` reproduces the paper's default run; override any field for an
ablation — most importantly ``BackboneSpec`` for the backbone-strength study.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Tuple


@dataclass(frozen=True)
class BackboneSpec:
    """The single frozen model that plays all five pipeline agents.

    The paper shares one frozen backbone (Gemma-4 31B) across Analyzer,
    Retriever, Allocator, Proposer, and Evolver. Swapping this spec is the whole
    backbone-strength experiment: mock (offline), an OpenAI-compatible endpoint
    (e.g. a 9router gateway -> GPT-5.5 / Claude), or a local MLX model.
    """

    name: str = "gemma-4-31b"          # paper's frozen backbone
    provider: str = "mock"             # "mock" | "openai" | "mlx"
    base_url: Optional[str] = None     # OpenAI-compatible endpoint, if any
    temperature: float = 0.7
    max_tokens: int = 1024


@dataclass(frozen=True)
class Config:
    """Full run configuration. Defaults == the paper's faithful setting."""

    # --- two-timescale schedule (Alg. 1 fast loop / Alg. 2 slow loop) ---
    fast_iterations: int = 5           # task-skill updates on the fast loop
    meta_horizon_H: int = 2            # meta-skill update every H fast iterations

    # --- child budgets ---
    k_init: int = 2                    # Allocator's initial child budget K
    k_max: int = 3                     # K in [1, k_max]
    k_meta: int = 2                    # accumulating child edits on the slow loop (K_m)

    # --- frontier (Eq. 4): keep top-K_F branches with decaying weights ---
    frontier_size: int = 3             # K_F
    frontier_weights: Tuple[float, ...] = (1.0, 0.5, 0.25)  # eta_1..eta_3

    # --- retrieval (Appendix D) ---
    p_cross: float = 0.2               # prob. of drawing a cross-branch inspiration
    l_same: int = 3                    # same-branch inspirations
    l_cross: int = 2                   # cross-branch inspirations

    backbone: BackboneSpec = field(default_factory=BackboneSpec)
    seed: int = 0

    def with_backbone(self, **kw) -> "Config":
        """Return a copy with an overridden backbone (for the strength study)."""
        return replace(self, backbone=replace(self.backbone, **kw))


def faithful() -> Config:
    """The paper's default configuration (Gemma-4 31B, H=2, 5 fast iters)."""
    return Config()


# --- backbone registry: the *strength axis* for the backbone-strength study ---
# Weak end runs locally on a 24 GB Mac (small MLX models); strong end goes
# through an OpenAI-compatible gateway (9router -> GPT-5.5 / Claude). The paper's
# own backbone (Gemma-4 31B, ~20 GB at 4-bit) does NOT fit 24 GB locally — reach
# it only through a hosted route, if one exists. Exact MLX repo ids for the
# E-series are best-effort and confirmed at M5 (they may need mlx_vlm to load).
BACKBONES = {
    # weak end -- local Apple-Silicon, fits 24 GB
    "gemma-4-e2b": BackboneSpec(name="mlx-community/gemma-4-e2b-it-4bit", provider="mlx"),
    "gemma-4-e4b": BackboneSpec(name="mlx-community/gemma-4-e4b-it-4bit", provider="mlx"),
    # paper's backbone -- needs >24 GB locally, or a hosted route
    "gemma-4-31b": BackboneSpec(name="mlx-community/gemma-4-31B-it-OptiQ-4bit", provider="mlx"),
    # strong end -- via 9router (OpenAI-compatible)
    "gpt-5.5": BackboneSpec(name="gpt-5.5", provider="openai"),
    "claude": BackboneSpec(name="claude-sonnet-5", provider="openai"),
    # offline / tests
    "mock": BackboneSpec(name="mock", provider="mock"),
}

# approximate capability ladder for the sweep (weak -> strong)
STRENGTH_ORDER = ["gemma-4-e2b", "gemma-4-e4b", "gemma-4-31b", "claude", "gpt-5.5"]


def backbone(key: str) -> BackboneSpec:
    """Look up a named backbone from the registry (see ``BACKBONES``)."""
    if key not in BACKBONES:
        raise KeyError(f"unknown backbone {key!r}; known: {sorted(BACKBONES)}")
    return BACKBONES[key]
