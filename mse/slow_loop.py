"""Slow loop — Alg. 2: evolve the meta-skill m (arXiv:2607.05297).

Every H fast iterations, the *same* five-agent pipeline is applied to the meta
files {ψ,σ,α,π,ε} themselves — the agents improve the very procedure that
improves task skills. That self-application is the recursion; no extra model, no
extra objective.
"""
from __future__ import annotations

import re
from typing import List

from .agents import analyze, evolve, propose
from .skills import META_COMPONENTS


def meta_productivity(deltas: List[float]) -> float:
    """P̂ = (1/|H|) Σ ΔU over recent descendants (empirical mean improvement)."""
    return sum(deltas) / len(deltas) if deltas else 0.0


def _apply_meta_edit(store, epsilon: str, edit_text: str, default_target: str) -> str:
    """Route an edit to a meta file; honours an optional 'TARGET: <symbol>' header."""
    target, body = default_target, edit_text
    m = re.match(r"\s*TARGET:\s*(\w+)\s*\n(.*)", edit_text, re.S)
    if m and m.group(1) in META_COMPONENTS:
        target, body = m.group(1), m.group(2)
    store.write_meta(target, evolve(epsilon, store.read_meta(target), body))
    return target


def slow_iteration(cfg, llm, store, dag, recent_deltas: List[float], it: int,
                   default_target: str = "pi") -> str:
    """One slow-loop step: improve the meta-skill via its own pipeline."""
    P = meta_productivity(recent_deltas)
    trace = (
        f"Meta-failure trace: recent meta-productivity P_hat={P:.3f} over the last "
        f"{len(recent_deltas)} task-skill edits. Improve the improvement pipeline so "
        f"future edits raise utility faster."
    )
    meta = store.meta()
    tag, analysis = analyze(llm, meta["psi"], "meta-improvement", trace, P)

    edited = default_target
    for k in range(cfg.k_meta):  # K_m accumulating edits (child k+1 reads k's writes)
        edit = propose(llm, meta["pi"], trace, analysis, [], k, cfg.k_meta)
        edited = _apply_meta_edit(store, meta["epsilon"], edit, default_target)
        meta = store.meta()  # accumulate

    dag.add(kind="meta", iter=it, utility=P, tag=tag, snapshot=store.snapshot())
    return edited
