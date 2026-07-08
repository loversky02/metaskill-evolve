"""Diversity-aware Retriever (σ) via greedy DPP MAP selection.

Vendored (numpy-only, deterministic) from the shared selector in `automem-vn`
(`automem/meta/curate.py`) and `scalable-robot-demo-curation`
(`robocurate/dpp.py`), so metaskill-evolve stays self-contained and testable.
Picks a *diverse* subset of past edits as inspirations rather than the most recent
ones. The paper's ablation finds σ least impactful, so this is a real but minor
tie-in — kept optional (numpy) with a recency fallback.
"""
from __future__ import annotations

import hashlib
import re
from typing import List


def _tok_hash(tok: str, dim: int) -> int:
    return int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim


def _featurize(texts: List[str], dim: int = 64):
    """L2-normalised hashed bag-of-tokens (deterministic; no embedder needed)."""
    import numpy as np

    X = np.zeros((len(texts), dim))
    for i, t in enumerate(texts):
        for tok in re.findall(r"\w+", t.lower()):
            X[i, _tok_hash(tok, dim)] += 1.0
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def _greedy_dpp(L, k: int) -> List[int]:
    """Greedy MAP for a DPP with kernel L: iteratively add the item that most
    increases the log-determinant (diversity)."""
    import numpy as np

    n = L.shape[0]
    selected: List[int] = []
    remaining = list(range(n))
    for _ in range(min(k, n)):
        best, best_gain = None, -1.0
        for j in remaining:
            idx = selected + [j]
            gain = float(np.linalg.det(L[np.ix_(idx, idx)]))
            if gain > best_gain:
                best, best_gain = j, gain
        if best is None:
            break
        selected.append(best)
        remaining.remove(best)
    return selected


def dpp_retrieve(pool: List[str], k: int) -> List[str]:
    """Return up to ``k`` diverse items from ``pool`` (recency fallback if numpy
    is unavailable or anything goes wrong)."""
    items = [s for s in pool if s]
    if not items or k <= 0:
        return []
    if len(items) <= k:
        return items
    try:
        import numpy as np

        F = _featurize(items)
        L = F @ F.T + 1e-6 * np.eye(len(items))
        return [items[i] for i in _greedy_dpp(L, k)]
    except Exception:
        return items[-k:]
