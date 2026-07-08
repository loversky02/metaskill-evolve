"""E4 — is recursion worth the tokens? Recursion multiplies inference (you evolve
the evolvers). Report token cost vs final utility for fast-only vs two-level.
"""
from __future__ import annotations

from pathlib import Path

from mse.config import faithful
from mse.dag import DAG
from mse.evolve import run_evolution
from mse.llm import CountingLLM

from ._proxies import WeakBackbone, fresh_store, rule_task


def run_e4(inner_llm, task, root, cfg=None):
    cfg = cfg or faithful()
    root = Path(root)
    rows = []
    for two in (False, True):
        counter = CountingLLM(inner_llm)
        st = run_evolution(cfg, counter, task,
                           fresh_store(root / ("two" if two else "one")),
                           DAG(), two_level=two)
        rows.append({
            "mode": "two_level" if two else "single_level",
            "final_utility": st.history[-1],
            "llm_calls": counter.calls,
            "approx_tokens": counter.tokens,
        })
    return rows


def smoke(root):
    return run_e4(WeakBackbone(), rule_task(), root)


if __name__ == "__main__":
    for row in smoke("runs/e4"):
        print(row)
