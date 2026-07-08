"""E2 — faithful reproduction: single-level vs two-level on a benchmark.

Smoke uses the toy; the real run points at SealQA / ALFWorld (mse.benchmarks) with
a real backbone. The paper's absolute numbers used a 31B backbone that does not fit
24 GB locally — we reproduce the method and the qualitative single-vs-two-level gap.
"""
from __future__ import annotations

from pathlib import Path

from mse.config import faithful
from mse.dag import DAG
from mse.evolve import run_evolution

from ._proxies import WeakBackbone, fresh_store, rule_task


def run_e2(llm, task, root, cfg=None):
    cfg = cfg or faithful()
    root = Path(root)
    one = run_evolution(cfg, llm, task, fresh_store(root / "one"), DAG(), two_level=False)
    two = run_evolution(cfg, llm, task, fresh_store(root / "two"), DAG(), two_level=True)
    return {
        "single_level": one.history[-1],
        "two_level": two.history[-1],
        "gain": round(two.history[-1] - one.history[-1], 3),
        "single_curve": one.history,
        "two_curve": two.history,
    }


def smoke(root):
    return run_e2(WeakBackbone(), rule_task(), root)


if __name__ == "__main__":
    print(smoke("runs/e2"))
