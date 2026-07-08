"""E3 — EN->VI meta transfer: does a meta-skill evolved on English tasks speed up
evolution on Vietnamese ones? Signature angle; the paper doesn't test it.

Smoke uses two RuleTask instances as EN/VI stand-ins with the Weak proxy; the real
run uses GSM8K (EN) + vi-gsm8k (VN) via mse.benchmarks.load_gsm8k.
"""
from __future__ import annotations

from pathlib import Path

from mse.config import faithful
from mse.dag import DAG
from mse.evolve import run_evolution
from mse.fast_loop import run_fast
from mse.skills import META_COMPONENTS

from ._proxies import WeakBackbone, fresh_store, rule_task


def run_e3(llm, en_task, vi_task, root, cfg=None):
    cfg = cfg or faithful()
    root = Path(root)

    # Phase A: evolve a meta-skill on EN (two-level)
    en_store = fresh_store(root / "en")
    run_evolution(cfg, llm, en_task, en_store, DAG(), two_level=True)
    evolved_meta = en_store.meta()

    # Phase B (fresh): VI fast loop from the weak seed
    fresh = fresh_store(root / "vi_fresh")
    fresh_curve = run_fast(cfg, llm, vi_task, fresh, DAG()).history

    # Phase B (transfer): VI fast loop seeded with the EN-evolved meta
    trans = fresh_store(root / "vi_trans")
    for sym in META_COMPONENTS:
        trans.write_meta(sym, evolved_meta[sym])
    trans_curve = run_fast(cfg, llm, vi_task, trans, DAG()).history

    return {
        "fresh_final": fresh_curve[-1],
        "transfer_final": trans_curve[-1],
        "fresh_curve": fresh_curve,
        "transfer_curve": trans_curve,
        "transfer_helps": trans_curve[-1] > fresh_curve[-1],
    }


def smoke(root):
    return run_e3(WeakBackbone(), rule_task(), rule_task(), root)


if __name__ == "__main__":
    print(smoke("runs/e3"))
