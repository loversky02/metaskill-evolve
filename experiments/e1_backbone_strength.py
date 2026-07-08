"""E1 (headline) — does the meta-loop's value shrink as the backbone strengthens?

Not in the paper. Offline smoke uses proxy backbones; the real run swaps in
Gemma-4 E2B/E4B (MLX, local) and GPT-5.5/Claude (9router) from mse.config.BACKBONES.
"""
from __future__ import annotations

from pathlib import Path

from mse.config import faithful
from mse.dag import DAG
from mse.evolve import run_evolution

from ._proxies import STRENGTH_PROXIES, fresh_store, rule_task


def _final(llm, root, cfg, task, two_level) -> float:
    st = run_evolution(cfg, llm, task, fresh_store(root), DAG(), two_level=two_level)
    return st.history[-1]


def run_e1(backbones, root, cfg=None):
    """backbones: list of (name, llm). Returns one row per backbone."""
    cfg = cfg or faithful()
    root = Path(root)
    rows = []
    for name, llm in backbones:
        task = rule_task()
        one = _final(llm, root / f"{name}_one", cfg, task, False)
        two = _final(llm, root / f"{name}_two", cfg, task, True)
        rows.append({"backbone": name, "single": one, "two_level": two,
                     "meta_gain": round(two - one, 3)})
    return rows


def smoke(root):
    return run_e1([(n, cls()) for n, cls in STRENGTH_PROXIES], root)


def format_table(rows) -> str:
    out = ["| backbone | single-level | two-level | meta-gain |",
           "|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['backbone']} | {r['single']:.2f} | {r['two_level']:.2f} "
                   f"| {r['meta_gain']:+.2f} |")
    return "\n".join(out)


if __name__ == "__main__":
    print(format_table(smoke("runs/e1")))
