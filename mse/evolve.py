"""Two-timescale driver: fast loop + slow loop under the H schedule.

``two_level=False`` gives the *single-level* baseline (fast loop only, meta-skill
frozen) — the core ablation that isolates what the recursion actually adds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .fast_loop import fast_iteration
from .slow_loop import slow_iteration
from .tasks import Task, evaluate


@dataclass
class EvolveState:
    history: List[float]      # utility after each iteration (index 0 = root)
    best_skill: str
    meta: Dict[str, str]
    two_level: bool


def run_evolution(cfg, llm, task: Task, store, dag, *, two_level: bool = True,
                  examples: Optional[List[dict]] = None,
                  val_examples: Optional[List[dict]] = None) -> EvolveState:
    examples = examples if examples is not None else task.examples()
    val = val_examples if val_examples is not None else examples

    skill = store.read_task()
    u_prev = evaluate(task, skill, val, llm)
    parent = dag.add(kind="task", iter=0, utility=u_prev, snapshot=store.snapshot())
    hist: List[float] = [u_prev]
    deltas: List[float] = []
    insp: List[str] = []

    for it in range(1, cfg.fast_iterations + 1):
        parent, u = fast_iteration(cfg, llm, task, store, dag, examples, val,
                                   parent, it, insp)
        deltas.append(u - u_prev)
        u_prev = u
        hist.append(u)
        if two_level and it % cfg.meta_horizon_H == 0:
            slow_iteration(cfg, llm, store, dag, deltas[-cfg.meta_horizon_H:], it)

    return EvolveState(history=hist, best_skill=store.read_task(),
                       meta=store.meta(), two_level=two_level)
