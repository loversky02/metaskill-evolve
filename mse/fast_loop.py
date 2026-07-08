"""Fast loop — Alg. 1: evolve the task skill s (arXiv:2607.05297).

One iteration: restore the current branch, find the worst case, run the five-agent
pipeline (Analyzer -> Retriever -> Allocator -> Proposer -> Evolver), spawn K
children, evaluate them, commit to the DAG, and advance to the best child. The
slow loop (M3) hooks in every ``H`` iterations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .agents import allocate, analyze, evolve, propose, retrieve
from .tasks import Task, evaluate, worst_example


@dataclass
class FastState:
    best_skill: str
    best_utility: float
    history: List[float]  # utility after each iteration (index 0 = root)


def fast_iteration(cfg, llm, task: Task, store, dag, examples, val_examples,
                   parent_id: int, it: int, inspirations: List[str]) -> tuple:
    """One fast-loop step; returns (best_child_id, best_child_utility)."""
    skill = store.read_task()
    meta = store.meta()

    # worst case -> analysis -> inspirations -> child budget K
    wc, wc_r = worst_example(task, skill, examples, llm)
    tag, analysis = analyze(llm, meta["psi"], task.desc, wc["text"], wc_r)
    insp = retrieve(llm, meta["sigma"], tag, inspirations, cfg.l_same)
    K = allocate(llm, meta["alpha"], analysis, cfg.k_max)

    # propose K children, evolve, evaluate on val, commit to DAG
    best_id, best_u, best_skill = parent_id, evaluate(task, skill, val_examples, llm), skill
    for k in range(K):
        edit = propose(llm, meta["pi"], wc["text"], analysis, insp, k, K)
        child = evolve(meta["epsilon"], skill, edit)
        u = evaluate(task, child, val_examples, llm)
        snap = store.snapshot()
        snap["skill.md"] = child
        cid = dag.add(parent_id=parent_id, kind="task", iter=it, utility=u,
                      tag=tag, snapshot=snap)
        inspirations.append(edit)
        if u > best_u:
            best_id, best_u, best_skill = cid, u, child

    store.write_task(best_skill)  # advance the frontier (top-1 for now)
    return best_id, best_u


def run_fast(cfg, llm, task: Task, store, dag,
             examples: Optional[List[dict]] = None,
             val_examples: Optional[List[dict]] = None) -> FastState:
    """Run the fast loop for ``cfg.fast_iterations`` steps."""
    examples = examples if examples is not None else task.examples()
    val_examples = val_examples if val_examples is not None else examples

    skill = store.read_task()
    u0 = evaluate(task, skill, val_examples, llm)
    parent_id = dag.add(kind="task", iter=0, utility=u0, snapshot=store.snapshot())
    history = [u0]
    inspirations: List[str] = []

    for it in range(1, cfg.fast_iterations + 1):
        parent_id, u = fast_iteration(cfg, llm, task, store, dag, examples,
                                       val_examples, parent_id, it, inspirations)
        history.append(u)

    return FastState(best_skill=store.read_task(), best_utility=history[-1],
                     history=history)
