"""Tasks and evaluation.

A ``Task`` exposes examples, a rollout (skill -> prediction) and a reward in
[0, 1]. Benchmark adapters (M4: ALFWorld / SealQA / OfficeQA / vi-gsm8k)
implement the same protocol. ``RuleTask`` is a backbone-free toy that makes
utility a deterministic function of skill *content*, so the fast/slow loops can
be verified offline at $0 — isolating the evolution machinery from backbone
reasoning quality.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple


class Task(Protocol):
    desc: str

    def examples(self) -> List[dict]: ...
    def rollout(self, skill_text: str, example: dict, llm=None): ...
    def reward(self, prediction, example) -> float: ...


@dataclass
class RuleTask:
    """N latent rules; example i is solved iff the skill mentions its rule token."""

    n_rules: int = 4
    n_examples: int = 8
    desc: str = "Apply the correct latent rule to each case."

    def rule_token(self, k: int) -> str:
        return f"rule_{k}"

    def examples(self) -> List[dict]:
        return [
            {
                "id": i,
                "required": self.rule_token(i % self.n_rules),
                "text": f"case {i}: requires {self.rule_token(i % self.n_rules)}",
            }
            for i in range(self.n_examples)
        ]

    def describe(self, example: dict) -> str:
        return example["text"]

    def rollout(self, skill_text: str, example: dict, llm=None) -> bool:
        return example["required"] in skill_text

    def reward(self, prediction, example) -> float:
        return 1.0 if prediction else 0.0


def evaluate(task: Task, skill_text: str, examples: List[dict], llm=None) -> float:
    """Mean reward = U(s) = E[r(A_s(x), y)] (arXiv:2607.05297, Eq. 1)."""
    if not examples:
        return 0.0
    return sum(task.reward(task.rollout(skill_text, ex, llm), ex)
               for ex in examples) / len(examples)


def worst_example(task: Task, skill_text: str, examples: List[dict],
                  llm=None) -> Tuple[Optional[dict], float]:
    """The lowest-reward example (ties broken by order) — the fast loop's target."""
    worst, worst_r = None, 2.0
    for ex in examples:
        r = task.reward(task.rollout(skill_text, ex, llm), ex)
        if r < worst_r:
            worst, worst_r = ex, r
    return worst, worst_r
