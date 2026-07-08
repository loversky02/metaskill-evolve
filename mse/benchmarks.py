"""Benchmark adapters — all implement the ``Task`` protocol so the two-timescale
engine runs on them unchanged.

* ``QATask``        - generic question answering (SealQA, OfficeQA, GSM8K, vi-gsm8k)
* ``ALFWorldTask``  - sequential embodied text env (+ ``MockAlfEnv`` for $0 tests)

Live dataset loaders (``load_sealqa``, ``load_gsm8k``) are guarded: they import
``datasets`` lazily so the core stays dependency-free and offline-testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .scoring import contains_scorer, numeric_scorer


# --------------------------------------------------------------------------- #
# Question answering (SealQA / OfficeQA / GSM8K / vi-gsm8k)
# --------------------------------------------------------------------------- #
@dataclass
class QATask:
    """prediction = backbone(skill + question); reward = scorer(prediction, item)."""

    items: List[dict]                       # [{"question": ..., "answer": ...}, ...]
    scorer: Callable                        # from mse.scoring
    desc: str = "Answer the question. Reason step by step, then give a final answer."
    system_preamble: str = "You are a careful problem solver. Use the skill below.\n"

    def examples(self) -> List[dict]:
        return self.items

    def rollout(self, skill_text: str, example: dict, llm=None) -> str:
        if llm is None:
            return ""
        system = self.system_preamble + skill_text
        return llm.complete(system, f"Question: {example['question']}\nFinal answer:")

    def reward(self, prediction, example) -> float:
        return self.scorer(prediction, example)


def load_gsm8k(split: str = "test", n: Optional[int] = None,
               vietnamese: bool = False) -> QATask:
    """GSM8K (EN) or vi-gsm8k (VN). Numeric-match reward. Needs `datasets`."""
    from datasets import load_dataset

    if vietnamese:
        ds = load_dataset("vuongtsc/vi-gsm8k-agentic")["train"]
        items = [{"question": r["question"], "answer": r["answer"]} for r in ds]
    else:
        ds = load_dataset("gsm8k", "main")[split]
        items = [{"question": r["question"],
                  "answer": r["answer"].split("####")[-1].strip()} for r in ds]
    return QATask(items[:n], numeric_scorer(), desc="Solve the math word problem.")


def load_sealqa(config: str = "seal_0", n: Optional[int] = None,
                judge_llm=None) -> QATask:
    """SealQA (HF vtllms/sealqa). Free-form; LLM-judge if provided, else contains.
    Needs `datasets`."""
    from datasets import load_dataset
    from .scoring import judge_scorer

    ds = load_dataset("vtllms/sealqa", config)["test"]
    items = [{"question": r["question"], "answer": r["answer"]} for r in ds][:n]
    scorer = judge_scorer(judge_llm) if judge_llm is not None else contains_scorer()
    return QATask(items, scorer, desc="Answer using careful search-style reasoning.")


# --------------------------------------------------------------------------- #
# ALFWorld (sequential embodied text env)
# --------------------------------------------------------------------------- #
class MockAlfEnv:
    """Tiny deterministic text env standing in for ALFWorld: success iff the agent
    issues the goal action within a few steps. Keeps the adapter testable at $0."""

    def __init__(self, goal_action: str = "take apple", max_steps: int = 3):
        self.goal = goal_action
        self.max_steps = max_steps
        self.steps = 0

    def reset(self) -> str:
        self.steps = 0
        return f"Goal: {self.goal}. You are in a room. What do you do?"

    def step(self, action: str):
        self.steps += 1
        done = self.goal in action.lower()
        obs = "You succeeded." if done else "Nothing happens."
        return obs, (1.0 if done else 0.0), (done or self.steps >= self.max_steps)


@dataclass
class ALFWorldTask:
    """Runs an agent (skill + backbone) in a text env; reward = task success."""

    env_factory: Callable[[dict], object]   # example -> env with reset()/step()
    n: int = 4
    max_steps: int = 10
    desc: str = "Complete the household task by issuing text actions."
    system_preamble: str = "You are an embodied agent. Follow the skill below.\n"

    def examples(self) -> List[dict]:
        return [{"id": i} for i in range(self.n)]

    def rollout(self, skill_text: str, example: dict, llm=None) -> float:
        env = self.env_factory(example)
        obs = env.reset()
        success = 0.0
        for _ in range(self.max_steps):
            action = "" if llm is None else llm.complete(
                self.system_preamble + skill_text, f"Observation: {obs}\nAction:")
            obs, r, done = env.step(action)
            if done:
                success = r
                break
        return success

    def reward(self, prediction, example) -> float:
        return float(prediction)
