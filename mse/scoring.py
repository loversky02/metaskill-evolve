"""Reward scorers for the benchmark adapters.

Each factory returns a ``scorer(prediction, example) -> float in [0,1]``:

* ``exact_scorer``   - normalised string equality
* ``numeric_scorer`` - last-number match within relative tolerance (OfficeQA/GSM8K)
* ``contains_scorer``- gold appears in the prediction (offline free-form fallback)
* ``judge_scorer``   - an LLM grader replies YES/NO (SealQA-style)
"""
from __future__ import annotations

import re


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _last_num(s):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", str(s))
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", ""))
    except ValueError:
        return None


def exact_scorer():
    def s(pred, ex):
        return 1.0 if _norm(pred) == _norm(ex["answer"]) else 0.0
    return s


def numeric_scorer(tol: float = 0.01):
    """Relative-tolerance numeric match (tol=0.01 -> within 1%)."""
    def s(pred, ex):
        p, g = _last_num(pred), _last_num(ex["answer"])
        if p is None or g is None:
            return 0.0
        return 1.0 if abs(p - g) <= tol * max(1e-9, abs(g)) else 0.0
    return s


def contains_scorer():
    def s(pred, ex):
        return 1.0 if _norm(ex["answer"]) in _norm(pred) else 0.0
    return s


def judge_scorer(judge_llm):
    """LLM-judge (SealQA free-form). Costs one backbone call per scored example."""
    def s(pred, ex):
        out = judge_llm.complete(
            "You are a strict grader. Reply with exactly YES or NO.",
            f"Question: {ex['question']}\nReference answer: {ex['answer']}\n"
            f"Prediction: {pred}\nIs the prediction correct?",
        )
        return 1.0 if "yes" in out.lower() else 0.0
    return s
