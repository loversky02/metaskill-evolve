"""The five improvement-pipeline agents (arXiv:2607.05297, Sec. 2).

Each agent reads its meta-skill file — the prompt that governs it — and calls the
shared frozen backbone. Parsing is deliberately tolerant so weak backbones can
still drive the loop. The slow loop (M3) rewrites these same meta files, which is
what makes the whole system recursive.
"""
from __future__ import annotations

import json
import re
from typing import List, Tuple


def _parse_json(text: str, default: dict) -> dict:
    """Tolerant JSON parse: whole string, else first {...} block, else default."""
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return default


def analyze(llm, psi: str, task_desc: str, failed_case: str, reward: float) -> Tuple[str, str]:
    """ψ: map a failure to a tag + free-form analysis."""
    user = (
        f"Task: {task_desc}\nFailed case: {failed_case}\nReward obtained: {reward}\n"
        'Return JSON: {"tag": "<short_snake_tag>", "analysis": "<root cause>"}'
    )
    d = _parse_json(llm.complete(psi, user, json_mode=True),
                    {"tag": "unknown", "analysis": ""})
    return str(d.get("tag", "unknown")), str(d.get("analysis", ""))


def retrieve(llm, sigma: str, tag: str, pool: List[str], l_same: int,
             *, diverse: bool = True) -> List[str]:
    """σ: select up to ``l_same`` inspirations from past edits. With ``diverse``
    (default) a greedy-DPP picks a varied subset (the automem/robocurate tie-in);
    otherwise fall back to the most recent edits."""
    items = [s for s in pool if s]
    if diverse:
        from .retriever_dpp import dpp_retrieve
        return dpp_retrieve(items, l_same)
    return items[-l_same:]


def allocate(llm, alpha: str, analysis: str, k_max: int) -> int:
    """α: choose the child budget K ∈ [1, k_max]."""
    user = (
        f"Analysis: {analysis}\nChoose the child budget.\n"
        f"Return a single integer K between 1 and {k_max}."
    )
    m = re.search(r"\d+", llm.complete(alpha, user))
    return max(1, min(k_max, int(m.group(0)))) if m else 1


def propose(llm, pi: str, worst_case: str, analysis: str,
            inspirations: List[str], k: int, K: int) -> str:
    """π: emit one concrete edit to the task skill (diversity hint when K>1)."""
    hint = "" if K == 1 else f"\nYou are variant {k + 1} of {K}: take a distinct angle."
    user = (
        f"Worst case: {worst_case}\nAnalysis: {analysis}\n"
        f"Inspirations: {inspirations}{hint}\n"
        "Return ONLY the text to append to the task skill (a new rule/instruction)."
    )
    return llm.complete(pi, user).strip()


def evolve(epsilon: str, skill_text: str, edit_text: str) -> str:
    """ε: apply the edit and verify. Mechanical append + non-empty check; the
    ε meta file governs the acceptance policy for real backbones."""
    edit = edit_text.strip()
    if not edit:
        return skill_text  # reject empty edit
    return skill_text.rstrip() + "\n" + edit + "\n"
