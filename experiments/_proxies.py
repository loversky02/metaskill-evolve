"""Illustrative proxy backbones for offline experiment smoke runs.

Real experiments swap these for models from ``mse.config.BACKBONES`` (Gemma-4
E2B/E4B via MLX, GPT-5.5/Claude via 9router). The proxies emulate a strength
ladder on the RuleTask toy so the harnesses run end-to-end at $0 and reproduce the
*shape* of the headline result: meta-gain shrinks as the backbone strengthens.
"""
from __future__ import annotations

import json
import re

from mse.skills import SkillStore
from mse.tasks import RuleTask

WEAK_SEED_PI = "# Proposer (pi)\nMODE: echo_variant\n"


class _Proxy:
    """Shared Analyzer/Allocator/meta-edit behaviour; Proposer varies by strength."""

    @staticmethod
    def _missing(user: str) -> str:
        m = re.search(r"rule_\d+", user)
        return m.group(0) if m else "rule_0"

    @staticmethod
    def _mode(system: str) -> str:
        modes = re.findall(r"MODE:\s*(\w+)", system)
        return modes[-1] if modes else "echo_variant"

    def complete(self, system, user, *, json_mode=False):
        rule = self._missing(user)
        if json_mode:  # Analyzer
            return json.dumps({"tag": f"missing_{rule}", "analysis": f"lacks {rule}"})
        if "integer K" in user:  # Allocator
            return "2"
        if "improvement pipeline" in user or "meta-productivity" in user:  # meta edit
            return "TARGET: pi\nMODE: target_missing"
        return self._propose(system, user, rule)

    def _propose(self, system, user, rule):
        raise NotImplementedError


class WeakBackbone(_Proxy):
    """Obeys the (dumb) seed meta: echoes by variant until the slow loop fixes it."""

    def _propose(self, system, user, rule):
        if self._mode(system) == "target_missing":
            return rule
        m = re.search(r"variant (\d+) of", user)
        v = int(m.group(1)) - 1 if m else 0
        return f"rule_{v}"


class MidBackbone(_Proxy):
    """Competent except it can't crack the hardest rule without a meta upgrade."""

    def _propose(self, system, user, rule):
        if rule == "rule_3" and self._mode(system) != "target_missing":
            return "rule_0"
        return rule


class StrongBackbone(_Proxy):
    """Competent regardless of the meta instruction — recursion adds ~nothing."""

    def _propose(self, system, user, rule):
        return rule


STRENGTH_PROXIES = [("weak", WeakBackbone), ("mid", MidBackbone), ("strong", StrongBackbone)]


def fresh_store(root, weak_seed: bool = True) -> SkillStore:
    store = SkillStore.create(root)
    if weak_seed:
        store.write_meta("pi", WEAK_SEED_PI)
    return store


def rule_task(n_rules: int = 4, n_examples: int = 8) -> RuleTask:
    return RuleTask(n_rules=n_rules, n_examples=n_examples)
