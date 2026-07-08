"""The recursion (slow loop) must add causal value over single-level, offline."""
import json
import re

from mse.config import faithful
from mse.dag import DAG
from mse.evolve import run_evolution
from mse.skills import SkillStore
from mse.tasks import RuleTask


class MetaSensitiveOracle:
    """Offline backbone whose Proposer behaviour depends on a MODE directive in
    its meta file. A weak seed (echo_variant) plateaus; the slow loop rewrites the
    Proposer meta to `target_missing`, which unlocks the task. This makes the
    recursion's value causal and observable at $0.
    """

    @staticmethod
    def _mode(system: str) -> str:
        modes = re.findall(r"MODE:\s*(\w+)", system)
        return modes[-1] if modes else "echo_variant"

    def complete(self, system, user, *, json_mode=False):
        rule_m = re.search(r"rule_\d+", user)
        rule = rule_m.group(0) if rule_m else "rule_0"
        if json_mode:  # Analyzer
            return json.dumps({"tag": f"missing_{rule}", "analysis": f"lacks {rule}"})
        if "integer K" in user:  # Allocator
            return "2"
        # Proposer: is this a meta-improvement edit, or a task edit?
        if "improvement pipeline" in user or "meta-productivity" in user:
            return "TARGET: pi\nMODE: target_missing"
        if self._mode(system) == "target_missing":
            return rule  # competent: add the actually-missing rule
        vm = re.search(r"variant (\d+) of", user)  # dumb: echo by variant index
        v = int(vm.group(1)) - 1 if vm else 0
        return f"rule_{v}"


def _run(two_level, root):
    cfg = faithful()
    task = RuleTask(n_rules=4, n_examples=8)
    store = SkillStore.create(root)
    store.write_meta("pi", "# Proposer (pi)\nMODE: echo_variant\n")  # weak seed
    dag = DAG()
    st = run_evolution(cfg, MetaSensitiveOracle(), task, store, dag,
                       two_level=two_level)
    return st, store, dag


def test_recursion_beats_single_level(tmp_path):
    two, store2, dag2 = _run(True, tmp_path / "two")
    one, _store1, _dag1 = _run(False, tmp_path / "one")

    # single-level plateaus: weak Proposer + K=2 can only reach rules 0,1
    assert one.history[-1] == 0.5
    # two-level: the slow loop upgrades the Proposer, so the task is fully solved
    assert two.history[-1] == 1.0
    assert two.history[-1] > one.history[-1]
    # recursion left evidence: a meta node in the DAG + a rewritten Proposer file
    assert len(dag2.best(100, kind="meta")) >= 1
    assert "target_missing" in store2.read_meta("pi")


def test_single_level_leaves_meta_untouched(tmp_path):
    _one, store1, dag1 = _run(False, tmp_path / "one")
    assert "target_missing" not in store1.read_meta("pi")  # meta frozen
    assert len(dag1.best(100, kind="meta")) == 0            # no slow-loop nodes
