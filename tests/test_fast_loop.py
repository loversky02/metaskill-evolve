"""Fast loop must actually self-improve a skill, offline, at $0."""
import json
import re

from mse.config import faithful
from mse.dag import DAG
from mse.fast_loop import run_fast
from mse.llm import MockLLM
from mse.skills import SkillStore
from mse.tasks import RuleTask, evaluate


class OracleLLM:
    """Competent-backbone stand-in for RuleTask: reads the missing rule from the
    prompt and returns the right structured output per agent. Deterministic, $0."""

    def complete(self, system, user, *, json_mode=False):
        m = re.search(r"rule_\d+", user)
        rule = m.group(0) if m else "rule_0"
        if json_mode:  # Analyzer
            return json.dumps({"tag": f"missing_{rule}", "analysis": f"skill lacks {rule}"})
        if "integer K" in user:  # Allocator
            return "2"
        return rule  # Proposer


def test_fast_loop_improves_utility(tmp_path):
    cfg = faithful()
    task = RuleTask(n_rules=4, n_examples=8)
    store = SkillStore.create(tmp_path)
    dag = DAG()

    st = run_fast(cfg, OracleLLM(), task, store, dag)

    # utility is monotonically non-decreasing and reaches the ceiling
    assert st.history[0] == 0.0
    assert all(b >= a - 1e-9 for a, b in zip(st.history, st.history[1:]))
    assert st.best_utility == 1.0
    # the adopted skill is what scores 1.0 on disk
    assert evaluate(task, store.read_task(), task.examples()) == 1.0
    # the DAG grew children beyond the root
    assert len(dag.best(100)) > cfg.fast_iterations


def test_fast_loop_runs_with_plain_mock(tmp_path):
    # A non-reasoning mock backbone must not crash the loop (robustness).
    cfg = faithful()
    task = RuleTask()
    store = SkillStore.create(tmp_path)
    dag = DAG()

    st = run_fast(cfg, MockLLM(), task, store, dag)

    assert 0.0 <= st.best_utility <= 1.0
    assert len(st.history) == cfg.fast_iterations + 1
