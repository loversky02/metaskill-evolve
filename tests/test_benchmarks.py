"""Benchmark adapters: scorers + QA rollout + ALFWorld, all offline at $0."""
from mse.benchmarks import ALFWorldTask, MockAlfEnv, QATask
from mse.scoring import (contains_scorer, exact_scorer, judge_scorer,
                         numeric_scorer)


# --- scorers --------------------------------------------------------------
def test_exact_scorer():
    s = exact_scorer()
    assert s("Hello ", {"answer": "hello"}) == 1.0
    assert s("world", {"answer": "hello"}) == 0.0


def test_numeric_scorer_tolerance():
    s = numeric_scorer(0.01)
    assert s("the answer is 100.5", {"answer": "100"}) == 1.0   # within 1%
    assert s("about 102", {"answer": "100"}) == 0.0             # outside 1%
    assert s("no number", {"answer": "5"}) == 0.0


def test_contains_scorer():
    s = contains_scorer()
    assert s("The capital is Paris.", {"answer": "paris"}) == 1.0
    assert s("I don't know", {"answer": "paris"}) == 0.0


def test_judge_scorer():
    class Judge:
        def __init__(self, verdict):
            self.verdict = verdict

        def complete(self, system, user, *, json_mode=False):
            return self.verdict

    ex = {"question": "q", "answer": "a"}
    assert judge_scorer(Judge("YES"))("pred", ex) == 1.0
    assert judge_scorer(Judge("NO, wrong"))("pred", ex) == 0.0


# --- QATask ---------------------------------------------------------------
class _NumOracle:
    """Answers correctly only when the skill contains the hint 'add'."""

    def complete(self, system, user, *, json_mode=False):
        return "The answer is 4" if "add" in system else "The answer is 0"


def test_qatask_rollout_and_reward():
    task = QATask([{"question": "2+2", "answer": "4"}], numeric_scorer())
    good = task.rollout("Rule: add the numbers", task.examples()[0], _NumOracle())
    assert task.reward(good, task.examples()[0]) == 1.0
    bad = task.rollout("", task.examples()[0], _NumOracle())
    assert task.reward(bad, task.examples()[0]) == 0.0


def test_qatask_runs_through_evolution(tmp_path):
    # regression: fast_iteration must not assume the RuleTask example schema
    # (it used wc["text"]; QATask examples have "question"/"answer").
    from mse.config import Config
    from mse.dag import DAG
    from mse.evolve import run_evolution
    from mse.skills import SkillStore

    task = QATask([{"question": "2+2", "answer": "4"},
                   {"question": "3+5", "answer": "8"}], numeric_scorer())
    st = run_evolution(Config(fast_iterations=2), _NumOracle(), task,
                       SkillStore.create(tmp_path), DAG(), two_level=True)
    assert len(st.history) == 3  # ran end-to-end, no KeyError


# --- ALFWorld -------------------------------------------------------------
class _AlfOracle:
    def complete(self, system, user, *, json_mode=False):
        return "take apple" if "take apple" in system else "look around"


def test_alfworld_mock_env():
    task = ALFWorldTask(lambda ex: MockAlfEnv("take apple"), n=2, max_steps=5)
    ex = task.examples()[0]
    assert task.reward(task.rollout("Always: take apple", ex, _AlfOracle()), ex) == 1.0
    assert task.reward(task.rollout("", ex, _AlfOracle()), ex) == 0.0
