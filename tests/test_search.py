"""Search-augmented QA task (Project 1) — offline with a fake search tool."""
from mse.scoring import contains_scorer
from mse.search import SearchQATask


class FakeTool:
    def __init__(self, snippets):
        self.snippets = snippets

    def search(self, query):
        return self.snippets


class _CtxOracle:
    """Answers from the provided context (a competent reader)."""

    def complete(self, system, user, *, json_mode=False):
        return "Based on the results, the answer is Serban Ghenea."


def test_search_qa_rollout_uses_context():
    tool = FakeTool(["Grammy AOTY: Serban Ghenea holds the record as engineer/mixer."])
    task = SearchQATask([{"question": "who holds the AOTY record?", "answer": "Serban Ghenea"}],
                        tool, contains_scorer())
    ex = task.examples()[0]
    pred = task.rollout("", ex, _CtxOracle())
    assert task.reward(pred, ex) == 1.0


def test_search_qa_runs_through_evolution(tmp_path):
    from mse.config import Config
    from mse.dag import DAG
    from mse.evolve import run_evolution
    from mse.llm import MockLLM
    from mse.skills import SkillStore

    tool = FakeTool(["snippet a", "snippet b"])
    task = SearchQATask([{"question": "q1", "answer": "a"}, {"question": "q2", "answer": "b"}],
                        tool, contains_scorer())
    st = run_evolution(Config(fast_iterations=2), MockLLM(), task,
                       SkillStore.create(tmp_path), DAG(), two_level=True)
    assert len(st.history) == 3  # end-to-end, no crash
