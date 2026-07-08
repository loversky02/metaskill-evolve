"""Procedure-bottleneck synthetic task (LetterCountTask)."""
from mse.synthetic import LetterCountTask


def test_counts_are_correct():
    task = LetterCountTask(words_per_item=6, n_items=5, target="r")
    for ex in task.examples():
        true = sum(w.count("r") for w in ex["words"])
        assert ex["answer"] == str(true)
        assert len(ex["words"]) == 6


def test_reward_exact_integer():
    task = LetterCountTask()
    ex = {"words": ["strawberry"], "question": "q", "answer": "3"}
    assert task.reward("the answer is 3", ex) == 1.0
    assert task.reward("I think it is 5", ex) == 0.0


def test_runs_through_evolution(tmp_path):
    from mse.config import Config
    from mse.dag import DAG
    from mse.evolve import run_evolution
    from mse.llm import MockLLM
    from mse.skills import SkillStore

    task = LetterCountTask(n_items=4)
    st = run_evolution(Config(fast_iterations=2), MockLLM(), task,
                       SkillStore.create(tmp_path), DAG(), two_level=True)
    assert len(st.history) == 3  # ran end-to-end
