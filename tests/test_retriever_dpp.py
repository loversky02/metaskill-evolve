"""Diversity-aware Retriever (σ) — the automem/robocurate DPP tie-in."""
import pytest

from mse.retriever_dpp import dpp_retrieve

np = pytest.importorskip("numpy")  # DPP path needs numpy; skip cleanly if absent


def test_dpp_prefers_diverse():
    pool = ["add rule_0", "add rule_0", "add rule_0", "totally different xyz"]
    picked = dpp_retrieve(pool, 2)
    assert len(picked) == 2
    assert "totally different xyz" in picked  # diversity beats the triplicate


def test_dpp_is_deterministic():
    pool = ["alpha one", "beta two", "gamma three", "alpha one"]
    assert dpp_retrieve(pool, 2) == dpp_retrieve(pool, 2)


def test_dpp_small_pool_returns_all():
    assert dpp_retrieve(["a"], 3) == ["a"]
    assert dpp_retrieve([], 3) == []
    assert dpp_retrieve(["a", "b"], 0) == []
