"""Smoke tests for the four experiment harnesses — run end-to-end offline at $0."""
from experiments.e1_backbone_strength import format_table, smoke as e1
from experiments.e2_faithful_repro import smoke as e2
from experiments.e3_en_vi_transfer import smoke as e3
from experiments.e4_money_plot import smoke as e4


def test_e1_meta_gain_decreases_with_strength(tmp_path):
    rows = e1(tmp_path)
    gains = {r["backbone"]: r["meta_gain"] for r in rows}
    # the headline shape: recursion helps weak models, not strong ones
    assert gains["weak"] > gains["mid"] > gains["strong"]
    assert gains["strong"] == 0.0
    assert isinstance(format_table(rows), str)


def test_e2_two_level_at_least_single(tmp_path):
    r = e2(tmp_path)
    assert r["two_level"] >= r["single_level"]
    assert r["gain"] >= 0


def test_e3_transfer_helps(tmp_path):
    r = e3(tmp_path)
    assert r["transfer_helps"] is True
    assert r["transfer_final"] >= r["fresh_final"]


def test_e4_recursion_costs_more_tokens(tmp_path):
    rows = e4(tmp_path)
    single = next(r for r in rows if r["mode"] == "single_level")
    two = next(r for r in rows if r["mode"] == "two_level")
    assert two["llm_calls"] > single["llm_calls"]  # the slow loop adds calls
    assert two["approx_tokens"] > single["approx_tokens"]
