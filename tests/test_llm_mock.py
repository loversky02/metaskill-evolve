import json

from mse.config import BackboneSpec
from mse.llm import MockLLM, make_llm


def test_mock_is_deterministic():
    m = MockLLM()
    a = m.complete("sys", "user")
    b = m.complete("sys", "user")
    assert a == b
    assert a.startswith("[mock:")


def test_mock_json_mode_valid():
    m = MockLLM()
    parsed = json.loads(m.complete("s", "u", json_mode=True))
    assert "stub" in parsed


def test_mock_scripted():
    m = MockLLM(scripted={"FAIL": "tag: parse_error"})
    assert m.complete("", "the FAIL happened") == "tag: parse_error"
    assert m.complete("", "nothing here").startswith("[mock:")


def test_mock_records_calls():
    m = MockLLM()
    m.complete("s", "u1")
    m.complete("s", "u2")
    assert len(m.calls) == 2
    assert m.calls[0]["user"] == "u1"


def test_make_llm_mock():
    llm = make_llm(BackboneSpec(provider="mock"))
    assert isinstance(llm, MockLLM)
