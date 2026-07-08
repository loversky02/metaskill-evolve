"""Provider-agnostic LLM client.

All five pipeline agents call one frozen backbone, so the interface is tiny and
the backbone-strength ablation is a one-line swap:

* ``MockLLM``            - deterministic, offline, $0 (tests and dry runs)
* ``OpenAICompatClient`` - any OpenAI-compatible endpoint (9router -> GPT-5.5/Claude)
* ``MLXClient``          - a local Apple-Silicon model (the weak-backbone end)
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional, Protocol


class LLM(Protocol):
    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str: ...


class MockLLM:
    """Deterministic offline backbone (no network, $0).

    Output is a pure function of (system, user) so runs are reproducible. Seed
    ``scripted`` with ``substring -> reply`` to drive specific agent behaviour in
    tests; the first matching needle (checked against user then system) wins.
    """

    def __init__(self, scripted: Optional[Dict[str, str]] = None, default: str = ""):
        self.scripted = scripted or {}
        self.default = default
        self.calls: List[Dict[str, str]] = []

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        self.calls.append({"system": system, "user": user})
        for needle, reply in self.scripted.items():
            if needle in user or needle in system:
                return reply
        if self.default:
            return self.default
        h = hashlib.sha256((system + "\x00" + user).encode()).hexdigest()[:8]
        return json.dumps({"stub": h}) if json_mode else f"[mock:{h}]"


class OpenAICompatClient:
    """Targets any OpenAI-compatible /chat/completions endpoint (9router, etc.)."""

    def __init__(self, model: str, base_url: Optional[str] = None,
                 api_key: Optional[str] = None, temperature: float = 0.7,
                 max_tokens: int = 1024):
        from openai import OpenAI  # lazy: keep the dep optional

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = OpenAI(
            base_url=(base_url or os.environ.get("OPENAI_BASE_URL")
                      or os.environ.get("NINEROUTER_BASE_URL")),
            api_key=(api_key or os.environ.get("OPENAI_API_KEY")
                     or os.environ.get("NINEROUTER_API_KEY") or "sk-none"),
        )

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        # Router quirks: cc/* (Claude) reject `temperature`; gpt-5.5 only accepts
        # temperature=1. We skip response_format for compatibility and rely on the
        # tolerant JSON parsing in agents._parse_json.
        kwargs = {"max_tokens": self.max_tokens}
        m = self.model
        if m.startswith("cc/"):
            pass  # Claude via the router rejects a temperature argument
        elif m.startswith("gpt-5") or m.startswith("cx/"):
            kwargs["temperature"] = 1.0
        else:
            kwargs["temperature"] = self.temperature
        resp = self._client.chat.completions.create(
            model=m,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )
        return resp.choices[0].message.content or ""


class MLXClient:
    """Local Apple-Silicon backbone via mlx-lm (best-effort; validated at M5)."""

    def __init__(self, model: str, temperature: float = 0.7, max_tokens: int = 1024):
        from mlx_lm import generate, load  # lazy

        self._generate = generate
        self._model, self._tokenizer = load(model)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        prompt = f"<system>\n{system}\n</system>\n<user>\n{user}\n</user>\n"
        return self._generate(
            self._model, self._tokenizer, prompt=prompt, max_tokens=self.max_tokens
        )


def make_llm(backbone) -> LLM:
    """Build an LLM from a ``config.BackboneSpec``."""
    p = backbone.provider
    if p == "mock":
        return MockLLM()
    if p == "openai":
        return OpenAICompatClient(
            backbone.name, base_url=backbone.base_url,
            temperature=backbone.temperature, max_tokens=backbone.max_tokens,
        )
    if p == "mlx":
        return MLXClient(
            backbone.name, temperature=backbone.temperature,
            max_tokens=backbone.max_tokens,
        )
    raise ValueError(f"unknown provider: {p!r}")


class CountingLLM:
    """Wraps any backbone to count calls and (approx) tokens — for the money-plot.

    Token estimate is chars/4; good enough to compare fast-only vs recursive cost.
    """

    def __init__(self, inner: LLM):
        self.inner = inner
        self.calls = 0
        self.tokens = 0

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        self.calls += 1
        self.tokens += (len(system) + len(user)) // 4
        out = self.inner.complete(system, user, json_mode=json_mode)
        self.tokens += len(out) // 4
        return out
