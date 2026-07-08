"""Web-search tool + a search-augmented QA task (Project 1).

Reproduces the paper's tool-using setup so a knowledge benchmark like SealQA
becomes *procedure*-bottlenecked (reason over noisy/conflicting results) instead
of floor (bare LLM lacks the knowledge). $0 and keyless: primary = ``ddgs``
(DuckDuckGo, multi-backend), fallback = Wikipedia REST. The evolved skill guides
HOW to read and reconcile the results.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List


class SearchTool:
    """Keyless web search with backoff and a Wikipedia fallback."""

    def __init__(self, k: int = 5, pause: float = 1.5, retries: int = 3,
                 user_agent: str = "metaskill-evolve/0.1 (research)"):
        self.k = k
        self.pause = pause
        self.retries = retries
        self.user_agent = user_agent

    def search(self, query: str) -> List[str]:
        return self._ddgs(query) or self._wiki(query)

    def _ddgs(self, query: str) -> List[str]:
        try:
            from ddgs import DDGS
        except Exception:
            return []
        for attempt in range(self.retries):
            try:
                res = DDGS().text(query, max_results=self.k)
                out = [f"{r.get('title', '')}: {r.get('body') or r.get('snippet') or ''}"
                       .strip(": ").strip() for r in res]
                if out:
                    return out
            except Exception:
                time.sleep(self.pause * (attempt + 1))  # throttle backoff
        return []

    def _wiki(self, query: str) -> List[str]:
        try:
            import re

            import requests
        except Exception:
            return []
        try:
            r = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={"action": "query", "list": "search", "srsearch": query,
                        "format": "json", "srlimit": self.k},
                headers={"User-Agent": self.user_agent}, timeout=15,
            )
            hits = r.json().get("query", {}).get("search", [])
            return [f"{h['title']}: {re.sub('<[^>]+>', '', h.get('snippet', ''))}" for h in hits]
        except Exception:
            return []


@dataclass
class SearchQATask:
    """Answer a fact-seeking question using retrieved (noisy) context + the skill."""

    items: List[dict]
    tool: object
    scorer: object
    desc: str = ("Answer the fact-seeking question using the search results, which "
                 "may be noisy, conflicting, or contain traps.")

    def examples(self) -> List[dict]:
        return self.items

    def describe(self, example: dict) -> str:
        return example["question"]

    def rollout(self, skill_text: str, example: dict, llm=None) -> str:
        if llm is None:
            return ""
        snippets = self.tool.search(example["question"])[:5]
        ctx = "\n".join(f"- {s}" for s in snippets) or "(no results)"
        system = ("You answer fact-seeking questions using web search results that "
                  "may be noisy or conflicting. Use the skill below.\n" + skill_text)
        return llm.complete(system, f"Question: {example['question']}\n"
                            f"Search results:\n{ctx}\nFinal answer:")

    def reward(self, prediction, example) -> float:
        return self.scorer(prediction, example)
