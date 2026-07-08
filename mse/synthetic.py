"""Synthetic procedure-bottleneck tasks.

These sit in the narrow regime the live probe identified: the base policy is
neither at ceiling (trivial) nor floor (missing knowledge/tools) — it fails for
lack of a *systematic procedure* that a frozen backbone can discover and write
into the skill. That makes them the vehicle for a positive live recursion signal.

``LetterCountTask``: count total occurrences of a target letter across a word
list. LLMs are notoriously weak at this (tokenization), but a "spell each word,
count per word, then sum" procedure reliably helps. Answers are integers, so the
reward is exact and cheap.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .scoring import numeric_scorer

# r/e-heavy words so the target-letter counts are non-trivial and error-prone.
_WORDBANK = [
    "strawberry", "raspberry", "cranberry", "blueberry", "mulberry", "blackberry",
    "cherry", "apricot", "tangerine", "mandarin", "orange", "grapefruit",
    "pear", "peach", "raisin", "currant", "banana", "watermelon", "elderberry",
    "gooseberry",
]


@dataclass
class LetterCountTask:
    words_per_item: int = 6
    n_items: int = 12
    target: str = "r"
    desc: str = ("Count how many times the target letter appears in total across "
                 "the given words.")
    system_preamble: str = "You are a careful counter. Use the skill below.\n"

    def __post_init__(self):
        self._scorer = numeric_scorer(0.001)  # integers -> effectively exact

    def examples(self) -> List[dict]:
        n = len(_WORDBANK)
        out = []
        for i in range(self.n_items):
            start = (i * 3) % n
            words = [_WORDBANK[(start + j) % n] for j in range(self.words_per_item)]
            count = sum(w.count(self.target) for w in words)
            q = (f"How many times does the letter '{self.target}' appear in total "
                 f"across these words: {', '.join(words)}?")
            out.append({"id": i, "words": words, "question": q, "answer": str(count)})
        return out

    def describe(self, example: dict) -> str:
        return example["question"]

    def rollout(self, skill_text: str, example: dict, llm=None) -> str:
        if llm is None:
            return ""
        system = self.system_preamble + skill_text
        return llm.complete(system, example["question"] + "\nFinal answer (a number):")

    def reward(self, prediction, example) -> float:
        return self._scorer(prediction, example)
