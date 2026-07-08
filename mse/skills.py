"""On-disk skill and meta-skill files.

A *branch* owns a task skill (``skill.md`` — what the agent does) and a
meta-skill ``m = (psi, sigma, alpha, pi, epsilon)`` — five Markdown files that
parameterise the Analyzer, Retriever, Allocator, Proposer, and Evolver of the
improvement pipeline (arXiv:2607.05297, Sec. 2). The slow loop rewrites these
five files with the *same* pipeline, which is what makes the system recursive.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

# meta-skill symbol -> (filename, the pipeline agent it parameterises)
META_COMPONENTS: Dict[str, Tuple[str, str]] = {
    "psi": ("analyzer.md", "Analyzer"),
    "sigma": ("retriever.md", "Retriever"),
    "alpha": ("allocator.md", "Allocator"),
    "pi": ("proposer.md", "Proposer"),
    "epsilon": ("evolver.md", "Evolver"),
}

# Human-authored seed prompts — the "authored once, held fixed" starting point
# that the slow loop is allowed to rewrite.
SEED_META: Dict[str, str] = {
    "psi": (
        "# Analyzer (psi)\n\n"
        "Read the task and the failed trajectory. Output a short failure TAG "
        "(a few words, snake_case) and a free-form ANALYSIS of the root cause.\n"
    ),
    "sigma": (
        "# Retriever (sigma)\n\n"
        "Given the failure tag, select the most relevant prior skill edits as "
        "inspirations: prefer same-branch edits, occasionally pull cross-branch "
        "ones that addressed a similar tag.\n"
    ),
    "alpha": (
        "# Allocator (alpha)\n\n"
        "Given the analysis and remaining budget, choose how many child variants "
        "K in [1, K_max] to propose this step. Spend more where the analysis is "
        "uncertain, fewer where the fix is obvious.\n"
    ),
    "pi": (
        "# Proposer (pi)\n\n"
        "Given the worst case, the analysis, and the retrieved inspirations, emit "
        "one concrete EDIT to the task skill. When asked for variant k, take a "
        "distinct intervention angle from the other variants.\n"
    ),
    "epsilon": (
        "# Evolver (epsilon)\n\n"
        "Apply the proposed edit to the task skill file and verify the result is "
        "well-formed. Reject edits that corrupt the skill or leave it empty.\n"
    ),
}

SEED_TASK = (
    "# Task Skill\n\n"
    "No specialised procedure yet. Solve the task directly, show your reasoning, "
    "and state a final answer clearly.\n"
)

Snapshot = Dict[str, str]  # relative path -> file contents


@dataclass
class SkillStore:
    """Reads/writes one branch's skill + meta-skill files, and snapshots them."""

    root: Path

    @classmethod
    def create(cls, root, *, seed: bool = True) -> "SkillStore":
        root = Path(root)
        (root / "meta").mkdir(parents=True, exist_ok=True)
        store = cls(root)
        if seed:
            store.write_task(SEED_TASK)
            for sym, text in SEED_META.items():
                store.write_meta(sym, text)
        return store

    # --- task skill ---------------------------------------------------
    @property
    def task_path(self) -> Path:
        return self.root / "skill.md"

    def read_task(self) -> str:
        return self.task_path.read_text() if self.task_path.exists() else ""

    def write_task(self, text: str) -> None:
        self.task_path.parent.mkdir(parents=True, exist_ok=True)
        self.task_path.write_text(text)

    # --- meta skill ---------------------------------------------------
    def meta_path(self, symbol: str) -> Path:
        if symbol not in META_COMPONENTS:
            raise KeyError(symbol)
        return self.root / "meta" / META_COMPONENTS[symbol][0]

    def read_meta(self, symbol: str) -> str:
        p = self.meta_path(symbol)
        return p.read_text() if p.exists() else ""

    def write_meta(self, symbol: str, text: str) -> None:
        p = self.meta_path(symbol)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    def meta(self) -> Dict[str, str]:
        return {s: self.read_meta(s) for s in META_COMPONENTS}

    # --- snapshot / restore (Alg. 1: "restore branch snapshots to disk") --
    def snapshot(self) -> Snapshot:
        snap = {"skill.md": self.read_task()}
        for sym, (fname, _) in META_COMPONENTS.items():
            snap[f"meta/{fname}"] = self.read_meta(sym)
        return snap

    def restore(self, snap: Snapshot) -> None:
        for rel, text in snap.items():
            p = self.root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
