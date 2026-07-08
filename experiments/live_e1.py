"""E1 (live) — meta-gain vs real backbone strength via an OpenAI-compatible gateway.

Set ``OPENAI_BASE_URL`` + ``OPENAI_API_KEY`` (a 9router-style gateway), then:

    python -m experiments.live_e1 --backbones claude-haiku claude-opus

No MLX download needed: the Claude family (haiku < sonnet < opus) spans the
strength ladder through the gateway. Whether the meta-gain actually shrinks as the
backbone strengthens is the empirical question — the result may be null/noisy on a
small task, which is a legitimate (honest) finding.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mse.benchmarks import QATask
from mse.config import Config, backbone
from mse.dag import DAG
from mse.evolve import run_evolution
from mse.llm import make_llm
from mse.scoring import numeric_scorer
from mse.skills import SkillStore

# Small self-contained word-problem set (numeric answers) so the runner needs no
# dataset download. A good procedural skill (evolved by the pipeline) should help a
# weaker backbone more than a strong one.
DEFAULT_TASK = [
    {"question": "A shelf holds 3 boxes; each box has 7 books. How many books total?", "answer": "21"},
    {"question": "Sarah had 20 apples, gave away 8, then bought 5 more. How many now?", "answer": "17"},
    {"question": "A train goes 60 km/h for 2.5 hours. How far (km)?", "answer": "150"},
    {"question": "A shirt costs 24 and is 25% off. What is the sale price?", "answer": "18"},
    {"question": "5 teams of 4 players each; 3 players are absent. How many present?", "answer": "17"},
    {"question": "A tank holds 200 L, is 40% full, then 30 L is added. How many L now?", "answer": "110"},
]


def run_live_e1(backbone_keys, task=None, cfg=None, root="runs/live_e1"):
    cfg = cfg or Config(fast_iterations=3)
    task = task or QATask(DEFAULT_TASK, numeric_scorer())
    root = Path(root)
    rows = []
    for key in backbone_keys:
        llm = make_llm(backbone(key))
        one = run_evolution(cfg, llm, task,
                            SkillStore.create(root / f"{key}_one"), DAG(), two_level=False)
        two = run_evolution(cfg, llm, task,
                            SkillStore.create(root / f"{key}_two"), DAG(), two_level=True)
        gain = round(two.history[-1] - one.history[-1], 3)
        rows.append({"backbone": key, "single": one.history[-1],
                     "two_level": two.history[-1], "meta_gain": gain})
        print(f"{key:16s} single={one.history[-1]:.2f} "
              f"two_level={two.history[-1]:.2f} gain={gain:+.2f}", flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=["claude-haiku", "claude-opus"])
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--out", default="runs/live_e1/result.json")
    a = ap.parse_args()
    rows = run_live_e1(a.backbones, cfg=Config(fast_iterations=a.iters))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=2))
    print("saved", a.out)
