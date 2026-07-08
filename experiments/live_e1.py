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
    {"question": "A store sells pens at 3 for $2. A customer buys 12 pens and pays "
                 "with a $10 bill. How much change do they get, in dollars?", "answer": "2"},
    {"question": "A car goes 90 km in 1.5 h, then 60 km in 1 h. What is its average "
                 "speed over the whole trip, in km/h?", "answer": "60"},
    {"question": "After a 20% increase, a price becomes $60. It is then reduced by "
                 "$6. What is the final price, in dollars?", "answer": "54"},
    {"question": "A tank leaks 2 L every 15 minutes. Starting from 50 L, how many "
                 "liters remain after 2 hours?", "answer": "34"},
    {"question": "A rectangle is 8 by 5 cm. Both sides are doubled, then 20 cm^2 is "
                 "cut off. What is the remaining area, in cm^2?", "answer": "140"},
    {"question": "3 workers paint 3 fences in 3 hours. At that rate, how many fences "
                 "do 6 workers paint in 6 hours?", "answer": "12"},
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
