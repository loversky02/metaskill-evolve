# MetaSkill-Evolve — a $0 reproduction & dissection

An open, Apple-Silicon-friendly reproduction of **MetaSkill-Evolve: Recursive
Self-Improvement of LLM Agents via Two-Timescale Meta-Skill Evolution**
([arXiv:2607.05297](https://arxiv.org/abs/2607.05297)), plus an honest look at
*when* the recursion actually pays off.

## What the paper does

Self-improving agents rewrite their own **task skill** (a Markdown file — *what
the agent does*) from execution traces. MetaSkill-Evolve makes this **recursive**:
every branch also carries a **meta-skill** `m = (ψ, σ, α, π, ε)` — five Markdown
files that parameterise the five agents of the improvement pipeline:

| symbol | agent | role |
| --- | --- | --- |
| ψ | Analyzer | map a failure to a tag + free-form analysis |
| σ | Retriever | pull same-/cross-branch inspirations for that tag |
| α | Allocator | set the child budget `K ∈ [1, K_max]` |
| π | Proposer | emit a concrete edit to the task skill |
| ε | Evolver | apply the edit and verify the result |

Task skills evolve on a **fast loop**; the meta-skill evolves on a **slow loop**
(`every H iterations`) under the *same* pipeline applied to itself — no extra
model, no extra objective. All five agents share one **frozen** backbone.

## The honest angle (why this repo exists)

The paper's own Table 1 shows most of the win comes from the **fast loop**, and
the recursive meta-skill adds a **modest, task-dependent** bonus:

| benchmark | no-skill | static | single-level (fast) | + meta (recursion) |
| --- | --- | --- | --- | --- |
| OfficeQA | 31.78 | 36.09 | **48.94** (+17.16) | 55.32 (**+6.38**) |
| SealQA | 29.17 | 29.41 | **37.21** (+8.04) | 45.26 (**+8.05**) |
| ALFWorld | 92.31 | 90.38 | **92.31** (+0.00) | 94.23 (**+1.92**) |

Recursion helps most on retrieval-heavy QA and is within noise on a
near-ceiling embodied benchmark (ALFWorld starts at 92%; *static* even hurts it).

**Headline experiment (not in the paper): does the meta-loop's value shrink as
the backbone gets stronger?** The paper uses Gemma-4 31B. We sweep a strength
ladder — Gemma-4 E2B/E4B locally (weak) → GPT-5.5 / Claude via an
OpenAI-compatible gateway (strong) — and measure the marginal accuracy of the
slow loop. Hypothesis: a strong backbone already "knows how to analyze/propose,"
so the evolved meta-skill is scaffolding it doesn't need — making recursive
self-improvement mainly worth it for **cheap/weak** models.

Secondary angles: **EN→VI transfer** (does a meta-skill evolved on English tasks
speed up evolution on Vietnamese ones?) and a **cost money-plot** (recursion
multiplies inference — is the token spend worth the gain?).

## Status — engine complete, offline & $0 (33 tests green)

All five milestones are built and verified on the **mock/proxy backbone** — no API
key, no GPU, no downloads:

- **M1 — engine core** ✅ skill/meta-skill store + snapshot/restore, SQLite branch
  DAG + weighted frontier, provider-agnostic LLM client, paper-faithful config.
- **M2 — five agents + fast loop (Alg. 1)** ✅ toy self-improves U 0→1.0.
- **M3 — slow loop (Alg. 2), the recursion** ✅ the pipeline rewrites its own
  `(ψ,σ,α,π,ε)`; two-level beats single-level on a meta-sensitive toy (1.0 vs 0.5).
- **M4 — benchmark adapters** ✅ `QATask` (SealQA/OfficeQA/GSM8K/vi-gsm8k) +
  `ALFWorldTask` (+ `MockAlfEnv`) + scorers; live loaders lazy-guard `datasets`.
- **M5 — four experiment harnesses** ✅ E1 backbone-strength, E2 faithful repro,
  E3 EN→VI transfer, E4 money-plot — each with an offline `smoke()`.

### Offline smoke result — the headline shape (toy proxy, mechanism check)

`experiments/e1_backbone_strength.py` on the RuleTask toy with weak/mid/strong
proxy backbones:

| backbone | single-level | two-level | meta-gain |
|---|---|---|---|
| weak | 0.50 | 1.00 | **+0.50** |
| mid | 0.75 | 1.00 | **+0.25** |
| strong | 1.00 | 1.00 | **+0.00** |

Recursion's value shrinks to zero as the backbone strengthens — the predicted law.
**These are toy numbers that verify the mechanism, not benchmark results;** real
numbers come from the live runs below.

## Run

```bash
pip install -r requirements.txt
python -c "import pytest,sys; sys.exit(pytest.main(['-q']))"   # 33 tests, offline, $0

# offline experiment smoke runs (toy proxy backbones):
python -m experiments.e1_backbone_strength
python -m experiments.e4_money_plot
```

### Live runs (real numbers)

Swap the proxy for a real backbone and point the tasks at real data — no code
changes beyond the backbone spec:

- **weak / local (24 GB Mac):** `mse.config.backbone("gemma-4-e4b")` → MLX.
- **strong:** set `OPENAI_BASE_URL` to a 9router / OpenAI-compatible endpoint and
  use `backbone("gpt-5.5")` / `backbone("claude")`.
- **benchmarks:** `mse.benchmarks.load_sealqa(...)`,
  `load_gsm8k(..., vietnamese=True)`, or `ALFWorldTask` with a real env. OfficeQA
  needs a one-time HuggingFace access request.

The paper's absolute numbers used a 31B backbone that does not fit 24 GB locally;
we reproduce the method and the qualitative findings.
