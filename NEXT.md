# NEXT — MetaSkill-Evolve resume pointer

Resume phrase: **"tiếp tục metaskill-evolve"**

## Where we are
- **M1 (engine core) DONE, offline $0.** `mse/`: `config.py` (paper hyperparams +
  backbone registry), `llm.py` (mock/openai/mlx), `skills.py` (skill + meta-skill
  files, snapshot/restore), `dag.py` (SQLite branch DAG + frontier).
- **M2 (five agents + fast loop, Alg. 1) DONE.** `mse/agents.py` (ψ/σ/α/π/ε),
  `mse/tasks.py` (RuleTask), `mse/fast_loop.py`. Toy self-improves U 0→1.0.
- **M3 (slow loop, Alg. 2 — recursion) DONE.** `mse/slow_loop.py` + `mse/evolve.py`
  (`run_evolution`, `two_level` toggle). Two-level 1.0 vs single-level 0.5 on the
  meta-sensitive toy; meta node + rewritten Proposer as evidence.
- **M4 (benchmark adapters) DONE.** `mse/benchmarks.py` (`QATask`, `ALFWorldTask`,
  `MockAlfEnv`, `load_sealqa`/`load_gsm8k`) + `mse/scoring.py` (4 scorers).
- **M5 (four experiment harnesses) DONE.** `experiments/e1..e4` each with an
  offline `smoke()`. **36 tests pass, $0.** Smoke E1 meta-gain +0.50/+0.25/+0.00
  (weak/mid/strong) — the headline law; E3 EN→VI transfer helps; E4 two-level
  +30% tokens for its extra gain. (Toy proxy numbers, mechanism check only.)
- **SHIPPED + paper + TIE-IN 1.** Public: github.com/loversky02/metaskill-evolve
  (main). `paper/main.tex` compiles (pdflatex → main.pdf). σ Retriever now uses a
  vendored greedy-DPP (`mse/retriever_dpp.py`, numpy-optional + recency fallback) —
  the automem/robocurate tie-in.
- **TIE-IN 2 done+pushed** (super-agent axis-4 EVO {none,fast_only,two_level}).
- **LIVE probe = both failure modes** (2026-07-08): word problems → CEILING
  (haiku/opus/gemma-2-2b all U0=1.0, meta-gain 0); SealQA Seal-Hard bare-LLM → FLOOR
  (haiku 0/12, no knowledge/tools). Recursion only helps the narrow middle
  (procedure-bottleneck). Paper §4.2 + tab:law-live report both. MLXClient works
  (chat-template + retry); a cheap U0-probe-before-big-run saved two null runs.
  **Left (optional, bigger builds): a search-augmented agent OR a moderate
  procedure-bottlenecked task for a positive live signal; E3 vi-gsm8k; E4 fig.**
  Offline mechanism study stays THE clean demo; live conclusion = honest
  narrow-regime characterisation.

## Paper facts locked (arXiv:2607.05297)
- Backbone: **Gemma-4 31B, frozen**, shared by all 5 agents.
- Schedule: 5 fast iters, **H=2** (2 meta-updates), K_init=2, K_max=3, K_F=3,
  η=(1.0,0.5,0.25), p_cross=0.2, L_same=3, L_cross=2.
- Meta-skill `m=(ψ,σ,α,π,ε)` → Analyzer/Retriever/Allocator/Proposer/Evolver.
- Utility `U(s)=E[r(A_s(x),y)]`; meta-productivity `P(m|s)=E[(1/K)Σ(U(s'_k)-U(s))]`.
- Results (held-out): OfficeQA 31.78→55.32, SealQA 29.17→45.26, ALFWorld
  92.31→94.23. **Two-level (recursion) gain is the small part**: +6.38/+8.05/+1.92.
- Component ablation: **Allocator (α) + Proposer (π) load-bearing**; Retriever
  (σ, cross-branch) least impactful → DPP-as-Retriever tie-in is real but minor.

## Hardware / routing reality
- Mac **24 GB RAM** → cannot run 31B locally (~20 GB 4-bit is too tight).
- Strength ladder for the headline experiment: **Gemma-4 E2B/E4B (local MLX,
  weak)** → **GPT-5.5 / Claude via 9router (strong)**. 31B is a cited midpoint we
  can't run locally → faithful repro = method + qualitative, not exact numbers.
- Confirm at M5: exact MLX repo ids for Gemma-4 E2B/E4B and whether they need
  `mlx_vlm` (multimodal) vs `mlx_lm` to load.

## Benchmarks (from research)
- **ALFWorld** PUBLIC — `pip install alfworld`, text-only on Mac CPU, no GPU.
- **SealQA** PUBLIC — HF `vtllms/sealqa` (seal_0=111, seal_hard=254, longseal=254),
  free-form + LLM-judge, offline eval OK (LongSeal ships doc contexts).
- **OfficeQA** PARTIAL — code open, data gated on HF `databricks/officeqa` (one
  access request); numeric fuzzy-match scoring.

## Next: LIVE runs (real numbers) — needs the user's machine / credentials
Engine is done and offline-verified. What remains is money, downloads, and time —
run on the 24 GB Mac + 9router:
1. **Backbones:** confirm exact MLX repo ids for Gemma-4 E2B/E4B (may need
   `mlx_vlm` not `mlx_lm`); set `OPENAI_BASE_URL` for 9router (GPT-5.5/Claude).
2. **E1 (headline) for real:** run `experiments.e1_backbone_strength.run_e1` with
   `[(name, make_llm(backbone(k))) for k in STRENGTH_ORDER]` on a real benchmark.
   Confirm meta-gain shrinks as backbone strengthens.
3. **E2 faithful repro:** SealQA (`load_sealqa`, offline-capable) + ALFWorld
   (`pip install alfworld`, text-only). OfficeQA needs a HF access request.
4. **E3 EN→VI:** `load_gsm8k` (EN) + `load_gsm8k(vietnamese=True)` (vi-gsm8k).
5. **E4 money-plot:** already wired via `CountingLLM`; add a matplotlib figure.
6. **Tie-ins:** swap σ (Retriever) for the DPP selector from `automem-vn`/
   `robot-lfd`; expose skill-evolution as the 4th `FactoredPolicy` axis in
   `super-agent`. (User: "thêm vào cả 3".)
7. **Paper:** write up honest thesis + E1 law + E3/E4; ship repo public first.

## Conventions
- Public artifacts in **English**. **No** `Co-Authored-By: Claude` trailer in
  commits (academic/paper repo).
- RTK breaks bare `pytest`; run via
  `python -c "import pytest,sys; sys.exit(pytest.main(['-q']))"`.
