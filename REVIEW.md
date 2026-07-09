# Empirical review — MetaSkill-Evolve — two-timescale recursive skill evolution

arXiv [2607.05297](https://arxiv.org/abs/2607.05297) · paper-forge empirical-review lane (heuristic)

> Unlike a static manuscript review (e.g. PAT), this verdict comes from *executed* evidence: the reproduction ran, and each central claim is checked against the numbers it produced — including whether the claim's own risk regime was ever exercised.

**Hypothesis under test:** The fast loop is the workhorse; recursion helps only in a narrow procedure-bottleneck regime. On near-saturated tasks (ALFWorld ~92%) and strong backbones the meta-gain vanishes (+0.00). These are EN benchmarks at this eval size — whether the recursion gain transfers to Vietnamese tasks or holds at larger eval scale is untested.

**Summary:** 3 claim(s) — 🟢 2 hold · 🔴 0 break · 🟡 0 mixed · ⚪ 1 inconclusive

## Claims

### 🟢 HOLDS — Recursive (two-level) skill evolution beats the single-level fast loop on a reasoning-heavy benchmark.
**Measured:** Δ +6.38 (12%) · regime *OfficeQA, single-level fast loop vs two-level recursion* · confidence **medium** · ⚠ risk regime untested

- **Why:** OfficeQA accuracy, + meta (recursion) vs single-level moved 48.94→55.32 (12%) in the claimed 'higher' direction. But the hypothesis names a failure regime this evidence never touched, so the claim is confirmed only in the tested regime.
- **Honest finding:** Reproduced: OfficeQA accuracy, + meta (recursion) vs single-level moved 48.94→55.32 (12%) in the claimed direction under 'OfficeQA, single-level fast loop vs two-level recursion'. But the failure regime named in the hypothesis was never exercised, so this holds only in the tested regime — treat it as regime-local, not a general confirmation.

### 🟢 HOLDS — The recursion gain also holds on a search-bottlenecked benchmark.
**Measured:** Δ +8.05 (18%) · regime *SealQA, single-level fast loop vs two-level recursion* · confidence **medium** · ⚠ risk regime untested

- **Why:** SealQA accuracy, + meta (recursion) vs single-level moved 37.21→45.26 (18%) in the claimed 'higher' direction. But the hypothesis names a failure regime this evidence never touched, so the claim is confirmed only in the tested regime.
- **Honest finding:** Reproduced: SealQA accuracy, + meta (recursion) vs single-level moved 37.21→45.26 (18%) in the claimed direction under 'SealQA, single-level fast loop vs two-level recursion'. But the failure regime named in the hypothesis was never exercised, so this holds only in the tested regime — treat it as regime-local, not a general confirmation.

### ⚪ INCONCLUSIVE — The recursion gain generalizes to embodied, near-saturated tasks.
**Measured:** Δ +1.92 (2%) · regime *ALFWorld, single-level fast loop vs two-level recursion* · confidence **low**

- **Why:** ALFWorld success rate, + meta (recursion) vs single-level moved only 2.0% (92.31→94.23) — within noise.
- **Honest finding:** Inconclusive: the evidence for ALFWorld success rate, + meta (recursion) vs single-level (92.31→94.23) is too flat or too thin to confirm or refute — a near-flat number is not a finding.
