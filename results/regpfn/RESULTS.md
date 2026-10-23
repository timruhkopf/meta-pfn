# Phase 1 (regpfn), round 1: references and the back half with oracle / perturbed seeds — RESULTS

Spec: `docs/spec_phase1_experiments.md`. Notebook: `notebooks/04_2026-10-23_regpfn.ipynb` (description = pre-registration,
Observations, Verdict). Issues #8 (architecture, branch `feat/8-regpfn-arch`) and #9 (experiment, branch
`exp/9-regpfn-phase1`). All numbers are from the snapshots below; d = 1; NLL in nats on A's 128 queries, y ∈ [0, 1].

## What was run (ulysses RTX 3090, 25 min training each, commit c1f9f3d)

| Run | Snapshot | Purpose |
|---|---|---|
| `aonly` | `results/regpfn/pilot_20261023-115248` | E1 A-only PFN (pfn-repro PFN, 6 × 256), E2 effective-data curves |
| `back true, n_iter 0` | `…-115419` | E3a / E3b: read-out of the true field |
| `back perturb, n_iter 8` | `…-115249` | E4: refinement |
| `back perturb, n_iter 0` | `…-115307` | E4 control: read-out only |
| `back perturb, n_iter 8, resid, max_step 0.1` | `…-123300` (commit ff4d20b) | refinement with explicit residuals (update below) |

## Numbers

- **E0** (tests/unit/test_regpfn.py, figure `reports/figures/regpfn/e0_prior_d1_C11.png`): B_inA exact < 5e-7 without noise,
  all cells, d = 1, 2. Coverage (inside fraction) 1.00 (C00, C01), 0.85 (C10, C11).
- **E1:** NLL −0.19 / −0.68 / −1.20 / −1.79 / −2.28 at n_A = 0 / 2 / 5 / 10 / 20 (C00); the cells agree within 0.13.
- **E2** (`e2_effective_data_curves.png`): monotone; all of B_inA added gives −2.98 (C00), −3.03 (C01), −2.64 to −2.94
  (C10, C11; lower coverage). The cells differ only through coverage.
- **E3** (true seed, n_iter 0): NLL −2.63 / −2.69 / −2.69 / −2.65 / −2.59 (C00, n_A = 0 … 20); m* = 37 / 41 / 38 / 28 / 14
  (C00), 19 / 25 / 26 / 21 / 8 (C10). Null weight rises 0.03 → 0.33 with n_A.
- **E4** (n_A = 10): y-perturbations recovered (field error 0.20 → 0.065 in C00 at σ = 0.2, NLL −2.05 vs −1.76
  read-out only); x/shift not (0.20 → 0.19, radius 0); refinement degrades perfect seeds (0 → 0.05–0.07 with T on).

## Update: refinement with explicit residuals and bounded steps (`…-123300`, commit ff4d20b)

At n_A = 10, with an unequal budget (18,300 steps vs 8,500 for read-out only):
- Perfect seeds are no longer damaged (field error ≤ 0.018), and with a perfect seed it beats read-out only (C00 −2.90 vs −2.65).
- Lower NLL than read-out only in every (cell, kind, σ).
- y-perturbations are corrected (0.20 → 0.085–0.106; radius 0.2 in C00, C10).
- x/shift are not corrected (radius 0); the model hedges with the null slot (w_null up to 0.66).

## Verdicts (pre-registered rules, drafted by Claude)

E3a **FAIL** (m* < 128), E3b **FAIL** (min m* 7.9–13.7 per cell). E4 is not interpreted (stop rule).

## Diagnosis and next steps

1. The learned read-out is a weaker reader than the A-only PFN. Use the A-only PFN itself as the seed slot's read-out
   (frozen or as the initialization): E3 then holds by construction and the field becomes the only question.
2. The weight head over-weights the null slot as n_A grows (every pair is related here). Feed it the slot's own
   predictive score on A's points (evidence weights).
3. Refinement fixes y, not x: a position error is visible only through y-mismatch at a few A points. The variant
   with explicit residuals and bounded steps is below.
4. Position errors need a direction signal: local slopes, or shape matching in the lookup, or the front half (GW landmarks).
5. Equal-budget runs (the variant had 2× the steps), longer training, then d = 2.

## Deviations from the spec, and [decide] items (defaults taken, flagged)

- BNN: pfn-repro's own ranges (PFNs4BO-style), evaluated *raw* (unnormalized) so f_A extends beyond [0, 1]^d.
- T: s_max 0.3, a ∈ logU[0.7, 1.4], Kumaraswamy log-sd 0.35 (identity outside [0, 1]); order warp ∘ scale ∘ shift.
- h: affine (α ∈ logU[0.5, 2], β ∈ U[−1, 1]) ∘ sinh-arcsinh warp (δ ∈ logU[0.5, 2], ε ∈ U[−1, 1]) on A's normalized scale.
  The affine part is removed by B's normalization.
- Normalization: per task, min/max of the noise-free function on a dense grid of the task's own window (512 points
  at d = 1, 64² at d = 2) onto [0.1, 0.9]; noise σ_ε ∈ logU[1e-3, 3e-2] added after it; clipped to [0, 1].
- Bar distribution: 64 *uniform* bins on [0, 1] (not prior quantiles).
- Model sizes: A-only PFN 6 layers × 256 (spec default); back half d_model 128, 3 read-out layers, T_iter 8, Fourier
  features (8 octaves).
- Not done: A_inB (not needed for the back half); W&B; d = 2 runs; the shift/scale/warp sub-ladder.
- The E2 curve uses inside-window B_inA points only.
