# MetaPFN — Phase 1 Experiment Spec (for Claude Code)

> Hand-off spec. Read `metapfn_problem_context.md` (project context) and the design doc
> "MetaPFN — Registration Design: GW Landmarks + Displacement Field" first.
> **Working rule:** every component gets a cheap sanity check before it is used in an experiment.
> Items marked **[decide]** are open: ask the user before choosing, or use the proposed default and flag it in the results.

---

## 1. Goal of phase 1

Establish the evaluation frame and test the **back half** of the registration architecture (refinement + read-out + mixing) under controlled distortions, using **oracle or approximate-oracle correspondences** as input.

Phase 1 answers:
1. How much is B worth to A when the transfer is perfect? (upper reference / effective-data curve)
2. Can the read-out turn a correct correspondence into that value? (true-field seed)
3. How wrong may a correspondence be before refinement can no longer recover it? (perturbed-field seeds)
4. How do the answers differ between input distortions T and output distortions h?

## 2. Scope

**In scope**
- 100 % related tasks: B is always a distorted copy of A.
- One BNN draw f_A per task pair; B derived from it by transforms.
- Factorial over input transform T and output transform h (on/off).
- d = 1 first, then d = 2.
- No residual (η = 0).

**Out of scope for now** (later phases)
- Residual η·r_B, mixing several BNN instances, unrelated tasks (p_unrel).
- Front half: GW seeding, learned seeding, end-to-end training.
- MTPFN-style or flat task-ID baselines.
- d > 2, real HPO data.

## 3. Notation

- f_A: base BNN function on latent domain U ⊇ [0,1]^d.
- T: input transform U → U (shift, per-dim scale, monotone warp).
- h: output transform ℝ → ℝ (affine scale/offset, monotone warp).
- f_B(u) = h( f_A( T(u) ) ). Observations y = f(·) + ε.
- Correspondence: the B point at u lives in A's frame at **B_inA = ( T(u), h⁻¹(y_B) )**, with normalizations folded in (§4.4).
- n_A: A context size (0–20). N_B: B context size. A_test: query points on A.

## 4. Component 1: prior sampler

### 4.1 Base function
BNN prior exactly as in context doc §3: sample ψ = (L, H, σ_act, σ_w, σ_b, σ_ε), then an MLP f_A on U.
**[decide]** hyperparameter ranges for ψ. Default: reuse PFNs4BO-style BNN prior ranges.

### 4.2 Input transform T
Applied on the latent domain U, never constructed to map [0,1]^d onto itself (no bounding-box leakage).
- `shift`: u → u + s, s ~ U[−s_max, s_max]^d.
- `scale`: per-dim u → c + a(u − c), a ~ logU[a_min, a_max].
- `warp`: per-dim monotone map, e.g. Kumaraswamy-CDF or monotone rational-quadratic spline on an interval of U, extended as identity/linear outside.
- Composition order: warp ∘ scale ∘ shift.

Factorial setting `T ∈ {off, on}`, where `on` samples the full composition. Optional sub-ladder: `shift` → `shift+scale` → `shift+scale+warp`.
**[decide]** s_max, a-range, warp strength. Proposed default: s_max = 0.3, a ∈ [0.7, 1.4], moderate warp.

### 4.3 Output transform h
- `affine`: y → α y + β, α > 0.
- `monotone`: monotone warp (e.g. composition of a few softplus/sigmoid layers with positive weights, or a monotone spline).

Factorial setting `h ∈ {off, on}`. Optional sub-ladder: `affine` → `affine+monotone`.
Note: a negative α (rank inversion) is excluded in phase 1.

### 4.4 Windowing and normalization
- Sample x ∈ [0,1]^d for both tasks (B sampled uniformly; acquisition bias is deprioritized).
- Output normalization per task: an affine map to [0,1] computed from the **latent function over its sampling region U** (e.g. min/max on a dense grid or large sample), **not** from observed points.
- **[decide]** exact normalization scheme and how to handle noise pushing y outside [0,1] (clip vs. margin). Flag your choice.
- The oracle map must include normalizations: y_B^inA = norm_A( h⁻¹( norm_B⁻¹( y_B ) ) ).
- B points whose T(u) falls outside [0,1]^d are dropped from B_inA (log the fraction).

### 4.5 Output of one sample
```
A_train: (x, y) × n_A        A_test: (x, y) × n_test
B_train: (u, y) × N_B
privileged: B_inA (x̂, ŷ) × N_B (+ mask for outside-window), A_inB for A points,
            transform params, normalization constants
```
Proposed defaults: N_B = 256, n_test = 128, n_A ~ curriculum over [0, 20] weighted toward small values.

### 4.6 Sanity checks (must pass before any training)
1. **Exactness:** with σ_ε = 0, B_inA points lie exactly on f_A (max abs error ≈ float precision).
2. **Plots (d = 1):** for 20 random samples, overlay f_A, f_B, A points, B points, B_inA. B_inA must coincide with f_A.
3. **Leakage check:** the distribution of normalized y and of the positions of B's optimum is not trivially determined by the transform parameters (e.g. no systematic pinning of extremes to window edges).
4. **Coverage stats:** fraction of B_inA inside the window per factorial cell.

## 5. Component 2: references

### 5.1 Lower reference: A-only PFN
Standard PFN (context doc §2) trained on A data only from the same base prior. Bar distribution over y ∈ [0,1], bucket borders from prior y-quantiles.
**[decide]** model size. Default: small (e.g. 6 layers, d_model 256), enough for d ≤ 2.

### 5.2 Upper reference: effective-data curve
For each test sample:
1. Take A_train ∪ subset of m points from B_inA (random subsets, ≥ 5 permutations).
2. Predict A_test with the A-only PFN.
3. Record NLL(m) for m ∈ {0, 1, 2, 4, 8, 16, 32, 64, 128, 256}.

Report the curve per factorial cell and per n_A bin.
Note: B_inA y-values carry noise distorted by h⁻¹, so the curve is an approximate ceiling, not a strict bound.
Caveat to check: the A-only PFN must have seen contexts of size n_A + m during training, otherwise large m is out of distribution. Train it with context lengths up to ~300.

### 5.3 Effective-data metric m*
For any model, m* = the m at which the reference curve (same cell, same n_A bin) attains the model's NLL, by monotone interpolation. This is the headline metric for every later experiment.

## 6. Component 3: back half of the registration model

Architecture as in the design doc, steps 3–5. Phase 1 uses **one seeded slot plus the null slot**.

- **Slot state:** per B token i, a field (x̂_i, ŷ_i) in A's frame and a hidden state h_i.
- **Refinement operator** (weight-tied, run T_iter times):
  1. Lookup: B token at x̂_i cross-attends to A_train tokens with bias −‖x̂_i − x_a‖² / 2ℓ²; values carry residuals and A features.
  2. Update: GRU(h_i, [lookup, x̂_i, ŷ_i, context_i]).
  3. Propose δ_i = MLP(h_i).
  4. Coherence smoothing: attention over B tokens with bias −‖u_i − u_j‖² / 2ℓ_h² (B's own coordinates, learned ℓ per head); apply smoothed Δx̂, Δŷ.
  5. (Slot exchange: not needed with a single seeded slot.)
- **Read-out:** A_test tokens attend to A_train ∪ rendered B (x̂, ŷ) with distance bias in A's frame → bar logits per slot.
- **Null slot:** A-only path (shares the read-out, no B).
- **Mixing:** p(y*) = w_seed · p_seed + w_null · p_null; w from a learned weight head.
- **Losses:** mixture NLL on A_test + LUPI field loss ‖(x̂, ŷ) − B_inA‖ at every iteration with rising weights γ^(T_iter − t) (masked for outside-window points).

**[decide]** encoding of x (Fourier features recommended), T_iter (default 8), model width.

### 6.1 Sanity checks for the back half
1. With T_iter = 0 and the true field as seed, the read-out alone must approach the upper reference at m = N_B (see E3).
2. With n_A = 0, the refinement must leave the seed unchanged (no residual signal) — check field change ≈ 0.
3. Gradient check: LUPI loss decreases on a single fixed batch (overfit test).

## 7. Experiments

Factorial cells: **C00** (T off, h off), **C10** (T on), **C01** (h on), **C11** (both). d = 1 first, repeat in d = 2.

### E0: prior validation
All checks in §4.6. Deliverable: plot grid per cell + coverage table.

### E1: lower reference
Train the A-only PFN. Deliverable: NLL vs n_A.

### E2: upper reference curves
§5.2 for all four cells. Deliverable: NLL(m) curves per cell and n_A bin.
Expected: curves differ across cells only through coverage and noise distortion; a big difference signals a sampler or normalization bug.

### E3: sanity corner and true-field read-out
- **E3a (corner C00):** T = id, h = id, seed = identity = true field. Model NLL must reach the upper curve at m ≈ N_B and must never be worse than the lower reference.
- **E3b (all cells):** seed = true B_inA, T_iter = 0. Measures read-out quality alone. Target: m* close to N_B (inside-window count).

### E4: approximate oracle (perturbed seeds) and convergence radius
Seed = true field + perturbation, at increasing magnitude σ:
- **input perturbation:** x̂ = T(u) + δ(u), δ a smooth random field (e.g. GP or small BNN sample) scaled to amplitude σ_x; include a pure global-shift variant;
- **output perturbation:** ŷ = y_inA + smooth perturbation of amplitude σ_y (function of y, so it stays a map).

Train with σ sampled from a range; evaluate per σ on a fixed ladder.
Compare T_iter = 0 vs T_iter > 0 (does refinement help?), per cell.

Deliverables:
- m* vs σ (per cell, per n_A bin).
- Field error before vs after refinement vs σ → **convergence radius**: largest σ at which refinement reduces error by a set fraction (e.g. 50 %).
- Null-slot weight vs σ (should rise as the seed becomes useless).

Questions answered: is T or h harder to recover from? How does the radius scale with n_A?

## 8. Logging and outputs
- One config per run (YAML or Hydra); seeds fixed and logged.
- Metrics to CSV (and W&B if available): NLL by n_A bin, m*, field error (function space, on B points), null weight, fraction outside window.
- d = 1 diagnostic plots per experiment: f_A, A points, rendered B before/after refinement, predictive bands.
- A short `RESULTS.md` per experiment: what was run, plots, the numbers above, deviations from this spec, decisions taken on **[decide]** items.

## 9. Order of work
1. Prior sampler + E0.
2. A-only PFN (E1), then reference curves (E2).
3. Back half + §6.1 checks.
4. E3a → E3b → E4.
5. Repeat E2–E4 in d = 2.

Stop and report after each step; do not proceed past a failed sanity check.

## 10. Later phases (not now)
Front half (GW order-consistency seeding, multiple slots, slot exchange, weight CE), end-to-end training, residual η, mixed BNN instances, unrelated tasks (p_unrel corner: model must match the A-only PFN), higher d, real HPO data.
