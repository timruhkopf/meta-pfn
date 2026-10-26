# Brainstorm: a two-context PFN that registers B onto A in context (2026-10-23)

Purpose: the design space for the weekend's Meta-PFN work (E1-prototype), before it is scheduled. Written
by Claude from the conversation of 2026-10-22/23. Not a pre-registration; the experiment notebook will be.
Literature ids marked *(to verify)* go through `/add-reference` before they are cited anywhere.

## 1. The problem, stated exactly

A and B share the search space [0, 1]^d. A is scarce (n_A = 0..8d), B dense (n_B = 32d..64d). The landscapes
are related by an unknown map T (A's coordinates → B's), possibly an output map g, and possibly a residual (ρ).
If T and g were known, the problem would collapse to a single-task PFN on B ∪ T(A), or equivalently on
B_inA ∪ A (B expressed in A's frame). They are not known, so the Bayes predictive is a **mixture**:

    p(y* | x*, D_A, D_B) = ∫ p(T, g | D_A, D_B) · p(y* | x*, D_A, D_B, T, g) d(T, g)
    p(T, g | D_A, D_B)  ∝ p(T, g) · Π_i p(y_i | x_i, D_A,<i, D_B, T, g)            (chain rule)

In the user's terms:
- n_A = 0: the **in-context prior**, B pushed through the prior over (T, g, relatedness). It is B "smeared" by
  how uncertain the correspondence is, mixed with the BNN prior via p(unrelated).
- n_A > 0: the **soft in-context prior**, the posterior. Its likelihood is the registration: how well each
  candidate correspondence explains A's points.

E0 and sampler_check measured this with the frozen PFN: B is worth 1.4–2.1 nats per query at n_A = 0, and with a
proper sampler the Bayes learner keeps 43–88% of the known-T headroom at n_A = 2–8 (G_smc, sampler_check
`results/sampler_check/pilot_20261022-085704` + `..._143116`). So the information is there; the question is
architectural.

## 2. Why plain attention finds this hard

1. **T is a global latent; attention is pairwise.** One attention step computes, per token, a weighted average
   over other tokens with weights from pairwise similarity of their embeddings. The correct correspondence
   x_B ≈ T(x_A) is not a pairwise property: it is a function of the *whole* context (all of A's points
   jointly constrain T). Pairwise x-similarity is simply wrong once T ≠ id; (x, y)-similarity is a candidate
   but ambiguous per pair (many B points share A's y-value: a level set).
2. **The answer is a mixture, attention averages.** With small n_A the posterior over T is broad or
   multimodal. The Bayes predictive mixes *predictions* under each T; a single attention pass that softly
   averages B tokens over candidate locations produces the prediction at an *averaged* location, which is
   neither mode and is overconfident or biased. A model needs a place to keep several hypotheses apart.
3. **There is no state that holds the hypotheses.** A has few tokens (that is the regime), so A's tokens cannot
   carry a posterior over T in their states; B's tokens each know only local things. A global hypothesis needs
   its own carriers.
4. **Precision vs noise.** At the BNN prior's noise (sd ≈ 1e-3) the posterior over T is razor-sharp and a T
   error of 0.01 costs nats (sampler_check). Softmax attention with bounded logits has limited spatial
   resolution; matching to that precision needs very peaked attention, which is hard to learn and unstable.
   At larger noise the posterior over T is broad and smooth, matching needs less precision, but the right
   answer is a wide mixture.
5. **Count imbalance.** n_B ≫ n_A: in joint attention, B tokens dominate softmax mass for every query
   regardless of how relevant they are (the doc's 1:1000 point; here 1:4 to 1:32).
6. **Copy detection is the easy shortcut.** The identity stratum (T = id) can be solved by x-matching; a model
   can look good by detecting copies and gain nothing on real warps (Meta_FTPFN v0 lesson).

## 3. What changes with larger noise

- **Identification gets blurry, not impossible.** The likelihood flattens, p(T | D) widens; fewer modes are
  separable; n_A needed to pin T grows. The value of B per query shrinks, but so does the cost of a slightly
  wrong T. Prediction: the gap flat-attention → Bayes is *largest at low noise* (precise, multimodal
  registration needed) and shrinks at high noise (blurring is near-optimal). So noise is a stratum to report,
  not a nuisance to average over.
- **Noise must be inferred.** If noise is a prior variable, the model must estimate it from the context. B is
  dense, so B gives a good noise estimate for B; A's noise may differ (per-task noise, #7).
- **References at higher noise.** The frozen pfn_large is only the Bayes reference at its own noise. For larger
  noise we need either a retrained single-task PFN (#7) or bounds from the same model (§5.3: the privileged
  mode is an in-run upper bound).
- **Reporting vs training** (#7): gains normalized by bounds at the same noise level for reporting; training
  stays plain (or Rao-Blackwellized, §5.2) NLL.

## 4. Constraints from the user (2026-10-22/23)

- Start from a PFN with **two contexts, A and B**; batteries included (A-only, B-only, A|B from one model).
- **LUPI during training**, as *contexts*; **no supervision of explicit warp/transform parameters**; deal with
  registration architecturally.
- Not: x-rank-based attention (acquisition bias, small n_A), explicit registration heads, "read B as a
  function" (B's predictive as a feature).
- Prior: no sub-window strata (they discard most of B); shifts and stretches as full-box monotone bijections;
  output maps through the ECDF.

## 5. Ideas

### 5.1 Baseline M0: two-context PFN

Tokens (x, y, role) with role ∈ {A, B, query}; per-layer self-attention within A ∪ B (role embeddings), queries
cross-attend to the context (standard PFN masking). Batteries included by masking: A-only (drop B), B-only
(queries on B, drop A), A|B. Bar-distribution head on [0, 1]. This is design 1 of the project brief and the
control every other design must beat at the same parameter count and training budget.

Cheap fixes worth including in M0 because they address §2.5 without assuming anything about T:
- **Per-role attention normalization**: a query's attention mass is split into an A-softmax and a B-softmax,
  combined with a learned (context-dependent) gate, so B's count cannot swamp A.
- **B self-attention can be cached** (B ↛ A masking) as an *option*, measured, not assumed sufficient.

### 5.2 LUPI as a proper loss: Rao-Blackwellized targets from privileged contexts

The cleanest way to use privileged information without supervising T:
- The PFN loss is E[−log q(y* | D)] with y* a sample. For *any* privileged variable Z drawn jointly with the
  training example (T, g, A's dense data, noise-free values...), E_{Z|D}[p(y* | D, Z)] = p(y* | D). So the
  cross-entropy against the soft target p(· | D, Z) has **the same minimizer** (the posterior predictive) and
  lower gradient variance than the one-sample target. It never names a transform parameter: Z enters only
  through a predictive distribution.
- **Who computes p(· | D, Z)?** (a) The frozen PFN with Z = T: oracle_T = PFN(B ∪ T(A)) at T(x*), exact at
  ρ = 0 and the prior's noise. (b) The frozen PFN with Z = dense extra observations of A itself: valid for
  *any* relation (warps, g, ρ > 0), approximate when Z is finite. (c) **The model itself in privileged mode**
  (next point).
- **Self-distillation with privileged contexts** (one network): in a random fraction p of training examples,
  privileged context tokens are present: B_inA (B's points mapped into A's frame; closed form for
  full-box Kumaraswamy bijections), or A's points in B's frame, T(x_A). In privileged mode the task is a plain
  single-task problem, which the network learns fast. In non-privileged mode the target is the stop-gradient
  privileged-mode predictive on the same example (plus a weight on the sample NLL to anchor it). Anneal p.
  This is LUPI as contexts, as asked, and the privileged mode doubles as an **in-run upper bound** at any
  noise level (§3).
- Risk: the privileged mode is approximate, so its predictive biases the target. Mitigation: keep the sample
  NLL term; measure the privileged mode against oracle_T where the frozen PFN is valid.
- Plain *privileged dropout* (privileged tokens present or not, sample NLL only, shared weights) is the
  simpler variant and the ablation: does the representation learned with privileged tokens transfer?

### 5.3 M1: hypothesis slots with evidence weights (an amortized SMC)

Give the posterior over T a home, and make the mixture explicit:
- K slot tokens (K ≈ 8–16), initialized from learned embeddings plus noise (an amortized proposal).
  Slots cross-attend to A and B and are refined over a few weight-tied iterations (a learned move step).
- Each slot k conditions a shared decoder: a PFN-style predictive q_k(y | x, D_B, D_A, slot_k).
- **Slot weights are computed, not learned freely**: w_k ∝ exp(Σ_i log q_k(y_i | x_i, A_<i, B)), the slot's
  chain-rule log likelihood of A's own points. That is the architectural analogue of the evidence in E0/SMC.
  It is computed by the same decoder (teacher forcing over A's order, or leave-one-out to remove order
  dependence). At n_A = 0 the weights are uniform: the in-context prior is the equal mixture of slots.
- Query predictive: Σ_k w_k q_k(y* | x*, ...), a mixture of bar distributions (§2.2 handled by construction).
- No supervision of T: slots are trained only through the query NLL (and the Rao-Blackwellized target).
  The mixture NLL itself rewards slot diversity when the posterior is broad.
- Failure modes to watch: slot collapse (all identical; then it is M0), and evidence weights that are
  overconfident because the decoder is (tempering, or a learned temperature, as a control).

### 5.4 M2: implicit registration field (latent coordinates)

Make "the meaning of x shifts" explicit without parameters:
- Each A token and each query gets a latent position x̃ ∈ R^d in B's frame, produced by the network from the
  context (initialized at x̃ = x: the identity is the prior mean of full-box bijections).
- Cross-attention from A/query to B uses **relative-position logits** on x̃ − x_B (an RBF or learned
  function of the difference) plus content similarity of (y, local descriptors). Refine x̃ over a few
  iterations (ICP-like: match, then move), weight-tied.
- x̃ is never supervised with T(x); it is a latent the NLL shapes. With LUPI (§5.2), the privileged tokens
  give the network examples of correct correspondences in its input, not as targets.
- Combined with M1 (one x̃ field per slot) this is the full "particles in latent coordinate space" design.
- Precision (§2.4): relative-position logits can be made sharp (a learned, noise-conditioned bandwidth),
  which content-only dot products struggle with.

### 5.5 The user's (x, y)-similarity, made concrete: two-hop attention

A query x* has no y. Its route to B has to go through A:
1. query → A tokens (by x in A's frame: nearby A points);
2. A tokens → B tokens (by (y, local shape) similarity and x̃, i.e. their inferred correspondence);
3. B tokens near the matched locations → the prediction.
In M0 this is possible in principle with ≥ 2 layers; making it the explicit block structure (A↔B matching
layers, then query→A→B) is a cheap architectural bias. With n_A = 0 there is no hop: the query attends to B
by x directly (the in-context prior, smeared by a learned bandwidth).

Local descriptors for matching: within-set attention lets each B token summarize its neighbourhood (slope,
curvature, rank among neighbours); A tokens are too sparse for neighbourhoods, so they match on y and on what
other A tokens imply. Output maps g break raw-y matching; a per-set monotone normalization (within-context
ECDF of y) removes g but is distorted by acquisition-biased designs; keep raw y and let the network learn,
add the normalized y as a second feature (ablation).

### 5.6 Smaller ideas

- **Noise token**: a context summary that estimates per-task noise and conditions attention bandwidths (§3).
- **Curriculum over relatedness**: start with identity and small warps, widen. (Training concern; ablation
  only if training is unstable.)
- **Order invariance of the evidence** (M1): average the chain-rule likelihood over a few random orders of A.
- **Shuffled-B control** at evaluation: same B values at permuted x; any "gain" there is a bug or copy detection.

### 5.7 Option M3 (user design, 2026-10-23): GW landmarks + displacement field

Full note: `docs/design_2026-10-23_gw-landmarks-displacement-field.md`. In short:
- **Step 0:** encode A and B separately (B is cacheable).
- **Step 1:** a Gromov–Wasserstein-style coupling π (n_A × N_B) finds a few internally consistent landmark
  matchings, using only within-task order relations (left/right in x, higher/lower in y).
- **Step 2:** each matching seeds a slot with a dense displacement field placing every B point in A's frame.
  There are also an identity slot and a null slot (ignore B).
- **Step 3:** weight-tied iterative refinement, RAFT/AlphaFold style: look up A's evidence near each B
  point's current position, propose a correction, smooth it over B neighbours, exchange between slots.
- **Steps 4–5:** read out per slot, then a mixture Σ_k w_k p_k.

How it relates to the ideas above:
- **It combines M1 (slots) and M2 (latent coordinates)** and adds a robust initialization. The direction is
  reversed: the field moves B into A's frame (B_inA), where M2 moves A and the queries into B's.
- **The order relations are pairwise**, so they hold under any monotone warp or output map. They also don't
  depend on how densely a region is sampled, unlike within-set ranks. So they should survive acquisition
  bias better than the x-rank attention ruled out in §4. The design's "within-task y-ranks" feature in
  step 0 is still density-dependent; it's an ablation.
- **The null and identity slots** give the in-context prior's mixture over "unrelated / same" for free.

Points to settle when it's built:
- **LUPI on the field** pulls x̂ toward the true B_inA. That is supervision of the (non-parametric) registration,
  which §4 asked to avoid. Run M3 with and without it. The without-LUPI variant, plus the Rao-Blackwellized
  target (§5.2), is the one that fits the weekend constraint.
- **LUPI on the weights** (cross-entropy to "the right slot") is proper only if slots map to hypotheses
  identifiably. Otherwise it fights the mixture NLL; ablate it too.
- **Cost:** O(K · T · N_B) per forward pass; fine at N_B ≤ 64d.
- **Reference:** Gromov–Wasserstein matching (Mémoli 2011) *(to verify)*.

## 6. What the weekend should measure

Prior (training and evaluation): f ~ BNN prior; T = per-axis full-box Kumaraswamy bijection
(log α, log β ~ N(0, s²), s ∈ {0.25, 0.5}) or identity; optionally g (E0b family); unrelated with p = 0.2;
ρ = 0 first. Noise: the prior's own, plus added observation noise σ ∈ {0, 0.01, 0.05} on the ECDF scale as a
stratum (A and B separately). d ∈ {1, 2}; n_B = 32d; n_A ∈ {0, d, 2d, 4d, 8d}.

Models at equal size and budget: M0, M0 + privileged dropout, M0 + Rao-Blackwellized self-distillation,
M1 (slots + evidence), M2 (latent coordinates), and M3 (GW landmarks + displacement field, §5.7, which
subsumes M1 + M2), with and without its field/weight LUPI terms.

Read-outs per (stratum, noise, d, n_A):
- NLL on A's queries; gains vs the same model's A-only; vs the frozen PFN's A-only (marginal check: gains must
  not be prior mismatch);
- where the frozen PFN is valid (σ = 0 added): the exact references from E0 / sampler_check / E0b
  (oracle_T, bayes_smc), so G = share of the known-T headroom realized by one forward pass;
- elsewhere: the privileged-mode upper bound and n_eff in A-sample units;
- identity (copy detection), unrelated (negative transfer), shuffled-B controls; calibration (PIT).

Hypotheses to confirm or refute (cf. notebook 03, H2–H3): M1/M2 beat M0 most at low noise and small n_A;
LUPI self-distillation speeds training and improves small-n_A NLL; the in-context prior (n_A = 0) gain is
realized by all models (it needs no registration, only blurring).

## 7. Practicalities and risks

- **Scope.** CLAUDE.md gates Meta-PFN training on an E0 GO. This work is user-directed (2026-10-22); it is
  recorded as decision 0003 and lives in `src/meta_pfn/experimental/`.
- **Compute.** Training on the laptop CPU is too slow; ulysses (RTX 3090) is needed. Small models (d_model 256,
  6 layers, ~5M parameters), prior sampling on the GPU (pfn-repro's BNN prior is batched). Estimate per model
  and seed: 1–3 GPU hours to a usable curve; ~6 models × 2 seeds fits a weekend if ulysses is free.
  ulysses needs `just remote-setup cu126` and the pfn_large checkpoint there.
- **Training loop.** This repo was generated with `training=false`; pfn-repro has training pipelines but is a
  dependency, not a workspace: import its PFN blocks and bar distribution, write the loop here.
- **Biggest risks.** (1) Optimization: registration models are known to train slowly or collapse (Meta_FTPFN's
  LUPI/IRT line); (2) the frozen-PFN references exist only at the prior's noise; (3) a weekend gives one shot
  per design, so results are a pilot, not a verdict.

## 8. Literature to check *(ids to verify)*

PFNs (Müller et al. 2021, arXiv 2112.10510); PFNs4BO (Müller et al. 2023, arXiv 2305.17535); Neural Processes
(Garnelo et al. 2018, arXiv 1807.01622) and Attentive NPs (Kim et al. 2019, arXiv 1901.05761); Set Transformer
(Lee et al. 2019, arXiv 1810.00825); Perceiver (Jaegle et al. 2021, arXiv 2103.03206); Slot Attention
(Locatello et al. 2020, arXiv 2006.15055); LUPI (Vapnik & Vashist 2009, Neural Networks); generalized
distillation (Lopez-Paz et al. 2016, arXiv 1511.03643); privileged information via heteroscedastic dropout
(Lambert et al. 2018, arXiv 1805.11614); ICP (Besl & McKay 1992).

## 9. Design discussion, 2026-10-26 (user + Claude; decision 0003)

**M0 is a control, not a candidate.** The user trained the plain two-context PFN many times without it learning
to use a warped B. Direction: M3, simplified (decision 0003).

**Location and value are separate.** Each B point carries a field (x̂_i, ŷ_i): where it sits in A's frame and its
value on A's scale (ŷ = h⁻¹(y_B) after normalization). A match pins the displacement at a few B points; smoothing
with distance-biased attention in B's own frame u spreads it: a shift (constant displacement) is set by one
landmark, a stretch (linear) by two, a warp by more. Missing from the design: **monotonicity** of the field
(T is a bijection; a smoothed field can fold). At d = 1, per axis: parameterize x̂ as a cumulative sum of
positive increments in u.

**K = 1 first** (hypothesis slot = best GW matching, + null, + the identity field as a free fixed slot): no collapse,
no slot exchange, no slot-identity loss. Cost: a multimodal posterior over T is not represented. **Where K > 1 would
come from:** hypotheses are *born in GW* (distinct local optima of the non-convex GW objective, from restarts /
initializations, or K learned slot queries) and only *refined* by the field, which is a local optimizer and cannot
jump between modes (it can merge two that share a basin; dedup / slot exchange acts there).

**Evidence weights: chain rule, not leave-one-out.** The slot weight should be its marginal likelihood of A,
Π_i p(y_i | x_i, A_<i, B, slot) (each point used once, exact). Π_i p(y_i | x_i, B, slot) treats A's points as
independent given B (they share A's function where B doesn't pin it down) and LOO is a pseudo-likelihood (each
point in n − 1 contexts): both overcount and are overconfident. The PFN makes the chain rule order-dependent:
average over a few random orders. (The `metapfn` M3 code uses LOO with a learned temperature: a deviation to fix.)

**Testing the components separately against B_inA.**
- *Back half (field):* phase-1 E3 seeds the field *with* the oracle B_inA: nothing needs to move, so E3 tests the
  read-out (is a correct field worth its m*?). E4 seeds it with a **smoothly perturbed** oracle: x̂ = T(u) + δ(u),
  δ a random sum of 3 sines in u (or a global shift), RMS-scaled to σ ∈ {0.02, 0.05, 0.1, 0.2}; ŷ likewise with a
  smooth function of y. Not i.i.d. noise (smoothing removes that trivially): the perturbation imitates a seed that
  is coherently wrong, like a GW seed with a misplaced landmark. **Convergence radius** = the largest σ at which
  refinement cuts the field error by ≥ 50% (n_A = 10). Phase 1: radius 0.2 for y, 0 for x and shift.
- *Front half (GW), a GW-only experiment:* no training. Per pair, the true partner of A point a is u*_a = T⁻¹(x_a)
  (from B_inA). Score the coupling π (n_A × N_B): expected landmark error E_π|u − u*_a|, top-1 error, π-mass within
  ε of u*_a, and for multiple matchings recall@K (some matching has all landmark errors < ε); by n_A, cell, noise.
  Baselines: identity guess u = x_a, and value-only matching (nearest y in B, ambiguous on level sets).
- *Interface contract:* GW is good enough when its seed error lies inside the back half's convergence radius.
  Then compose, and compare with the oracle-seeded result (the gap is the front half's cost).

**Peyré et al. (2016) as the reference solver.** Entropic GW solves min_π Σ (C_A(a,a') − C_B(b,b'))² π_ab π_a'b'
− ε H(π) by projected mirror-descent / Sinkhorn iterations, with no learning. With C = within-task order relations
(sign(x − x'), sign(y − y')) it is exactly the step-1 consistency check done by an optimizer. Roles: (1) the GW-only
experiment runs on it directly (POT, `ot.gromov`), so it measures what order-GW *can* identify before anything is
learned; (2) a learned GW layer must at least match it; (3) restarts give K distinct optima for K > 1. Needed
variant: **semi-relaxed** GW (each A point matched, B's marginal free; balanced GW would force every B point to
be matched, including B points outside A's window). Sign-based order matrices also exclude the mirror solution
(x → 1 − x) that distance-based GW cannot tell apart.

**"Which B points do my A points match" vs "where does every B point land" are largely the same question** (both
estimate T); the difference is the job: GW is a global, combinatorial initializer (coarse, robust to large
displacements, multimodal), the field a local continuous optimizer (precise within its basin), as in
coarse-to-fine registration or ICP with an initialization.

**h strictly increasing** = a higher value on B always means a higher value on A: the output map preserves ranks;
B is A's landscape up to a reparameterization of x and a relabelling of y. Then the within-task y-order and the
x-order (monotone T) are both invariant, and order-GW is unaffected by h. Realism: for related HPO tasks this is the
idealized part; real pairs have imperfect rank correlation (local differences, interactions, modality changes,
`docs/hpo_task_distortions.md` §9–10), which this prior puts into the residual (out of scope in phase 1). Near-ties
(flat regions) and noise flip order relations; that is GW's real failure mode. **h = id first** (decision 0003).

**The operator's GRU.** Per B point, a hidden state across iterations with update gate z (stop moving once
converged) and memory of past residuals (momentum-like); RAFT's choice. It is per-point: interaction comes only
from the lookup and the smoothing. Phase 1's x-failure was a missing direction signal in its inputs, not the GRU:
hence the Gauss–Newton feature −r s / |s|² (r local residual, s local field slope). Alternative: a shared
transformer block over B tokens per iteration (AlphaFold-style). Open: detach the field between iterations (RAFT).

**What RAFT, AlphaFold and DETR offer, and the gaps.**
- *RAFT* (Teed & Deng 2020, arXiv 2003.12039, to verify): all-pairs correlation volume + a weight-tied ConvGRU
  update operator that looks up correlations at the current flow and refines a dense flow field, loss at every
  iteration. Maps to field / lookup / GRU / per-step LUPI. Gaps: two *dense* images with appearance features vs
  n_A ≤ 20 scalar observations (the "correlation volume" is nearly empty); h breaks brightness constancy; one
  flow, no mixture; trained with full ground-truth flow (= field LUPI).
- *AlphaFold2* (Jumper et al. 2021, doi 10.1038/s41586-021-03819-2, to verify): structure module with shared
  blocks updating per-residue frames, invariant point attention, recycling, losses on intermediate structures.
  Gaps: its strength is the pair representation from the MSA (our analogue: the A × B coupling π, not built
  yet); its invariance group is SE(3), ours monotone warps (infinite-dimensional, no IPA analogue); one structure
  plus a confidence, not a mixture.
- *DETR* (Carion et al. 2020, arXiv 2005.12872, to verify): learned object queries (slots) + Hungarian set loss,
  which settles which slot answers for which target. Maps to K hypothesis slots and to LUPI on the weights.
  Gap: one true T per example, so a set loss supervises one slot and leaves the others as "no object"; diversity
  would have to come from the mixture NLL. Slow convergence. Argues for K = 1 first.

**The Rao-Blackwellized target** (§5.2): replace the one-sample PFN loss −log q(y* | D) by CE(p(· | D, Z), q(· | D))
with privileged Z (B_inA, T, h, noise-free f). Since E_{Z|D} p(· | D, Z) = p(· | D), the minimizer is unchanged and
the gradient variance is lower (Rao 1945; Blackwell 1947, doi 10.1214/aoms/1177730497, to verify). Who computes
p(· | D, Z): the frozen PFN on A ∪ B_inA (near exact at the prior's noise) or the model's own privileged mode
(self-distillation; biased, keep the sample NLL). Related: distillation (Hinton et al. 2015, arXiv 1503.02531),
generalized distillation / LUPI (Lopez-Paz et al. 2016, arXiv 1511.03643; Vapnik & Vashist 2009), all to verify.

**The prior: where it ended up.** One BNN draw f_A per pair; B = h(f_A(T(u))) + noise with closed-form T (shift ∘
per-axis scale ∘ Kumaraswamy warp) and h (affine ∘ sinh-arcsinh); per-task normalization on a fixed grid;
B_inA in closed form (phase-1 spec §4, `experimental/regpfn/prior.py`). Cost: one batched BNN forward pass per
task plus the grid, on the GPU; streamable. Not built: the residual ("local differences"), which the user preferred
as an interpolation of y-surfaces with a second BNN draw (mixture weight) over an extra latent input. Open conflict:
§4/§6 above asked for full-box bijections (no window shift), while the phase-1 spec (later, user's) puts T on the
latent domain so shifts move B's window partly outside A's (coverage 0.85 in C10/C11); the code follows the spec.
