# Deferred and future work

Directions considered and deliberately **not** pursued now, each with the reason and what would make it
worth revisiting. Issues are the actionable backlog; this file is the scientific judgement behind what was
left out. It feeds the paper's limitations and future-work sections, and keeps a narrowed experiment honest
about what it no longer tests.

Add an entry whenever a design choice, a pre-registration or a verdict narrows the scope. Entries are
append-only like the track record: a later outcome is a dated **Status** line, not an edit of the text.

<!--
## <Short title> (YYYY-MM-DD)

- **Idea.** What would be done, in two or three sentences.
- **Why deferred.** The reason (cost, a missing prerequisite, out of scope, a verdict), with its evidence.
- **Revisit when.** The concrete trigger: a result, a resource, a question that would need it.
- **Pointers.** Log entry § / notebook / issue / references.
- **Status.** open
-->

## A Bayes reference for related-but-different tasks, σ > 0 (2026-10-22)

- **Idea.** At σ > 0, A's function is a weight-space relative of B's: B is still informative about A, as
  a soft prior on its shape, but no longer exact data. The E0 reference learners cannot express that:
  the frozen single-task PFN can only treat B's points as exact observations of the same function (after
  T), or ignore them. A proper reference needs p(f_A | f_B, σ): weight-space posterior sampling of the BNN
  (feasible at d = 1, expensive), or a model trained on the pair prior, which is a first E1 Meta-PFN.
- **Why deferred.** The E0 pilot (`runs/headroom/pilot_20261021-191753`) shows that both E0 learners are
  misspecified at σ > 0: with T known, pooling B costs 2–6 nats per query against A alone, and the Bayes
  learner falls back to roughly A alone. So the pre-registered GO cells (σ = 0.1) measure the reference
  learner's misspecification, not whether partial transfer exists. σ = 0 is pursued first.
- **Revisit when.** E0 at σ = 0 shows recoverable headroom, and before E1 is designed: E1's prior has to
  say how tasks differ, and its evaluation needs this reference.
- **Pointers.** Issue #3; log `2026-10-21_headroom.md` (§2 calibration); notebook 01.
- **Status.** open
- **Status (2026-10-22).** Tracked in issue #5, with options incl. lifting relatedness into an extra input dimension.

## Relatedness measured in noise units, not rank correlation (2026-10-22)

- **Idea.** Calibrate σ by the RMS gap between f_A and f_B relative to the observation noise, and/or add
  prior variants with larger observation noise, so that "related" means related at the resolution the
  likelihood sees.
- **Why deferred.** The pfn-repro BNN prior's observation noise sd is 4e-4–1.3e-3 on [0, 1]. σ = 0.05 keeps
  the median Spearman rank correlation at ≈ 0.9 (§2) but moves function values by many noise sds, so to a
  likelihood-based learner it is nearly an unrelated task. The calibration in rank terms (natural for HPO,
  where rankings matter) and the one in likelihood terms disagree by orders of magnitude. Needs the σ > 0
  reference above to be worth measuring.
- **Revisit when.** Together with the σ > 0 reference; or when real benchmark pairs (HPO-B, LCBench) are
  used, whose noise relative to task differences is far larger than this prior's.
- **Pointers.** Log `2026-10-21_headroom.md` §2; `pairs.py::function`.
- **Status.** open

## More dimensions and coupled warps (2026-10-22)

- **Idea.** E0 at d > 2, and warps that couple axes (rotations, general monotone maps).
- **Why deferred.** The particle reference degrades with the number of warp parameters (4 per axis); even
  at d ≤ 2 the importance sampler collapsed (sampler_check §1). Per-axis warps are the realistic HPO case.
- **Revisit when.** The SMC reference is validated at d ≤ 2 (sampler_check) and E0 is GO or NARROW.
- **Pointers.** Notebook 01, limits.
- **Status.** open

## Acquisition-biased designs for B and A (2026-10-22)

- **Idea.** Draw B's (and A's early) points from a previous optimizer's trajectory instead of uniformly:
  a real cheap BO policy, or a Boltzmann density ∝ exp(−f/τ) with random τ plus post-hoc log filtering.
  B then covers its good region densely and the rest thinly, as in a real previous HPO run.
- **Why deferred.** E0 first has to show that a uniform, dense B is worth anything (#1). If the selection
  depends only on observed points, it is ignorable for the Bayes posterior; it then matters for amortization
  (a PFN trained on uniform designs is out of distribution) and for coverage gaps, both E1 concerns.
  It also rules out architecture ideas that lean on B's or A's x-ranking.
- **Revisit when.** E1 prior design; it becomes its own GitHub issue then. #3 lists trace-biased B as one stratum.
- **Pointers.** #3; the project brief (memory, not in git) §4.
- **Status.** open

## Meta-acquisition: choose queries that disambiguate the relation (2026-10-22)

- **Idea.** Once T (and the output relation) is latent, some query points are worth evaluating mainly
  because they pin T down, which then makes all of B's data usable at once. An acquisition function should
  value that, e.g. entropy search over the joint of (x*, T), not only over A's surface. Early in a run, the
  best query may be the one that identifies the transform; this is likely where the largest gains over
  standard transfer BO lie.
- **Why deferred.** It needs a learner whose predictive is honest about T-uncertainty (E0 reference or an E1
  Meta-PFN) and a BO loop; Phase 0 measures prediction only.
- **Revisit when.** An E1 model realizes a gain and BO evaluation is designed.
- **Pointers.** Issue #6; E0 `bayes` / `smc.py` give p(T | D) at small d for a first test.
- **Status.** open
