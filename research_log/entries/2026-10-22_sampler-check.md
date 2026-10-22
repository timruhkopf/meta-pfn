---
experiment: sampler_check
sweeps: []
---
# E0 follow-up: is the warp shortfall the sampler or the data?

## Question

See the description at the top of `notebooks/02_2026-10-22_sampler_check.ipynb`; not retyped here.

## §1 Why, and design (2026-10-22, 09:00)

- **Trigger.** E0 pilot (`runs/headroom/pilot_20261021-191753`), first look at the σ = 0 cells before the
  notebook: the Bayes learner's share of the known-T headroom, G, is ≈ 0.5–0.65 on window pairs but
  0.1–0.3 on warp pairs. Two explanations: A's points don't identify a warp (data), or the reference
  learner is a poor Bayes learner (sampler).
- **The pilot's own diagnostics favour the sampler** (window / warp, d = 1 and 2, σ = 0):
  - median ESS within the true kind falls to 1.2–3.8 at n_A = 8 (warp: 1.2 and 1.5);
  - the MC error between two independent hypothesis sets is 0.07–0.54 nats per pair, against the 0.025
    E0's rule allows: no σ = 0 window/warp cell meets it;
  - the warp gain *falls* from n_A = 2 to 4 at d = 1 (0.58 → 0.25) while the oracle's stays at 2.1.
  - The pair prior's observation noise sd is 4e-4–1.3e-3 on [0, 1] (pfn-repro BNN prior), so the
    likelihood of A's points is extremely sharp, and the prior draws are importance proposals that
    almost never land in the posterior.
- **SMC learner** (`smc.py`): targets the same posterior (the PFN chain-rule likelihood), with particles
  moved into it by tempering, resampling and Metropolis moves. Smoke on 3 d = 1 and 1 d = 2 warp pairs
  (256 particles): 75–105 s per pair and family on 16 threads, acceptance 0.23–0.32, 15–19 tempering
  stages by n_A = 8. The posterior RMS error of T over the box falls from ≈ 0.2 (prior) to 0.007–0.017
  at n_A = 8 on the d = 1 pairs (0.15 on the d = 2 one), and the family-conditional SMC predictive beats the
  importance-sampled one by 1–3 nats at n_A = 8 on all three d = 1 pairs, but still trails the oracle
  (e.g. −3.98 vs −4.94).
- **Design**: notebook 02's description. 16 pilot pairs × {window, warp} × d ∈ {1, 2}, three SMC fits
  per pair (true kind twice, the other kind once for the full mixture). ≈ 4.6 min per pair → ≈ 5 h as a
  detached sweep on the laptop (ulysses is unreachable). Pre-registered rule in the notebook.
- Code change outside the new modules: `oracle.evaluate_pair` also returns the per-kind evidence and
  predictive (tested to recombine to `bayes` exactly); `headroom.py`'s progress metric is cast to
  float (its type check only runs under pytest, so the pilot was unaffected).
