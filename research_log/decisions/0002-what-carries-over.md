# 0002 — What carries over from Meta_FTPFN, and what doesn't

Status: accepted, 2026-10-21. Source: `~/PycharmProjects/Meta_FTPFN`, branch `meta-pfn-task-pair-prior`
(71d0430), with its docs (`CLAUDE.md`, `docs/ARCHITECTURE.md`, `docs/decisions.md`,
`docs/meta_pfn/{brief_v3,task_pair_prior_spec_v4,assessment}.md`, lab book up to 2026-10-19).

## Context

Meta_FTPFN went through several research lines: LUPI and registration PFNs, an SDE registration branch,
then Meta-PFN v0 on a "similarity prior". By its own lab book (2026-10-19):
- the trained v0 beats the frozen marginal PFN mostly through *prior mismatch*: its own marginal
  already beats the frozen marginal PFN on unrelated pairs;
- what it learned is mostly *copy detection*: Δ_identity ≈ 1.1 nats, but Δ on related pairs is +0.03 with
  a CI touching 0;
- its shuffled-B control fails;
- the similarity prior's related pairs share little structure (Spearman ≈ 0.2–0.3).

The question every model there depended on had a designed gate (E1: does a Bayes learner gain from B?),
smoke-tested, but never run at full size. Four days of training preceded the answer to "is the
information there?".

## Decision

Start with that gate, made cheap: E0 measures it with the frozen pfn_large as predictive *and*
likelihood. No GP, no SMC on a misspecified surrogate, no training. Build a model only on a GO.

| Carried over | How | Why |
|---|---|---|
| The problem: scarce A, dense B, unknown warp; T maps A → B; forward maps only | `pairs.py` | the core idea; forward maps avoid the numerical-inversion failures the old invariants guard against |
| The strata identity / related / unrelated, and the identity anchor | `pairs.KINDS` | identity is the sanity stratum (B is exact data); unrelated tests negative transfer |
| Per-axis window + monotone warp families (spec v4, I1–I3) | `WarpFamily` | the realistic HPO case; closed form; nested (window ⊂ warp) |
| The E1 decision structure: ΔNLL vs A alone, G = share of known-T headroom, MC-error validity that can block KILL but never support it | `analysis/headroom.py` | sound and pre-registered; thresholds kept (0.05 / 0.10 nats, G > 0.25) |
| Baseline lessons: compare with the *same model's* A-only prediction, not a different PFN; include a no-registration pooling control | predictors `lower`, `naive` | v0's main "gain" was prior mismatch against a different model |
| Prior-calibration lesson: measure relatedness (Spearman) before trusting a prior | log §2, notebook 01 | the similarity prior was uncalibrated, at 0.2–0.3 |

| Not carried over | Why |
|---|---|
| The similarity prior (~400 lines: hierarchical ρ, loadings, Student-t noise, failures, output maps) | uncalibrated, and too many knobs to attribute a null result to anything. E0 uses one relatedness knob (σ), calibrated to Spearman 0.9 / 0.7 |
| ifBO as an editable external dependency | pfn-repro provides the BNN prior and PFN, pinned by tag and sha256 |
| Meta-PFN v0–v2 models, the aux heads, the LUPI / SDE registration stacks | gated on E0; v0's lessons (copy detection, shuffled-B failure) define E1's controls |
| GP + tempered-SMC oracle (E1) | the GP is misspecified against the BNN prior. The frozen PFN *is* the prior's predictive (up to approximation), so it gives the likelihood directly |
| Output maps per task (O1/O2), failure cliffs, rotations | out of E0's scope; they change "how related", which σ covers for now. Candidates for NARROW / later strata |
| Role randomisation, the A-only twin (S3), curriculum | training concerns, relevant only after GO |
