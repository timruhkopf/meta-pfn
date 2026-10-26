# Track record

**Append-only.** New entries go at the bottom; old entries are never edited, a later entry corrects an
earlier one explicitly. Times are local. Each entry names the commit it refers to, what was done, what
was **verified and how**, and numbers with the snapshot or artifact they come from.

Autonomous agent sessions start with their mandate (goal, deadline, which launches are allowed) and end
with deliverables and open questions.

## Entries

### 2026-10-23 08:40 — E0b outmap built and launched (commits e562821, 3d681cb)
- Done: σ → rho rename (ef6c99b); sampler_check fix of `kind_is` overwritten in place by SMC values, with a
  regression test, and an importance-sampling-only recompute (b0b81c7, `runs/sampler_check/pilot_20261022-143116`);
  E0b outmap pairs, predictors, AMIS over g, analysis rule, tests; notebook 03 description and working hypotheses.
- Verified: 81 tests pass, ruff clean; the recompute reproduces every shared predictor of the SMC sweep exactly
  (max |diff| 0.0); outmap sanity controls (log `2026-10-22_outmap.md` §1).
- sampler_check verdict by its rule: IMPRECISE (SMC replicate error > 0.10 in 4 of 6 warp cells); SMC closes
  49–72% of the importance sampler's gap at d = 1 warp (`results/sampler_check/pilot_20261022-085704` +
  `..._143116`). Not yet recorded with `just decide`.
- Not done: the overnight ulysses checks requested 2026-10-22 were not scheduled before the session paused.

### 2026-10-23 11:30 — E0b outmap pilot analysed (commit: see git log, "E0b pilot")
- Done: snapshots `results/outmap/pilot_20261023-083513` (d = 1) and `..._100509` (d = 2, fetched from ulysses);
  notebook 03 loads both and executes end to end (nbconvert); Observations and Verdict cells written; log §2.
- Verified: 6/6 members DONE, 50 pairs each; the rule gives IMPRECISE (5 gate cells over the MC limit), identity
  sanity passes (G = 1.00 at 8d). Numbers in log `2026-10-22_outmap.md` §2.
- Next: a finer-sampler re-run of the gate cells on ulysses (pre-registered consequence of IMPRECISE).

### 2026-10-23 12:00 — E0b finer-sampler re-run analysed
- Done: `results/outmap/pilot_20261023-112025` fetched from ulysses (the remote copy moved to runs/_moved_snapshots/);
  notebook 03 uses it for the gate cells; log §3.
- Verified: 2/2 DONE; the rule gives IMPRECISE with 1 imprecise gate cell (0.030 at d = 1, n_A = 2), down from 5.

### 2026-10-23 12:45 — Phase-1 round 1: back half built and evaluated (branches feat/8-regpfn-arch, exp/9-regpfn-phase1)
- Mandate (user, 11:30): 1.5 h autonomous; build the back half of the registration architecture and the phase-1 suite
  (docs/spec_phase1_experiments.md), run it on ulysses, with an architecture branch and an experiment branch.
- Done: issues #8, #9; commits 551f016 (architecture), c1f9f3d (experiment), 3ef56a5 (analysis), ff4d20b (residual
  variant); 4 runs + 1 variant on ulysses; notebook 04 executed; RESULTS.md; figures in reports/figures/regpfn/.
- Verified: 9 regpfn/config tests pass (exactness < 5e-7; field change exactly 0 at n_A = 0, also in the trained model);
  all 4 runs DONE; E2 curves differ only through coverage, as the spec expects.
- Results: E3a/E3b FAIL (m* 8–41 of 256); refinement recovers y-, not x-perturbations (log `2026-10-23_regpfn.md` §1).

### 2026-10-23 13:00 — Phase-1 round 1 closed (commit: "Phase-1 update")
- Verified: the variant run `…-123300` is DONE; the no-evidence check holds; numbers in notebook 04 (Update), RESULTS.md, log §2.
- Open: equal-budget comparison; a direction signal for position errors; the A-only PFN as the read-out; d = 2.

### 2026-10-26 09:16 — Session 0e7b4868 closed (deliverables and open questions)
- Weekend: no launches by this session. The weekend mandate was taken over by session 63423c6c (issue #12, branches
  exp/12-metapfn, exp/12-metapfn-weekend, notebook 05); this session's heartbeat (cron 55d1d204) only checked that its
  mandate entry was present, every 30 min, Fri 16:00 – Mon 09:00, and is now deleted.
- Deliverables of this session: E0b outmap (notebook 03, log 2026-10-22_outmap.md §1–3; rule verdict IMPRECISE, 1 gate
  cell at 0.030); sampler_check fix + importance-sampling recompute (rule verdict IMPRECISE); phase-1 round 1 (notebook 04,
  results/regpfn/RESULTS.md, log 2026-10-23_regpfn.md §1–2; E3a/E3b FAIL, refinement fixes y- not x-perturbations);
  issues #2 (rewritten), #5, #6, #7, #8, #9; draft PRs #10, #11; docs/brainstorm_2026-10-23_registration.md (+ §5.7 M3).
- Open: `just decide` for headroom, sampler_check and outmap (the user's call); phase-1 next steps (A-only PFN as the
  read-out, evidence-based slot weights, equal-budget E4, a direction signal for position errors, d = 2), unless
  notebook 05 supersedes them.
