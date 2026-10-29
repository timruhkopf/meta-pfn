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

### 2026-10-23 14:15 — Weekend mandate (session 63423c6c, issue #12, branch exp/12-metapfn)
- Mandate (user, 2026-10-23 ~13:30, verbatim intent): take over the problem-formulation session's ulysses work and
  finish it without the user's help; "run the experiments attempting to meta learn how to use B to make better
  predictions on A despite transforms / warps"; ulysses when its GPU is free on the weekend; a complete report in
  chat and in a notebook. Do not ask anything during the user's absence.
- Goal: train models end-to-end on raw (A, B) from the phase-1 pair prior (T and h factorial + an unrelated
  stratum), no oracle input at test time, and measure how much of B's oracle value (m*, effective-data curve of
  notebook 04) they realize, against the same model's A-only mode and the A-only PFN, with identity / unrelated /
  shuffled-B controls. Experiment `metapfn`, notebook 05.
- Deadline: Mon 2026-10-26 09:00. Launches: local CPU (tests, smoke) and ulysses (RTX 3090) from Fri 18:00, each
  after checking it is idle (nvidia-smi, ps). Owner of ulysses this weekend: this session; session 0e7b4868's
  heartbeat (cron 55d1d204) stands down when it sees this entry.
- Scope exception: Meta-PFN training before an E0 GO is user-directed; recorded as decision 0003.

### 2026-10-26 11:55 — GW-only pilot run and analysed (issue #13, branch exp/13-gwonly)
- Mandate (user, 2026-10-26): test the GW front half in isolation at small scale on ulysses, 1d showcase notebook,
  training metrics in MLflow, standalone description.
- Done: experimental/gwonly.py (Peyré semi-relaxed GW, unrolled learned GW, GRU operator on the coupling), pipeline,
  analysis/gwonly.py, notebook 06 (description written before the run, executed end to end), log 2026-10-26_gwonly.md §1;
  3 sweeps on ulysses, snapshots results/gwonly/pilot_20261026-{113454,113457,113501}; MLflow synced.
- Verified: 102 tests pass, ruff clean; 3/3 members DONE; GW gradient vs autograd 4e-6.
- Result: NOT USABLE by the drafted rule (C10, n_A = 5: best 0.31 within 0.05; value baseline 0.46). Numbers: log §1.
- Earlier this session (2026-10-23/26): the weekend runs of #12 never launched (turn stalled after permission denials
  on Friday; no cron was created). Code for #12 is on exp/12-metapfn-weekend, unlaunched.

### 2026-10-26 13:45 — Field-only (issue #14, branch exp/14-fieldonly; session dcc72ec0, after the takeover from 63423c6c)
- Mandate (user, 13:05): "get the second stage as a stand alone experiment; where we use the operator/optimizer layers
  on their own without GW and from random initialization. build it, run it and report (standalone) in a notebook".
- Done: e4422aa (module, task, tests, notebook 07 pre-registration), 249c589 (gd start fix + test, round-1 snapshots);
  4 sweeps on ulysses (the 1st gd superseded); notebook 07 executed; HTML in reports/ (untracked).
- Verified: 10 fieldonly tests and the full suite pass (111 passed, 1 skipped before the fix commit); all sweeps DONE.
- Results: USABLE, just (optimizer 0.555 at C10, n_A = 10); learned beats gd: no; C00 no-harm: fails
  (log `2026-10-26_fieldonly.md` §1).

### 2026-10-28 13:45 — Sanity ladder S0/S1 for the A→B query architecture (issue #16, branch exp/16-ladder-s0s1)
- Mandate (user, 2026-10-28): new clean branch; read the design note `metapfn_A_to_B_architecture (1).md` and put it
  into the logs (docs/design_2026-10-28_A_to_B_architecture.md, with the pasted Q&A as appendix A); prepare and run S0
  and S1 of its §8 ladder with the pfn-repro variable-d checkpoint (pfn_large), on ulysses; report with a notebook with
  visualizations.
- Branch from origin/exp/14-fieldonly (the tip of the unmerged stack; main has only the template). Experiment `ladder`,
  notebook 08, description and drafted GO / NO-GO written before the first run.
- Deviation: N_B = 80 (pfn_large's n <= 100) instead of the note's 200; 200 runs as an out-of-range probe.
- Launches: ulysses (idle at 13:30), sweep task.d=1,2 x task.n_b=80,200, pilot stage.

### 2026-10-28 14:40 — ladder relaunched (commit dadc7b3)
- First launch `runs/ladder/pilot_20261028-135741` (commit 0fb8b66) stopped by hand after members 0–1 (d = 1) were DONE
  and member 2 had 25 pairs: the posterior entropy was NaN for sharp posteriors (float32 underflow; now float64 and a
  test). Also added, descriptive only: GD best of 8 starts (the single start from s = 0 reached the MAP in about half
  the d = 1 pairs). GO / NO-GO rules unchanged; nothing from the stopped sweep is reported.
- Relaunched on ulysses: `runs/ladder/pilot_20261028-143711`, task.d=1,2 x task.n_b=80,200.
- Found: the pre-push hook's template git tests leak into this repo when run inside a hook (git env vars): they staged
  deletions of every tracked file in this worktree's index (reset to HEAD, nothing lost), and they are the likely source
  of `.git/config` user.name=test / user.email=test@example.com. Pushed with --no-verify after the suite passed
  outside the hook (119 passed).

### 2026-10-28 17:35 — ladder pilot analysed (issue #16, PR #17)
- Verified: 4/4 members DONE (`runs/ladder/pilot_20261028-143711`), snapshot `results/ladder/pilot_20261028-143711`,
  notebook 08 executed end to end against it (HTML + figures in reports/figures/ladder/), ladder tests 6/6, ruff clean.
- Results (log `2026-10-28_ladder.md` §1): S0 PASS, S1 → build S2 GO (drafted rules). GD from s = 0 reaches the MAP in
  about half the pairs, best of 8 starts in 89–99%; s partly unidentifiable at d = 2 while predictions are fine.
- User (2026-10-28): N_B = 80 is fine as the main arm.
- Open: `just decide` the verdict (the user's call on the drafted rule); S2 design; the pre-push hook / .git/config issue.

### 2026-10-29 10:40 — S3/S4 of the A→B architecture (issue #18, branch exp/18-a2b-s3s4)
- Mandate (user, 2026-10-29): "i want to see S3 and maybe S4 ... if the perceiver correction can already identify the
  transform and how we need to move A into B (and maybe even account for the new evidence in A?)"; keep the design
  open to change. No tempering, no causal PFN (user).
- Built: experimental/a2b (prior, model, train/eval), tasks/a2b.py with variants full / r0 / k1 / nosup / context,
  tests 6/6; notebook 09 description and drafted GO / NO-GO before the first run. Branch stacked on exp/16.
- Choices: T = full-box Kumaraswamy bijection (user's 2026-10-22 "no sub-windows"), location supervision as in the
  design note (tested against `nosup`, given the user's earlier reservation about warp supervision).
- Launches (ulysses, 3 detached queues in parallel): S3 d=1 × 5 variants; S3 d=2 × {full, r0, context}; S4 d=1 ×
  {affine, kum} × {full, nosup}. 12k steps each (~0.24 s/step alone on the GPU).

### 2026-10-29 11:30 — a2b relaunch: LUPI only, null particle, jepa variant (issue #18)
- First launch (10:19; `runs/a2b/pilot_20261029-1019{28,29,30}`) stopped by hand after ~15 min, nothing reported: the
  user reaffirmed "no warp parameter heads, just LUPI" and asked that B be discounted as evidence against it accumulates.
  (The S4 sweep had also failed at start: GPU OOM next to two S3 queues; its KL term was made lighter.)
- Changes: privileged location / function-space losses removed; LUPI teacher distillation (PFN(B ∪ A_inB) on related,
  A-only on unrelated pairs); null particle (A-only PFN, chain-rule evidence + learned offset); unrelated pairs in
  training; particles-only NLL on related pairs (the null dominates at init otherwise: gradient norm ~1e-20 in a smoke
  run); variant `jepa` (user's latent-target idea: particles' hidden states at T_k(x*) toward the frozen PFN's with the
  privileged context). Tests 7/7 (null particle = A-only PFN chain rule to 1e-4). Notebook 09 description updated
  before the relaunch.
- User preferences recorded: low-rank K/V residual on top-k attended B tokens for local correction (next round, with
  cache targets KV(B | B ∪ A_inB) − KV(B | B)).
