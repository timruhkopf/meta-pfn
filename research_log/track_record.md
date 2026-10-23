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
