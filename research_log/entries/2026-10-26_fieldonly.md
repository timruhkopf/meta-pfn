---
experiment: fieldonly
sweeps: []
---
# Field-only: the second-stage operator / learned optimizer registers B into A's frame from a random start, without GW

## Question

See the description at the top of `notebooks/07_2026-10-26_fieldonly.ipynb`; not retyped here.

## §1 Pilot (2026-10-26, 13:24–13:40)

- **Ran** on ulysses, one sweep per method: `results/fieldonly/pilot_20261026-132409` (operator) and `…-132415`
  (optimizer) at e4422aa; `…-133636` (gd) at 249c589. All DONE (1/1 each; 0 failed / started / missing).
- **Correction:** the first gd sweep `…-132403` (e4422aa) started from a wrong field in about half the pairs (the slope
  recovered from two B points, with a clamped denominator); fixed in 249c589 with a test, rerun as `…-133636`.
  `…-132403` is superseded, kept for the record.
- **Verdict by the drafted rule:** USABLE, just (optimizer, C10, n_A = 10, random start: share 0.555 [0.496, 0.617],
  median err_x 0.046 [0.042, 0.050]). Learned beats gd: no (+0.11 at n_A = 10, +0.09 at 20 over gd_best8; +0.21 at
  n_A = 5). C00 no-harm: fails (0.026 > 0.02; gd also drifts to 0.026). The phase-1 operator: 1.6% (NOT USABLE).
- **Interpretation:** a gradient-fed learned optimizer finds the basin from a random start without GW, best at small
  n_A (it beats gd's oracle-of-8 at n_A = 5). Its failures are mostly not data-consistent, so they are precision and
  search failures, while gd's are information limits. It has a precision floor of 0.02–0.04.
- **Next:** optimizer + gd polish; K > 1 starts; the full model (read-out, NLL), then the RB / JEPA targets; d = 2, h ≠ id.
