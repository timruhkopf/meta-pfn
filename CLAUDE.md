# meta-pfn

In-context transfer between related tasks with a PFN: is a scarce target task's prediction improved by a dense, coordinate-warped source task, how much would a Bayes learner gain, and can a Meta-PFN amortize it?

@docs/agent_rules.md

## Current phase

Phase 0, E0 `headroom` (#1): is a dense related task B worth anything to a *Bayes* learner predicting a
scarce task A? Frozen pfn_large only, no training. Rule: `analysis/headroom.py::verdict`, pre-registered
in notebook 01. Gate: no Meta-PFN model or training code (E1, #2) unless E0 is GO.

## Project rules

- **pfn-repro is a dependency, not a workspace.** Never copy or patch its code here; change it in
  `../pfn-repro` (issue, branch, tag), then bump the tag (README § The PFN dependency).
- **The frozen PFN's range:** pfn_large, context n <= 100, d <= 6; keep n_B + n_A well inside it.
  pfn_smoke (n <= 20) is for tests only.
- **Warps map A's box forward into B's** (`pairs.apply_warp`), into [0, 1]^d. Nothing is ever inverted.
- **Pairs are identified by (split, d, kind, i)** (`pairs.pair_seed`): split 0 pilot, 1 confirm. Never
  tune anything on split 1.
- **Baselines use the same model**: every predictor in E0 is the same frozen PFN, with different
  contexts. A "gain" over a different model is prior mismatch, not transfer (decision 0002).
- **Always report** the identity sanity stratum, the unrelated (negative-transfer) stratum and the
  oracle's MC error next to any gain.
- Minimization conventions don't apply here (prediction, not optimization); y is on the BNN prior's
  ECDF scale [0, 1].
## Commands

`just` lists all recipes. Common: `just test`, `just lint`, `just smoke`, `just sweep <backend> <stage> experiment=<name> ...`,
`just sweep-status <dir>`, `just snapshot <dir>`, `just decide <dir> <verdict> "<reason>"`.
Process and conventions: `docs/process.md`. Machine notes: `docs/compute.local.md` (untracked).
