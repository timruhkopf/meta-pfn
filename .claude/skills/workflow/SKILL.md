---
name: workflow
description: Keep this research repo's git/GitHub/experiment workflow clean. Use when a piece of work is done or about to start, before or after a commit, push, pull, branch, PR or merge, before launching runs or spending compute, when results arrive, or when the user asks "what next" / "where are we". Gives the 1-3 next steps with their exact commands.
---

# Workflow reminders

Goal: a clean, consistent history without ceremony. This is research code, not a product: be brief and
reasonable, and remind only at natural checkpoints (work finished, before a launch, results in, end of a
session), at most once per checkpoint. Never nag about the same step twice.

## How

1. Run `just status` (read-only: git state, PR, sweeps that need a snapshot or a verdict).
2. Say what matters now in one or two lines, then give **at most three** next steps, each with the exact
   command in a code span. Skip anything that doesn't matter yet.
3. Run read-only commands yourself. Ask before anything that publishes, spends compute or records a
   decision: `git push`, `gh pr create/merge`, remote / SLURM launches, `just decide`.

## The steps and when they apply

- **Start of work**: an issue per idea/experiment/bug (`gh issue create --label experiment`), a branch from
  an up-to-date main: `git switch main && git pull --ff-only && git switch -c exp/<N>-<slug>`
  (`feat/`, `fix/` for non-experiments). Ideas without a plan yet are issues labelled `idea`.
- **While coding**: commit small and often; the commit hook runs ruff (if it reformatted files, `git add -u`
  and commit again). Commit before any tracked run, since provenance requires it.
- **Push**: at the end of a working session and before anything runs elsewhere: `git push` (first time:
  `git push -u origin HEAD`). The pre-push hook runs the fast tests.
- **Pull**: before starting on a branch others or other machines touched: `git pull --ff-only`. On a
  machine with running sweeps use `just sync-code` (code changes are safe; it refuses dependency changes
  under active sweeps unless forced).
- **Draft PR** once a branch has something worth seeing: `gh pr create --draft --fill` with `Closes #N`.
  It is the place to collect the evidence (sweep ids, snapshot, notebook).
- **Merge** (`gh pr merge --squash --delete-branch`, then `git switch main && git pull --ff-only`):
  - feature/fix: when `just test-all` passes;
  - experiment: when it is decided. GO: merge code, config, notebook and snapshot. NO-GO: merge the
    notebook, log entry and snapshot (they are results), and drop the experimental code from the PR.
  Sweep tags keep every run's commit reachable, so squash merges are fine.
- **Long-lived branch**: bring main in about weekly or before a confirm run: `git merge main`.

## What needs tests (research-level, not product-level)

- Yes: anything results depend on: priors, metrics, losses, data splits and transforms, statistics in
  `analysis/`, invariants of models (shapes, permutation/batch invariance). A new pipeline gets a tiny
  `experiment=<name>_smoke` config (the configs test composes it).
- No: plotting glue, notebook cells, one-off scripts, code in `experimental/` until it is promoted.
- `just test` before pushing (the hook does it); `just test-all` before merging into main.

## When to spend compute

A ladder; climb one rung at a time and say the cost before climbing:

1. scratch runs locally (`uv run <pkg> experiment=<name> ...`): minutes, CPU, no commit;
2. `just run local sanity experiment=<name>`: the pipeline works, controls behave;
3. a timing probe to estimate cost (`--cfg job` + one small run), then state "N members × t = X GPU-h";
4. pilot only once the notebook description says what GO / NO-GO means;
5. confirm only after a pilot GO (the gate enforces it); final only for the paper.

Ask before launching more than ~1 GPU-hour, anything on SLURM, or a second pilot of the same question
(two inconclusive pilots usually mean the question needs rethinking, not more compute).

## Results

When a sweep finishes: `just snapshot <dir>`, run its notebook (`just notebook <nb>`), propose a verdict
against the notebook description, and let the user run `just decide`. Then a track-record / log-entry
line, commit (`git add research_log notebooks results && git commit`), push.
