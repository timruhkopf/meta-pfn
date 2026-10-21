---
description: Scaffold a new experiment (config + notebook description + log entry) and run its sanity check
argument-hint: <snake_case_name> "<question in one sentence>"
---
Create the experiment `$ARGUMENTS`.

1. Run `just new-experiment <name> "<title>"`.
2. Fill in the experiment config (`src/*/configs/experiment/<name>.yaml`): select the task, override only what
   differs from the defaults. Replicates are a sweep axis at launch time, not part of the config.
3. Write the notebook's **Description** cell with me: question, setup (method, baselines, null controls,
   data/instances, seeds), what to look for, what result means GO / NO-GO / INCONCLUSIVE, and limits
   (validity range of the components). Ask about anything you cannot infer; do not invent the criterion.
4. Run a scratch check (`uv run <package> experiment=<name> ...` with small sizes), then commit, then
   `just run local sanity experiment=<name>` and `just snapshot` it.
5. Add a track-record entry with the commit and what was verified.
