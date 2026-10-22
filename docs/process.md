# How research works in this repository

This document is **owned by the template** (`research-template`) and is updated with `copier update`.
Project-specific rules go in `CLAUDE.md` and `research_log/decisions/`, not here.

## Contents

1. [Using the template](#using-the-template)
2. [Principles](#principles)
3. [Environment](#environment)
4. [Repository layout](#repository-layout)
5. [Experiment design: notebooks, stages, decisions](#experiment-design-notebooks-stages-decisions)
6. [Running experiments](#running-experiments)
7. [Configuration (Hydra)](#configuration-hydra)
8. [Tracking](#tracking)
9. [Provenance and reproducibility](#provenance-and-reproducibility)
10. [Data and checkpoints](#data-and-checkpoints)
11. [Analysis, snapshots, notebooks, figures](#analysis-snapshots-notebooks-figures)
12. [Research log and track record](#research-log-and-track-record)
13. [Literature](#literature)
14. [Writing research code](#writing-research-code)
15. [Development workflow](#development-workflow)
16. [Coding agents (Claude Code)](#coding-agents-claude-code)
17. [`just` recipes](#just-recipes)
18. [Naming conventions](#naming-conventions)

---

## Using the template

Projects are generated with [copier](https://copier.readthedocs.io/), not copied from a GitHub template, so
improvements to the process reach existing projects:

```bash
uvx copier copy --trust <path or git URL of research-template> meta-pfn   # answers the questions below
cd meta-pfn && git init && just setup cpu local
git add -A && git commit -m "Start from research-template"   # uv.lock included: tracked runs need a clean tree
uvx copier update --trust                                     # later: merge template changes into this project
```

| Question | Default | Effect |
|---|---|---|
| `project_name`, `package_name`, `author`, `email`, `github_user` | — | names in package, CLI, env-var prefix, CITATION.cff, badges |
| `python` | `3.12` | `requires-python` |
| `cuda_extra` | `cu126` | the GPU torch extra next to `cpu` (must match the GPU machines' driver) |
| `compute` | `[remote]` | `remote` (ssh box without a scheduler) and/or `slurm`; `local` is always there |
| `mlflow` | `true` | optional MLflow viewer fed from the run sinks ([Tracking](#tracking)) |
| `optuna` | `false` | worker-pool HPO on the sweep machinery |
| `training` | `false` | Fabric training loop with checkpoint/resume, and a `toy_train` example pipeline |

These answers are in this project's copier settings: [compute=remote,
mlflow=True, optuna=False, training=False]. Sections marked [remote], [slurm],
[mlflow], [optuna] or [training] apply only when that option is on.

**Who owns which file.** A file is either *template-owned* (changes come only from `copier update`) or
*project-owned* (created once, never touched by updates):

| Template-owned | Project-owned |
|---|---|
| `docs/process.md`, `docs/agent_rules.md`, `justfile` (it imports `project.just`), `src/meta_pfn/tracking/`, `src/meta_pfn/analysis/`, `scripts/slurm/`, `.claude/`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml`, `tests/template/` | `README.md`, `CLAUDE.md` (imports `@docs/agent_rules.md`), `project.just`, `research_log/`, `references.*`, `envs/*.example`, `configs/experiment/`, `tasks/`, everything else in `src/` |

Editing a template-owned file locally produces a merge conflict on the next update. Fix it in the template
instead and update. The template's own tests render a project for several option combinations and run
that project's lint, type check and full test suite.

---

## Principles

1. **Questions first, code second.** Every experiment's notebook states the question, and what result
   would mean GO or NO-GO, before the first run. No pipeline without a question.
2. **Controls beat point estimates.** Results are reported against baselines and null controls, with
   uncertainty over seeds or instances. Selection effects get a held-out estimate.
3. **Few rules, enforced in code.** Provenance, stage gates and the test-split embargo are checked by the
   tools, and failed or missing runs are always counted. Everything else is good practice, stated where it
   applies.
4. **One source of truth per fact.** The config holds the setup, the run sink holds the data and how it
   was run, the log entry holds the decision. Notebook headers are generated from these.
5. **Results are files, not prints.** Pipelines write tables (parquet) into the run sink. Notebooks and
   figures read committed snapshots and never recompute experiments.
6. **Prefer deleting over accumulating.** One pipeline per question, small modules, no speculative
   generality. Replicates are sweeps over a seed or instance axis, not new config files.
7. **Stay inside the validity domain.** Runs outside the range where components are valid (e.g. the
   training distribution of a frozen model) are separate robustness experiments.

---

## Environment

### Python

PyTorch comes in **mutually exclusive extras**, so one `uv.lock` serves every machine. **Always pass an
extra**: a bare `uv sync` can silently resolve a default torch wheel. The `just` recipes pass the extra
from `META_PFN_TORCH_EXTRA` in `.env`.

| Target | Command |
|---|---|
| Laptop / CI | `just setup cpu local` (= `uv sync --extra cpu --group notebooks` + git hooks + `.env`) |
| GPU box ([remote]) | `just remote-setup` (= `uv sync --extra cu126` on the box) |
| Cluster login node ([slurm]) | `uv sync --extra cu126 --frozen`; jobs use `.venv/bin/python`, never `uv run` |

On clusters, compute nodes often have no internet: sync on the login node, and put `UV_CACHE_DIR` on
scratch.

### Machine settings and secrets

Code and configs are identical everywhere. Everything machine-specific comes from environment
variables in a git-ignored `.env`. Committed templates live in `envs/`:

```
envs/local.env.example  remote.env.example  slurm.env.example  ci.env  secrets.env.example
.env                    # git-ignored, active setup (`just setup` / `just use-env <setup>` create it)
.env.secrets            # git-ignored, tokens; never shared between setups
docs/compute.local.md   # git-ignored: free-form machine notes (hostnames, quirks, GPU types)
```

| Variable | Example | Notes |
|---|---|---|
| `META_PFN_SETUP` | `local` / `remote` / `slurm` / `ci` | logged as a tag |
| `META_PFN_TORCH_EXTRA` | `cpu` / `cu126` | the extra `just` passes to `uv run` |
| `META_PFN_RUNS_ROOT` | `/scratch/<user>/meta-pfn/runs` | default `<repo>/runs`; shared FS on clusters |
| `META_PFN_DATA_ROOT`, `META_PFN_CKPT_ROOT` | `/scratch/<user>/meta-pfn/data` | defaults inside the repo |
| `META_PFN_REMOTE_HOST`, `META_PFN_REMOTE_DIR` | `<ssh-alias>`, `PycharmProjects/meta-pfn` | [remote], laptop `.env` only |
| `SBATCH_ACCOUNT`, `SBATCH_PARTITION` | `proj123`, `gpu` | [slurm], read by `sbatch` itself |
| `META_PFN_MLFLOW_URI` | `sqlite:////abs/path/mlflow.db` | [mlflow], default `<repo>/mlflow.db` |

Write absolute paths without `$USER`: `.env` is read by `just` and by Python, and neither expands variables
the same way. Hydra reads the roots with `${oc.env:VAR,default}` in `configs/paths/default.yaml`, so there
are no paths or hostnames in YAML or code. The CLI loads `.env` and `.env.secrets`, and SLURM array tasks
source both. `meta_pfn.utils.env.require(...)` fails at startup with the full list of missing
variables. `gitleaks` runs in pre-commit.

---

## Repository layout

```
meta-pfn/
├── README.md  CLAUDE.md  CITATION.cff  LICENSE  pyproject.toml  uv.lock
├── justfile  project.just       # template recipes / project recipes
├── .copier-answers.yml          # do not edit; used by `copier update`
├── envs/  .claude/  .github/workflows/ci.yml  .pre-commit-config.yaml
├── docs/
│   ├── process.md               # this file
│   └── agent_rules.md           # imported by CLAUDE.md
│
├── src/meta_pfn/
│   ├── main.py                  # the ONLY entry point (Hydra)
│   ├── configs/                 # shipped in the package
│   │   ├── config.yaml
│   │   ├── experiment/          # one file per question (config only)
│   │   ├── task/                # one file per pipeline; add groups per component type (model/, data/, ...)
│   │   ├── runtime/  paths/
│   │   └── hparams_search/      # [optuna]
│   ├── tasks/                   # pipelines: one function each, registered in TASKS
│   ├── data/  evaluation/       # split guard, metric names; project code grows here (models/, ...)
│   ├── tracking/                # run sink, provenance, sweeps, backends, logbook (decide, gates), cli,
│   │                            #   reproduce, scaffold, [mlflow_sync], [optuna_study]
│   ├── analysis/                # load_runs, stats (DONE-only aggregation), snapshot, notebook header
│   ├── training/  models/       # [training] Fabric loop, callbacks, example MLP
│   ├── experimental/            # quarantined ideas (excluded from pyright)
│   └── utils/                   # env, seeding (keep small)
│
├── tests/  template/  unit/  configs/   # template/: the machinery (template-owned); conftest.py
├── notebooks/                   # analysis only, executed, outputs committed
├── research_log/
│   ├── README.md                # index, generated by `just log-index` / `just decide`
│   ├── track_record.md          # append-only session diary
│   ├── deferred.md              # deferred and future work: what was left out, why, when to revisit
│   ├── decisions/               # design decision records (ADRs)
│   └── entries/                 # one per experiment
├── references.md  references.bib
├── scripts/  figures/  slurm/   # one script per paper figure / [slurm] presets + array_task.sh
├── data/                        # git-ignored except data/README.md and data/sample/
├── runs/  checkpoints/          # git-ignored (or symlinks to scratch)
├── results/                     # committed snapshots
└── reports/figures/             # final figures, regenerated from results/
```

---

## Experiment design: notebooks, stages, decisions

### An experiment is a config plus a notebook

`just new-experiment <name> "<title>"` creates three files:

- the config `configs/experiment/<name>.yaml`, which holds only configuration;
- the notebook `notebooks/NN_<date>_<name>.ipynb`, which starts with the description;
- the log entry `research_log/entries/<date>_<name>.md`.

The notebook is written **before** the run and completed once results exist. Its top always has the same
three parts. Only the first is written by hand; the other two are generated from the snapshot by
`show_header(snap)`.

1. **Description** (markdown, written before the first run and kept up to date):
   - the question and why it matters;
   - the setup in words (what is compared against what, baselines and null controls, seeds/instances);
   - what to look for, and what result would mean GO, NO-GO or INCONCLUSIVE;
   - known limits (for example, the range a frozen model was trained on).
2. **How it was run**:
   - the exact command, e.g. `just sweep detached confirm experiment=warmup_variance seed=0,1,2,3,4`;
   - stage, sweep id, git commit and tag, backend, host, first start and last end;
   - member status counts (done / failed / started / missing), wall time and GPU time;
   - a **Preliminary** warning when not all members are DONE, and **Not reportable** for `--allow-dirty`.
3. **Configuration**:
   - the experiment YAML as committed at the sweep's commit;
   - the varying parameters, one row per configuration, and the replicate axes (`seed ∈ {0, 1, 2}`);
   - the constants grouped by component (`task`, `model`, ...), not a flat dump.

Then come the analysis cells and, at the end, the **verdict**: the outcome described in part 1, evaluated
and stated in one line.

### Stages

The stage is an argument of the launch (`just run|sweep <backend> <stage> ...`). It is recorded with
the run and shown in the notebook header.

| Stage | Purpose | Gate to launch | Reportable |
|---|---|---|---|
| `sanity` | Pipeline is correct: tiny case, metrics match a hand-computed value, controls behave as controls | — | no |
| `pilot` | Is there a signal worth paying for? | — | no |
| `confirm` | Full-size run with baselines at the same commit | a `pilot` of this experiment decided `GO` | yes |
| `final` | Frozen numbers for the paper; the only stage that may read the test split | a `confirm` decided `GO` | yes |

Enforced in code:

- **Gates.** `just run|sweep … confirm|final` refuses unless the log entry records the required `GO`.
- **Test-split embargo.** `meta_pfn.data.splits.check_split(split, stage)` raises for `test` unless
  the stage is `final`. Call it wherever a split is loaded.

Good practice (not enforced, stated in the notebook description):

- Baselines get the same tuning budget as the method, and comparisons are paired (same seeds or
  instances).
- When a reported result was picked as the best of many (HPO winner, best of K variants), re-measure it
  on held-out seeds or instances and report that number.
- Runs outside a component's valid range (e.g. contexts larger than a frozen model saw) are a separate
  `<name>_robust` experiment, never mixed into main results.

### Recording decisions

```bash
just decide runs/warmup_variance/pilot_20261021-091500 GO "std ratio 0.62 (CI 0.48-0.81)"
```

`decide` writes the verdict into the YAML header of the experiment's log entry, together with the stage,
commit, date and member counts. It then regenerates the log index. That header is the one official
record, kept in git, and the stage gates read it. A sweep can be decided only once.
The next `just mlflow-sync` mirrors the verdict as tags on the MLflow parent run.
NO-GO and INCONCLUSIVE are results: they stay in the log and in `results/`.

---

## Running experiments

### One entry point, two kinds of runs

```bash
uv run meta_pfn experiment=baseline task.lr=0.1        # scratch run: no commit needed, never reportable
uv run meta_pfn experiment=baseline --cfg job          # print the resolved config
just run local pilot experiment=baseline                  # one tracked run
just sweep local pilot experiment=baseline seed=0,1,2     # a tracked sweep (Hydra sweep syntax)
```

**Tracked work goes only through `just run` / `just sweep`.** A plain `uv run meta_pfn` is for
debugging, a timing probe or trying a config. It writes to `runs/_scratch/`, is never reportable, and needs
no commit. An important one-off experiment uses `just run`: it is a sweep of one member, with the same
provenance check, tag, notebook and decision as any sweep. `--allow-dirty` launches a tracked run from
uncommitted code for debugging; such runs are not tagged and are marked not reportable.

### Sweeps: plan once, execute anywhere, resume any time

```
runs/<experiment>/<stage>_<YYYYMMDD-HHMMSS>/   # the sweep dir; its name is the sweep id
├── sweep.json      # exact command, stage, commit, tag, overrides, backend, host, created
├── jobs.jsonl      # {"member": i, "overrides": [...]}, one line per member
├── queue.log       # detached backend / slurm_*.out for the slurm backend
└── <i>/            # one member: Hydra run dir
    ├── .hydra/  tracking/  stdout.log  [last.ckpt]
    └── DONE | FAILED      # FAILED contains the traceback
```

`just sweep <backend> <stage> <overrides>`:

1. checks provenance (clean tree) and the stage gate;
2. expands the overrides into `jobs.jsonl` and records the command in `sweep.json`;
3. tags the commit `sweep/<experiment>/<sweep_id>`, so every logged SHA stays reachable whatever happens
   to branches later;
4. hands the members to the backend.

No process has to stay alive for a sweep to finish. All state lives in the sweep dir and every member is
idempotent. `just sweep-status <dir>` and `just sweep-resume <dir> [backend]` work the same for all
backends. A member is `DONE`, `FAILED`, `STARTED` (running, killed or preempted: check before resuming)
or `MISSING` (never started).

| Backend | Executes members | Notes |
|---|---|---|
| `local` | one after another, in the foreground | Hydra `-m` works too, for quick scratch sweeps |
| `detached` | one after another, in a new session that survives logout | for an ssh box; log in `<dir>/queue.log`, pid in `queue.pid`; `--start-at 20:00` plans and freezes now, starts then |
| `slurm` | one array element per member (`--preset gpu_small`, `--max-parallel 16`) | [slurm] `--signal=B:USR1@120 --requeue`: members checkpoint (if they can) and requeue |

**Members run from frozen code.** Planning a sweep creates `<sweep dir>/.code`, a git worktree at the sweep's
commit, and every member runs from it (it is removed once all members are DONE, and recreated by
`sweep-resume` if needed). Committing, pulling or pushing while a sweep is queued therefore neither crashes
nor changes it. Only dependencies are shared (`.venv`): `just sync-code` / `just push` refuse a change to
`uv.lock` or `pyproject.toml` while sweeps are active unless forced, and after a plain `git pull` the
post-merge hook lists the active sweeps and warns about dependency changes.

### [remote] Remote box workflow

```bash
just remote-setup                            # once: git repo, venv (uv sync --extra cu126) and .env there
just push                                    # committed HEAD -> box (git over ssh), checked out there
just remote sweep detached pilot experiment=x seed=0,1,2   # plans ON the box: provenance is checked there
just remote-ps                               # verify; an ssh timeout often means the launch succeeded
just remote sweep-status runs/x/pilot_...    # any cli command runs remotely the same way
just pull x                                  # runs/x/ and the sweep tags back; then snapshot and analyse locally
```

Code reaches the box by `git push` over ssh, so the box needs no GitHub credentials and only ever runs
committed code. `just push` refuses dependency changes while sweeps run there (`just push force=1`
overrides; then sync the venv with `just remote-setup` once they finish). Launch once. Verify with
`just remote-ps` before relaunching, and never `pkill -f`. `just pull` assumes the box keeps runs in
`<repo>/runs` (no `META_PFN_RUNS_ROOT` there); afterwards back up the tags with
`git push --tags`.

---

## Configuration (Hydra)

Config paths here are relative to `src/meta_pfn/configs/`.

- **Composition over copies.** One config group per component type. Experiments select and override;
  they never duplicate. **Replicates are a sweep axis** (`seed=0,1,2,3,4`), never `foo_rep1.yaml`.
- **Components are built with `_target_`**, so a new model or method is a class plus a YAML file. Nested
  instantiation at most 2 levels deep; `_partial_: true` for factories (optimizers).
- Interpolation for derived values (`model.in_dim: ${task.data.dim}`), never duplicated numbers.
- **Semantic vs runtime.** Everything that can change results is semantic and becomes run params.
  `runtime/` (workers, device, precision) and `paths/` don't. If an experiment varies a runtime value,
  move it into that experiment's semantic config.
- **Only `main.py` and `tasks/` see `DictConfig`.** Library code takes typed arguments and works without
  Hydra.

```yaml
# configs/config.yaml
defaults:
  - task: ???            # each experiment selects its pipeline
  - runtime: default
  - paths: default
  - experiment: null
  - _self_

seed: 0
stage: scratch           # set by `just run` / `just sweep`; read by the test-split guard
experiment_name: ${or_scratch:${hydra:runtime.choices.experiment}}   # resolver registered in main.py

tracking:                # bookkeeping, set by the sweep machinery; never a semantic param
  sweep_dir: null
  member: null
  replicate_keys: [seed] # excluded from config_hash

hydra:
  run:
    dir: ${paths.runs}/_scratch/${experiment_name}/${now:%Y%m%d-%H%M%S}
```

```python
# src/meta_pfn/main.py
@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> float | None:
    """Dispatch to a pipeline; the return value is the optimized metric for Hydra sweepers."""
    with tracked_run(cfg) as run:
        return TASKS[cfg.task.name](cfg, run)
```

A pipeline is `def my_pipeline(cfg: DictConfig, run: Run) -> float | None`, registered in
`tasks/__init__.py`, with `configs/task/my_pipeline.yaml` (`name: my_pipeline` plus its parameters).
`tasks/toy_regression.py` is the pattern to copy. Tracked runs override `hydra.run.dir` to the member
directory. If no experiment is selected, `or_scratch` maps the missing choice to `_scratch` instead of
producing `runs/None/`.

---

## Tracking

### The run sink is the source of truth

Every run writes only to its own directory. Nothing in a run talks to a server:

```
<member dir>/tracking/
├── run.json        # params, tags, summary, status, command, overrides, start/end, wall_s, device_h, restarts
├── metrics.jsonl   # append-only {"key", "value", "step", "ts"}; a requeued run keeps appending
└── artifacts/      # tables/<name>.parquet, config/resolved.yaml, anything logged with log_artifact
```

Pipelines log through the `Run` object they receive:

- `run.log_metric(key, value, step)` / `run.log_metrics({...}, step)` for series;
- `run.log_summary(**{"final/eval_mse": x})` for the scalars in result tables;
- `run.log_table(name, df)` for result tables;
- `run.log_artifact(path)` for files;
- `run.set_tags(...)` for extra tags, e.g. a `data_hash`.

Library code doesn't import tracking.

| What | Where | Why |
|---|---|---|
| Full **semantic** config, flattened (`task.lr`, ...) | `run.json` params | complete, so defaults are explicit |
| Runtime and paths config | `artifacts/config/resolved.yaml` | doesn't affect results |
| `config_hash` (semantic params without `seed`) | tag | groups replicates of one configuration |
| `experiment`, `stage`, `sweep_id`, `member`, `git_commit`, `git_dirty`, `reportable`, `setup`, `hostname`, `slurm_job_id` | tags | identity and filtering |

Metric names: `<split>/<name>` for series (`step = global_step`); `best/…`, `final/…`, `test/…`
(stage `final` only) for summaries. Names are constants in `meta_pfn.evaluation.metric_names`.

### [mlflow] MLflow as a viewer

MLflow is an optional **index and UI**, never read by analysis. `just mlflow-sync [paths]` pushes sinks
one way and idempotently into `META_PFN_MLFLOW_URI` (default: `<repo>/mlflow.db`), then
`just mlflow-ui` opens it.

- MLflow experiment = `meta_pfn/<experiment>`, parent run = sweep, child run = member.
- The parent's description is the command, and its tags carry provenance and the verdict from the log.
- Metrics are pushed incrementally from a byte offset, so syncing twice changes nothing.

Only the sync process writes the store, so a SQLite file on NFS is fine. No MLflow server runs as a job.
If you never open the UI, switch the option off.

---

## Provenance and reproducibility

Every reportable number traces to **commit + config + environment + data + seed**.

- **Clean tree.** `just run|sweep` refuses if `git status` (untracked files included) shows anything outside
  an allowlist: `research_log/`, `notebooks/`, `results/`, `reports/`, `docs/`, `references.*`, `README.md`,
  `CLAUDE.md`. `uv.lock`, `pyproject.toml`, `justfile` and `scripts/` are **not** exempt: they can change
  results. Git is run with `-C <repo of the package>`, never relative to the cwd.
- **Same commit for every member.** Members run from the sweep's frozen worktree and check that it is at the
  sweep's commit; later commits or pulls in the main checkout can't leak into a running sweep.
- **Reachable commits.** Each sweep tags its commit (`sweep/<experiment>/<id>`). Squash and rebase are
  then harmless, and branches can be deleted freely. They are annotated and `just setup` sets
  `push.followTags`, so `git push` carries them to GitHub.
- **Environment.** `uv.lock` is committed, so the commit pins the environment.
- **Editable dependencies** (another local checkout installed with `uv add --editable`, e.g. a shared model
  library under development) are outside this repo's commit and `uv.lock`. Tracked runs refuse them
  if they have uncommitted changes, and record their commit in `sweep.json` and the notebook header.
  `confirm` and `final` refuse them altogether: point `[tool.uv.sources]` at a git tag first. Note that
  the frozen worktree does not freeze an editable dependency's code.
- **Frozen inputs.** Pin checkpoints and other external artifacts by sha256 in their config and verify it
  at load. Read training history and settings from the checkpoint itself, not from the config that claims
  to describe it.
- **Randomness.** `seed_everything(cfg.seed)` seeds Python, NumPy and torch (CPU+CUDA). GPU float
  differences can flip near-ties, so state reproduction tolerances and don't assume bitwise equality across
  batch shapes or devices.
- **Reproducing.** `just reproduce <sweep dir | snapshot dir>` creates a git worktree at the sweep's tag,
  runs `uv sync --frozen` there (or `--no-sync` to reuse this venv), reruns the members into
  `runs/<experiment>/<sweep id>_repro_<time>`, and compares every summary metric within tolerance.
  Snapshots keep the sweep's plan, so a fresh clone can reproduce reported numbers from `results/`
  alone. A test does both for a fresh debug sweep.

---

## Data and checkpoints

- Roots come from `META_PFN_DATA_ROOT` / `META_PFN_CKPT_ROOT` (`cfg.paths.data`,
  `cfg.paths.checkpoints`). `data/README.md` documents source, license, download and preprocessing.
  Preprocessing is a pipeline, so it is tracked like any other run.
- `data/sample/` is a tiny, license-compatible subset for the quickstart and CI.
- Check data at load time: schema/shape, value ranges, and **train/val/test overlap** on IDs or content
  hashes. Call `check_split(split, cfg.stage)` there.

---

## Analysis, snapshots, notebooks, figures

### Loading runs, including the ones that did not finish

```python
from meta_pfn.analysis import load_runs, paired_diff, summarize

df = load_runs("runs/warmup_variance/confirm_20261024-140000")  # one sweep dir, or a list of them
summarize(df, "final/eval_mse")                                        # per configuration
paired_diff(df, "final/eval_mse", "method", "ours", "baseline")        # paired on seed
```

`load_runs` returns one row per **planned** member, read from the sinks. It has a `status` column (`DONE`,
`FAILED`, `STARTED`, `MISSING`), the varying params, and the summary metrics. Constant params are in
`df.attrs["constant"]`.

**Aggregates use DONE members only, and report everything else.** `summarize` gives, per configuration,
`n_done`, `n_failed`, `n_started`, `n_missing`, then mean, std and a bootstrap CI over the DONE members. A
configuration with any member not DONE is flagged `preliminary`: its numbers can still change. By default
it groups by the varying params (without `seed`) plus `git_commit`, so replicates of different code
versions are never pooled; a warning says when that happened. `paired_diff` pairs on seed (or instance),
uses pairs where both sides are DONE, and reports `n_unpaired`. For small `n`, plot every seed, not only
the CI.

### Snapshots

Notebooks must still run when the cluster account has expired or `runs/` was cleaned, so they read a
committed snapshot:

```
results/<experiment>/<sweep_id>/
├── runs.parquet      # load_runs() output, including members that are not DONE
├── constants.yaml    # params constant across the sweep
├── columns.json      # which runs.parquet columns are params / metrics
├── tables/           # each result table of all members, with a `member` column
├── curves.parquet    # metric series, downsampled to 500 points per member and key
├── experiment.yaml   # the experiment config as committed at the sweep's commit
├── sweep/            # sweep.json + jobs.jsonl: the plan, for `just reproduce <snapshot>`
└── manifest.json     # command, sweep id, commit, backend, host, times, status counts, wall and GPU hours
```

`just snapshot <sweep dir>` creates it; re-running it refreshes it. In a notebook:
`snap = load_snapshot("results/<experiment>/<sweep_id>")`, then `snap.runs`, `snap.table(name)`,
`snap.curves()`, `snap.preliminary`. Keep snapshots small (no checkpoints), and move `results/` to Git LFS
if it grows past a few MB.

### Notebooks

- Named `NN_YYYY-MM-DD_topic.ipynb`. Each one is executed top to bottom (`just notebook <nb>`) and
  committed with its outputs. The `nbstripout` hook keeps outputs and drops volatile metadata (execution
  counts, Python version), so re-running doesn't create noise diffs. Use modest figure DPI.
- The second cell names the snapshot path. A notebook never reads `runs/` and never trains.
- Top of every notebook: **description** (hand-written) · **how it was run** · **configuration**
  (generated; see [Experiment design](#experiment-design-notebooks-stages-decisions)). Then the analysis,
  including failure cases, and the **verdict** (recorded with `just decide`).
- Plotting and statistics code lives in `meta_pfn.analysis` and is tested. Notebooks contain glue only.
- CI executes every notebook against its snapshot (`just notebooks-check`).

### Figures

One script per paper figure in `scripts/figures/`. Each reads its snapshot with
`load_snapshot(path, require_stage="final")` and writes into `reports/figures/`. `just figures` runs them
all. The README's **Claims** table maps each claim to its snapshot and figure.

---

## Research log and track record

```
research_log/
├── README.md         # generated index: entry | experiment | decisions
├── track_record.md   # append-only, chronological
├── deferred.md       # deferred and future work
├── decisions/        # ADRs: Context -> Decision -> Consequences
└── entries/2026-10-21_warmup-variance.md
```

**Entries** are one per experiment, created by `just new-experiment`:

```markdown
---
experiment: warmup_variance
sweeps:                         # written by `just decide`, never by hand
- id: pilot_20261021-091500
  stage: pilot
  commit: 1a2b3c4d5e6f
  decision: GO
  reason: std ratio 0.62 (CI 0.48-0.81)
  members: 10/10 done
  decided: '2026-10-22'
---
# Does warmup reduce seed variance?

## Question
See the description at the top of `notebooks/03_2026-10-21_warmup_variance.ipynb`; not retyped here.

## §1 Sanity (2026-10-21)
What ran, the result (numbers from the snapshot), interpretation, next step.
```

Sections are numbered and appended. Once a sweep is decided, its section is frozen. A correction is a new
section that names the one it corrects.

**The track record** is the session diary of people and agents. Each entry has a local time, the commit
it refers to, what was done, what was **verified and how**, and numbers with the snapshot or artifact path
they come from. It also records launches and their verification, and deviations from the mandate.
Entries are never edited; later entries correct earlier ones explicitly.

The backlog of ideas is GitHub issues (label `idea`). Log entries and PRs link their issue.

**Deferred and future work** (`deferred.md`) records what was considered and deliberately left out: the
idea, why it was deferred (with its evidence), the concrete trigger for revisiting it, and pointers. An
issue is a task someone could pick up; a deferred entry is a scientific judgement, and the paper's
limitations and future-work sections are written from it. Add one whenever a design choice, a
pre-registration or a verdict narrows the scope, so a narrowed experiment stays honest about what it no
longer tests. Entries are append-only; later outcomes are dated status lines (picked up in an issue or
entry, or dropped and why).

---

## Literature

`references.md` is the annotated list and `references.bib` holds the BibTeX. Keys match, and
`just refs-check` verifies both directions. Entries are grouped by the role they play in the paper:
problem, closest related work, method building blocks, baselines, datasets, evaluation, tools.

- Every baseline config, borrowed component and dataset links to its entry, and the entry links back
  (**Used in**).
- Entries need a DOI or arXiv id from the actual paper. BibTeX comes from the publisher or DBLP, never
  typed by hand or generated. Unverified entries are `status: to-verify`.
- Log entries cite with `[key]`. Claims are paraphrased.

---

## Writing research code

The goal: a new idea touches **one file plus one config**, and an abandoned idea is easy to delete.

1. **Narrow interfaces.** A few `Protocol`s define the contracts. Data flows as typed dataclasses, not
   string-keyed dicts or positional tuples.
2. **Composition over inheritance.** No `BaseX → MyX → MyXV2`. Variation comes from injected components.
3. **No flag soup.** `if cfg.use_x:` inside a pipeline becomes a component chosen in config. Boolean
   flags are fine only at the edge (`runtime.compile`).
4. **Side effects at the edges.** Core code is pure. I/O, tracking, seeding and device placement happen
   in `tasks/`, so the core is unit-testable on CPU.
5. **Rule of three.** Duplicate once; abstract on the third occurrence.
6. **Quarantine.** New ideas start in `experimental/`. After a GO they are promoted with tests and type
   hints; after a NO-GO they are deleted (the tag and the log entry remain).
7. **Delete aggressively.** No commented-out code, no unused options.
8. **Fail early and loudly.** Validate configs, env vars, data, shapes and checkpoint hashes at startup.
   Never fall back silently (e.g. to CPU when a GPU was requested).
9. **Frozen components stay frozen.** Code carried over from elsewhere (a pretrained model, a prior) is
   kept verbatim in its own module; new behaviour goes in new modules.
10. **`logging`, not `print`,** in library code.

### Typing

| Where | Policy | Checked by |
|---|---|---|
| Protocols, dataclasses, public functions of core modules, `tracking/`, `analysis/` | type hints required | pyright `basic`, in CI |
| Tensor arguments crossing module boundaries | `jaxtyping` annotations, e.g. `Float[Tensor, "batch dim"]` | beartype, **during tests** |
| Internals, `experimental/`, notebooks, `scripts/` | optional | excluded from pyright |

During tests, every annotated function of the package is checked at runtime by the jaxtyping pytest
plugin (`--jaxtyping-packages=meta_pfn,beartype.beartype` in `pyproject.toml`). This covers
every submodule whenever it is imported. Shape bugs fail at the call, with a message naming the dimension.
Annotations must therefore be honest: beartype rejects an `int` where `float` is annotated, so write `24.0`.

---

## Development workflow

### The workflow, step by step

`just status` shows where you are and the next steps; Claude shows the short form at the start of every
session and reminds you at natural checkpoints (the `workflow` skill).

1. **Idea.** Open an issue: `gh issue create --label idea` (or `experiment`, `bug`, `infra`). Issues are the
   backlog; an idea becomes an experiment once it has a question and a GO / NO-GO criterion.
2. **Branch.** From an up-to-date main: `git switch main && git pull --ff-only && git switch -c exp/<N>-<slug>`
   (`feat/`, `fix/` for non-experiments; `<N>` is the issue number).
3. **Set up the experiment.** `just new-experiment <name> "<question>"`; fill in the config, and write the
   notebook's description (what would mean GO / NO-GO) before anything runs.
4. **Try it cheaply.** Scratch runs (`uv run <pkg> experiment=<name> ...`) until it works; tests for anything
   results depend on (priors, metrics, losses, splits, statistics), not for plotting glue.
5. **Commit.** Small commits; the hook runs ruff and other checks (if it fixed files, `git add -u` and commit
   again). Tracked runs need a clean tree, so commit before every one.
6. **Push.** `git push` (first time `git push -u origin HEAD`) at the end of a session and before anything runs
   elsewhere; the pre-push hook runs the fast tests. Open a draft PR early: `gh pr create --draft --fill`,
   with `Closes #<N>` in the body.
7. **Spend compute in steps.** `just run local sanity ...` → a timing probe (state the cost: members × time)
   → `just sweep <backend> pilot ...` → `confirm` only after a pilot GO (enforced). Ask yourself before
   anything above about a GPU-hour.
8. **Results.** `just snapshot <sweep dir>`, `just notebook <nb>`, then `just decide <sweep dir> <verdict> "<reason>"`;
   a line in the log entry; commit notebook, snapshot and log; push.
9. **Merge.** Feature / fix: when `just test-all` passes. Experiment: when decided. GO merges code, config,
   notebook and snapshot; NO-GO merges notebook, log and snapshot (results are results) and drops the
   experimental code. `gh pr merge --squash --delete-branch`, then `git switch main && git pull --ff-only`.
   Sweep tags keep every run's commit reachable (`git push` carries them), so squashing is safe.
10. **Keep long branches current.** `git merge main` about weekly and before a confirm run.
11. **Machines with running sweeps.** Sweeps run from frozen worktrees, so pulling code is safe. `just sync-code`
    (cluster) and `just push` (GPU box) refuse dependency changes (`uv.lock`, `pyproject.toml`) while sweeps
    are active, unless you pass `force=1`; a plain `git pull` prints a warning instead.

### Branches, tags, commits

- **Trunk-based.** `main` plus short-lived branches per issue (`exp/<N>-<slug>`, `feat/`, `fix/`). Squash
  merges are fine: provenance doesn't depend on branches, because sweeps tag their commits.
- Tags: `sweep/<experiment>/<id>` (automatic), `vX.Y` or `paper-<venue>-<year>` at milestones. The code
  behind a paper is a tag referenced in the paper.
- Commit messages: imperative summary, then what changed and why. Mention sweep ids and log sections
  where relevant. Conventional-commit prefixes are welcome but not enforced.
- Version: static in `pyproject.toml`, bumped by hand at milestones.

### Hooks and CI

| When | What |
|---|---|
| commit (fast) | large-file check, YAML check, whitespace, `ruff check --fix`, `ruff format`, gitleaks, nbstripout (keep outputs), `uv lock` consistency |
| push | fast tests (`pytest`, without `slow` / `gpu`) |
| after `git pull` (post-merge) | lists active sweeps; warns if dependencies changed under them |
| CI (GitHub Actions) | `uv sync --extra cpu --group notebooks --frozen` → `just lint` (ruff, pyright) → `just test-all` (incl. slow: reproduction, notebook execution) → `just smoke` → `just notebooks-check` → `just refs-check` |

Commit hooks stay fast on purpose: tracked runs need a commit first, so slow commits push you toward
untracked runs. `pre-commit` is kept below 4.6.1, which needs git ≥ 2.31.

### Tests

- `template/` (template-owned): provenance (untracked files and `uv.lock` block, allowlist doesn't);
  gates and `decide`; `config_hash` properties (hypothesis); sweep expansion, failure, kill and resume;
  the CLI; the SLURM submission with a fake `sbatch` [slurm]; DONE-only statistics; sink → `load_runs` →
  snapshot → notebook header round trip; a scaffolded notebook executes; scratch runs and the test-split
  embargo; reproduction from a worktree; [mlflow] sync idempotency; [optuna] budget and reaping;
  [training] exact resume and shape checks.
- `unit/`: your pipelines and components, with hand-computed values and properties (hypothesis profiles
  `dev` ≈ 50 and `ci` ≈ 500 examples, selected by `HYPOTHESIS_PROFILE`).
- `configs/`: every experiment config composes and selects a registered pipeline.
- Markers `slow` and `gpu` are deselected by default; `just test-all` runs everything.

---

## Coding agents (Claude Code)

```
CLAUDE.md                 # project-owned: purpose, current phase, project rules; imports @docs/agent_rules.md
docs/agent_rules.md       # template-owned: rules for agents
.claude/
├── settings.json         # committed: permissions (allow / ask / deny) and the format hook
├── settings.local.json   # git-ignored: personal overrides, e.g. a mandate for an autonomous session
├── hooks/format.sh       # formats only the .py file that was just edited
├── skills/workflow/      # reminds of the next workflow step (with its command) at natural checkpoints
└── commands/
    ├── new-experiment.md # /new-experiment: config + notebook description + log entry + sanity run
    ├── review-sweep.md   # /review-sweep: failures, provenance, verdict against the notebook description
    ├── log-entry.md      # /log-entry: draft the next § of an entry from a snapshot
    └── add-reference.md  # /add-reference <doi|arxiv>: verified BibTeX + annotated entry
```

`settings.json` allows local, read-only and reversible commands (tests, lint, local runs, status,
snapshots, reading issues and PRs). It asks before anything that leaves the machine or records a decision:
push, PRs and merges, issue creation, remote and SLURM launches, resume, `decide`, `reproduce`. A
SessionStart hook shows `just status --brief` to Claude, so the session starts from the actual state.
It denies reading `.env.secrets`, `pip install`, `pkill`
and `pgrep`. For an autonomous session, move the needed `ask` entries to `allow` in
`settings.local.json`; that is where a mandate is granted. The format hook runs ruff on the edited file
only, so it never touches files the agent didn't change.

---

## `just` recipes

| Recipe | What it does |
|---|---|
| `just setup <cpu\|cu126> [setup]` | `uv sync` with the extra, `.env` from `envs/<setup>.env.example`, git hooks |
| `just use-env <setup>` | switch the active `.env` |
| `just smoke` / `just test` / `just test-all` | debug run end to end / fast tests / all tests incl. slow |
| `just lint` / `just fmt` | ruff + pyright (as in CI) / format and fix |
| `just run <backend> <stage> <overrides>` | one tracked run |
| `just sweep <backend> <stage> <overrides>` | a tracked sweep: provenance + gate, plan, tag, execute |
| `just sweep-status <dir>` / `just sweep-resume <dir> [backend]` | member states / re-execute members that are not DONE |
| `just runs <dir>` | table of a sweep (varying params, summaries, status) |
| `just decide <dir> <GO\|NO_GO\|INCONCLUSIVE> "<reason>"` | record the verdict in the log entry, regenerate the index |
| `just snapshot <dir>` | export a sweep into `results/` |
| `just new-experiment <name> "<title>"` | config + notebook + log entry |
| `just notebook <nb>` / `just notebooks-check` | execute one notebook in place / execute all (CI) |
| `just reproduce <sweep or snapshot dir> [--no-sync]` | rerun a sweep at its tag in a worktree and compare |
| `just figures` / `just refs-check` / `just log-index` | paper figures / references consistency / log index |
| `just status` | where the repo stands, and the next steps with their commands |
| `just sync-code [ref] [force=1]` | fast-forward (or move to `ref`) on a machine with running sweeps, guarded |
| `just github-init` | once per GitHub repo: labels for the issue templates |
| `just push [force=1]` / `just pull <experiment>` | [remote] committed HEAD to the box (guarded) / runs and sweep tags back |
| `just remote-setup` / `just remote <cli args>` / `just remote-ps` | [remote] repo + venv + .env there / run any cli command there / what runs there |
| `just mlflow-sync [paths]` / `just mlflow-ui` | [mlflow] push sinks to MLflow / open the UI |

Project-specific recipes go in `project.just`.

---

## Naming conventions

| Thing | Convention | Example |
|---|---|---|
| Package / modules | `snake_case` nouns | `meta_pfn.analysis.stats` |
| Classes / functions | `PascalCase` / `snake_case` verbs | `ContrastiveLoss`, `build_dataloader` |
| Config groups / files | singular noun / describes the choice | `model/transformer_small.yaml` |
| Experiments | `<question>[_<variant>]`, snake_case | `warmup_variance`, `warmup_variance_robust` |
| Sweep ids | `<stage>_<YYYYMMDD-HHMMSS>` | `pilot_20261021-091500` |
| Metrics | `<split>/<name>`, `best/…`, `final/…` | `val/acc`, `final/eval_mse` |
| Branches | `<type>/<kebab-case>` | `exp/warmup-variance` |
| Tags | `sweep/<experiment>/<id>`, `vX.Y`, `paper-<venue>-<year>` | `sweep/warmup_variance/pilot_20261021-091500` |
| Notebooks / log entries | `NN_YYYY-MM-DD_<experiment>.ipynb` / `YYYY-MM-DD_<experiment>.md` | `03_2026-10-21_warmup_variance.ipynb` |
| Env vars | `META_PFN_<NAME>` | `META_PFN_RUNS_ROOT` |
