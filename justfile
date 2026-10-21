# Template-owned: changes come from `copier update`. Project-specific recipes go in project.just.
# Recipes are thin wrappers: `just sweep ...` == `python -m meta_pfn.tracking.cli sweep ...`.
set dotenv-load
set positional-arguments
set shell := ["bash", "-cu"]  # recipes use bash features (printf %q); sh may be dash
import? 'project.just'

extra := env("META_PFN_TORCH_EXTRA", "cpu")
py := "uv run --extra " + extra + " python"
cli := py + " -m meta_pfn.tracking.cli"
remote_host := env("META_PFN_REMOTE_HOST", "")
remote_dir := env("META_PFN_REMOTE_DIR", "")

default:
    @just --list --unsorted

# uv sync with a torch extra (cpu | cu126), git hooks, .env from envs/<setup>.env.example
setup extra setup="local":
    uv sync --extra {{extra}} {{ if setup == "local" { "--group notebooks" } else { "" } }}
    [ -f .env ] || cp envs/{{setup}}.env.example .env
    grep -q '^META_PFN_TORCH_EXTRA=' .env || echo 'META_PFN_TORCH_EXTRA={{extra}}' >> .env
    uv run --extra {{extra}} pre-commit install --hook-type pre-commit --hook-type pre-push --hook-type post-merge
    git config push.followTags true

# switch the active machine settings (.env) to envs/<setup>.env.example
use-env setup:
    cp envs/{{setup}}.env.example .env
    echo 'META_PFN_TORCH_EXTRA={{extra}}' >> .env

# where the repo stands and what to do next (also shown to Claude at session start)
status:
    {{cli}} status

# update this checkout (default: fast-forward to upstream), guarding running sweeps: just sync-code [ref] [force=1]
sync-code ref="@{u}" force="":
    git fetch --quiet
    {{cli}} sync-code {{ref}} {{ if force != "" { "--force" } else { "" } }}

# once per GitHub repo: labels used by the issue templates
github-init:
    -gh label create idea --color FBCA04 --description "an idea, not yet an experiment"
    -gh label create experiment --color 0E8A16 --description "a question with a GO/NO-GO criterion"
    -gh label create infra --color 1D76DB --description "tooling, pipelines, refactors"
    -gh label create bug --color D73A4A --description "something is wrong" --force

# tiny end-to-end scratch run
smoke:
    {{py}} -m meta_pfn.main experiment=debug

# fast tests / everything (slow, integration)
test *args:
    {{py}} -m pytest "$@"
test-all *args:
    {{py}} -m pytest -m "" "$@"

lint:
    uv run --extra {{extra}} ruff check .
    uv run --extra {{extra}} ruff format --check .
    uv run --extra {{extra}} pyright

fmt:
    uv run --extra {{extra}} ruff format .
    uv run --extra {{extra}} ruff check --fix .

# one tracked run: just run <local|detached|slurm> <stage> experiment=<name> [overrides]
run *args:
    {{cli}} run "$@"

# a tracked sweep: just sweep <local|detached|slurm> <stage> experiment=<name> seed=0,1,2 ...
sweep *args:
    {{cli}} sweep "$@"

sweep-status dir:
    {{cli}} sweep-status "$@"

# re-execute members that are not DONE (check sweep-status first: STARTED may still be running)
sweep-resume *args:
    {{cli}} sweep-resume "$@"

# tidy table of a sweep: varying params, summary metrics, status
runs dir:
    {{cli}} runs "$@"

# record a verdict in the experiment's log entry: just decide <sweep dir> <GO|NO_GO|INCONCLUSIVE> "<reason>"
decide dir verdict reason:
    {{cli}} decide "$@"

# export a sweep into results/ for notebooks and figures
snapshot dir:
    {{cli}} snapshot "$@"

# rerun a sweep at its commit in a git worktree and compare summaries
reproduce *args:
    {{cli}} reproduce "$@"

# new experiment: config + notebook (description first) + log entry
new-experiment name title:
    {{cli}} new-experiment "$@"

log-index:
    {{cli}} log-index

refs-check:
    {{cli}} refs-check

# execute every notebook against its snapshot (outputs discarded; CI)
notebooks-check:
    #!/usr/bin/env bash
    set -euo pipefail
    out=$(mktemp -d)
    shopt -s nullglob
    for nb in notebooks/*.ipynb; do
        echo "executing $nb"
        uv run --extra {{extra}} --group notebooks jupyter nbconvert --to notebook --execute --output-dir "$out" "$nb"
    done

# execute a notebook in place, keeping outputs (then commit it)
notebook nb:
    uv run --extra {{extra}} --group notebooks jupyter nbconvert --to notebook --execute --inplace "$@"

# regenerate reports/figures/ (each scripts/figures/*.py reads final-stage snapshots)
figures:
    #!/usr/bin/env bash
    set -euo pipefail
    shopt -s nullglob
    for f in scripts/figures/*.py; do echo "$f"; {{py}} "$f"; done

# --- remote box (META_PFN_REMOTE_HOST / _REMOTE_DIR in .env) ---------------------------
# Code moves by git over ssh (no GitHub credentials needed on the box; the pre-push test hook is for
# GitHub pushes and is skipped here); sweeps there run from frozen
# worktrees, so a push never breaks them. Only dependency changes under active sweeps need force=1.

# put the committed HEAD on the box and check it out there: just push [force=1]
push force="":
    git -c push.default=current push --quiet --no-verify --force {{remote_host}}:{{remote_dir}} HEAD:refs/heads/incoming
    ssh {{remote_host}} "cd {{remote_dir}} && .venv/bin/python -m meta_pfn.tracking.cli sync-code incoming {{ if force != "" { "--force" } else { "" } }}"

# runs of one experiment back, plus the sweep tags created on the box: just pull <experiment>
pull experiment:
    mkdir -p runs/{{experiment}}
    rsync -az --exclude .code {{remote_host}}:{{remote_dir}}/runs/{{experiment}}/ runs/{{experiment}}/
    git fetch --quiet {{remote_host}}:{{remote_dir}} 'refs/tags/sweep/*:refs/tags/sweep/*'
    @echo "back up the sweep tags: git push --tags"

# first time / dependency changes: repo, venv and .env on the box (extra defaults to cu126)
remote-setup extra="cu126":
    ssh {{remote_host}} 'mkdir -p {{remote_dir}} && cd {{remote_dir}} && ([ -d .git ] || git init -q)'
    git -c push.default=current push --quiet --no-verify --force {{remote_host}}:{{remote_dir}} HEAD:refs/heads/incoming
    ssh {{remote_host}} 'cd {{remote_dir}} && git checkout -q --detach incoming && PATH=$HOME/.local/bin:$PATH uv sync --extra {{extra}} && ([ -f .env ] || cp envs/remote.env.example .env) && (grep -q TORCH_EXTRA .env || echo META_PFN_TORCH_EXTRA={{extra}} >> .env)'

# run a cli command on the remote, e.g. just remote sweep detached pilot experiment=x seed=0,1,2
remote *args:
    ssh {{remote_host}} "cd {{remote_dir}} && .venv/bin/python -m meta_pfn.tracking.cli $(printf '%q ' "$@")"

# what is running on the remote (verify a launch before relaunching)
remote-ps:
    ssh {{remote_host}} 'ps -eo pid,etime,args | grep -E "[t]racking[.]cli|meta_pfn[.]main" | grep -v "grep -E" || echo "nothing running"'

# --- MLflow viewer ------------------------------------------------------------------------------

# push run sinks to MLflow (one way, idempotent); default: all of runs/
mlflow-sync *paths:
    {{cli}} mlflow-sync "$@"

mlflow-ui:
    uv run --extra {{extra}} mlflow ui --backend-store-uri "${META_PFN_MLFLOW_URI:-sqlite:///$PWD/mlflow.db}"
