# meta-pfn

<!-- Project-facing README: this is what a reviewer opens from your CV (~60 s).
     The research process lives in docs/process.md (owned by the template, updated by `copier update`). -->

[![CI](https://github.com/timruhkopf/meta-pfn/actions/workflows/ci.yml/badge.svg)](https://github.com/timruhkopf/meta-pfn/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Can a PFN transfer knowledge from a previous optimization run to a new, related one, in context,
> when the two landscapes live in differently warped coordinates?** Before building that model, this
> repo measures how much such transfer could be worth at all.

**Status:** Phase 0 (is the information there?) is running. See [Status](#status). **How we work:**
[docs/process.md](docs/process.md) · **Lab book and decisions:** [research_log/](research_log/README.md) ·
**Literature:** [references.md](references.md)

## The idea

Hyperparameter optimization rarely starts from scratch: the same pipeline was tuned yesterday on a
related dataset. That run's landscape B is *related* to today's landscape A, but not equal to it:
- optima shift with dataset size;
- regularization strength stretches with noise;
- sensitive ranges widen or narrow.

In the search space this is a **non-uniform warp of the inputs** (often per axis), plus some genuine
difference between the functions. Classic transfer methods either assume the landscapes coincide
(multi-task GPs, warm starts) or weight whole tasks by similarity (RGPE). Neither registers one
landscape onto the other.

A **prior-data fitted network (PFN)** learns Bayesian inference for whatever prior it is trained on.
That makes the transfer a *prior design* problem:
1. sample pairs (A, B) with known warps and relatedness;
2. train a **Meta-PFN** to predict A's points from A's few observations plus B's many;
3. supervise its internal registration with the known warp (privileged information, available only
   during training).

The hoped-for benefits:

- **Few-shot BO that uses the past run properly.** With 2–8 points on A, B's 32+ points can carry most of
  the information, if the model can place them correctly.
- **No hand-designed similarity measure.** The model infers in context whether, and how, B is related,
  per query.
- **Safe by construction.** If B is unrelated, the Bayes answer is to ignore it. A Meta-PFN trained on a
  prior that contains unrelated pairs learns to, which avoids negative transfer.
- **Amortized registration.** One forward pass, instead of fitting a warp per task pair.

**The catch** (what the predecessor ran into): a model can look good by detecting copies (B = A), or
by beating a *different* baseline model. Whether B helps on genuinely *related* pairs depends on how
much information the prior puts there. This must be measured before any training.

## What is measured, and how (Phase 0, E0)

[Notebook 01](notebooks/01_2026-10-21_headroom.ipynb) asks how much a **Bayes learner** gains from B,
using the frozen `pfn_large` as both the predictive and the likelihood:

| | |
|---|---|
| **Pairs** | one BNN-prior function, seen by B directly and by A through a per-axis warp into a sub-window (identity, affine window, Kumaraswamy warp), with A's weights perturbed by σ (Spearman ≈ 1 / 0.9 / 0.7), plus unrelated pairs |
| **lower** | PFN(A): A alone |
| **naive** | PFN(B ∪ A): pooling without registration |
| **oracle_T** | PFN(B ∪ T(A)): the warp known |
| **bayes** | a posterior over {unrelated, identity, window and warp hypotheses}, each weighted by the PFN's chain-rule likelihood of A's points given B: the warp inferred |
| **Decision** | KILL if no related cell shows ≥ 0.05 nats/query; GO if, at the *least related* level, the gain is > 0.10 nats with ≥ 25% of the known-warp headroom surviving; fixed before the run |

A GO means a Meta-PFN on this prior family has something to learn, and the size of the gain is the
target it must approach. A KILL means no model can help, which saves the training.

## Status

| Phase | Question | Verdict | Evidence |
|---|---|---|---|
| 0 | E0: is B worth anything to a Bayes learner on related pairs? | _running_ | [notebook 01](notebooks/01_2026-10-21_headroom.ipynb) |
| 1 | E1: does a Meta-PFN (with privileged registration targets) realize the E0 gain? | only if E0 is GO | [issue #2](https://github.com/timruhkopf/meta-pfn/issues/2) |

## Quickstart

```bash
git clone https://github.com/timruhkopf/meta-pfn.git && cd meta-pfn
just setup cpu local     # uv sync --extra cpu (pulls pfn-repro at its pinned tag), hooks, .env
just test                # runs on the CPU with the small PFN bundled in pfn-repro
uv run meta_pfn experiment=headroom task.pfn=pfn_smoke task.n_pairs=3 task.k=16 'task.k_by_d={}'
```

The reported runs use `pfn_large` (76 MB, not in git; see below). Reproduce a reported result with
`just reproduce results/<experiment>/<sweep_id>`. Requirements: [uv](https://docs.astral.sh/uv/),
[just](https://github.com/casey/just). Always pass a torch extra (`cpu` or `cu126`).

## The PFN dependency: pfn-repro

The BNN prior, the PFNs, the bar distribution and the checkpoints come from
[pfn-repro](https://github.com/timruhkopf/pfn-repro): frozen code, checkpoints pinned by sha256, golden-output
tests. This repo never copies its code. Its `docs/dependency_guide.md` weighs the options.

- **Default (every tracked run): a git tag.** `pyproject.toml` declares
  `pfn-repro = { git = "https://github.com/timruhkopf/pfn-repro", tag = "v0.1.0" }`, and `uv.lock` pins the
  commit. A commit of this repo pins pfn-repro too.
- **Developing both at once:** on a branch, `uv add --editable ../pfn-repro`. Tracked runs record its
  commit and refuse uncommitted changes in it, and `confirm` / `final` refuse it altogether. Before
  reporting, release a pfn-repro tag and switch back with
  `uv add "pfn-repro @ git+https://github.com/timruhkopf/pfn-repro" --tag vX.Y.Z`.
- **Checkpoints** are not in the package (except the smoke PFN and the ECDF). Set `PFN_REPRO_CKPT_ROOT` in
  `.env` to the directory holding `pfn_large.pt`. `python -c "from pfn_repro.checkpoints import available; print(available())"`
  shows what is found.
- **A GPU box without GitHub credentials:** point git at a local clone there,
  `git config --global url."/home/<user>/PycharmProjects/pfn-repro".insteadOf https://github.com/timruhkopf/pfn-repro`,
  and push the tag to that clone.

**PyCharm.** Open this repo with `.venv` as its interpreter, and attach pfn-repro (*File → Open →
pfn-repro → Attach*) for navigation and search across both. The interpreter keeps importing the *locked*
version from `.venv`, so "go to definition" lands in `site-packages/pfn_repro` unless the editable
install is active. Edit pfn-repro in its own attached project and release a tag. Mark `src/` as
*Sources Root*.

## Repository map

| Path | What |
|---|---|
| `src/meta_pfn/` | the package; one entry point `meta_pfn` (Hydra), configs ship inside it |
| `src/meta_pfn/pairs.py` | the task-pair prior on pfn-repro's BNN prior: warps, relatedness, strata |
| `src/meta_pfn/oracle.py` | the four predictors with the frozen PFN, incl. the chain-rule Bayes learner |
| `src/meta_pfn/analysis/headroom.py` | gains with CIs, oracle precision, the pre-registered E0 rule |
| `src/meta_pfn/configs/experiment/` | one file per research question (config only) |
| `research_log/` | lab book: entries with decisions, append-only track record, design decisions (backlog: GitHub issues) |
| `results/` | committed snapshots of finished sweeps; notebooks and figures read only these |
| `notebooks/` | executed analyses, outputs committed |
| `references.md` / `.bib` | annotated literature, keyed to BibTeX |

## Citation

```bibtex
@misc{ruhkopf2026meta_pfn,
  title  = {...},
  author = {...},
  year   = {2026},
  url    = {https://github.com/timruhkopf/meta-pfn}
}
```

## License and acknowledgements

MIT, see [LICENSE](LICENSE). Datasets are subject to their own licenses (`data/README.md`).
Funding, compute resources, and code this project builds on.
