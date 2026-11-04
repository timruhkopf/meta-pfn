# Instructions for Claude Code: AIR-CIS (unlearned and learned correspondence proposal)

**Goal.** Implement, test and evaluate the proposal in `2026-11-04_air_cis.md` (attached; the spec and pre-registration). It replaces AIR's flow proposal by correspondence importance sampling (CIS), in two versions: unlearned (U) and learned (L).

## 0. Ground rules

- Follow repo conventions: run logs with frontmatter; pre-registration committed before code; sweep IDs `pilot_YYYYMMDD-HHMMSS`; snapshots in `results/air/`; numbered notebooks.
- One commit per step. No amend or force-push.
- **Stop rule:** a failing correctness test stops the work. Report it; never sweep on untested code.
- Record every departure from the spec in a "Deviations" section of the log, with the reason. Never edit the spec's definitions or hypotheses after the pre-registration commit.
- **Standing decision:** every d ≥ 2 pre-registration states the intrinsic dimension k, the separable/non-separable split, the coefficients per axis, and a mis-specification check.

## 1. Store the log

Copy `2026-11-04_air_cis.md` verbatim next to the other AIR logs, then commit. That commit is the pre-registration.

## 2. Prior and decoder (stage S0)

**`air/prior.py`:**
- BNN g with sampled architecture: depth {1, 2, 3}, width {16, 32, 64}, as in spec §1.
- Recompute the ECDF map, pooled over the architecture prior (fixed seed). Store it in `results/air/ecdf_bnnarch.pt`.
- The hierarchical transform prior: $J\in\{0,2,4,8,16\}$ with p(J) = (0.2, 0.3, 0.2, 0.15, 0.15), $\log s_T$, $J=0$ = identity. Provide `log_prior(J, log_sT, theta)`.
- Affine h (H1).
- `branch_label(g_dense, u)` (evaluation only).
- Precompute the prior tables (spec §3.2): $\pi(u\mid x)$ (256 × 256) and $s_K(\Delta x)$ (32 bins), from $10^5$ draws, into `results/air/prior_tables_{config}.pt`.

**`air/decoder.py`:**
- Add the moved-B feature $\log (T^{-1})'(x_b)$ (an extra Fourier-featured input, summed into the B token embedding).
- Allow $n_A=0$.
- Retrain under the new prior: H0 first, then H1. Gradient clip 20, train to a plateau (validation NLL improves by < 0.01 over 5k steps).
- Also train a variant **without** the T′ feature (ablation C8).

**Tests:**
- prior samples have the specified p(J) frequencies (χ² test);
- the ECDF output is marginally uniform (KS test);
- $J=0$ gives exactly the identity;
- the prior tables are normalised;
- the decoder at $n_A=0$ returns the B-only predictive with no error.

## 3. Unlearned CIS (stages S1–S3)

Create a new package `air/cis/`:

| File | Contents |
|---|---|
| `bsurrogate.py` | per-task GP on B (spec §3.1): `mu(u)`, `var(u)`, `dmu(u)`; hyperparameters chosen by marginal likelihood over the fixed lengthscale set |
| `claims.py` | candidate sets (top-M ∪ prior neighbourhood ∪ local) and emissions for H0 and H1 (round 1 h-marginal on a 16 × 16 grid; rounds ≥ 2 plug-in) |
| `ffbs.py` | log-space forward filtering, backward sampling over variable-size candidate sets; returns samples and exact `log_q(c)`; also `log_q_of_path(path)` (used by L training) |
| `fit.py` | batched ridge initialisation → damped Gauss–Newton (5 iterations) for each (assignment, J); Laplace θ-block covariance; Laplace evidence; `q(J\|c)` with a 0.02 floor; autograd through the RK4 flow |
| `proposal.py` | sampling $c\to J\to\theta$ with the defensive mixture (ε = 0.1); `log_q_total`; auxiliary `log_r(c\|theta)` |
| `rounds.py` | R-round adaptive importance sampling; surrogate weights in rounds < R; the U update rule (spec §3.7); the last round with decoder weights |
| `estimator.py` | final weights (spec §3.6), predictive mixture, $\log\hat Z_1$, ESS; logs runtime per stage |

**Correctness tests** (`tests/test_cis.py`), which must pass before any sweep:
1. **FFBS vs brute force:** on random tiny problems ($n_A=4$, 3 candidates each), enumerate all paths. The sampled path frequencies match the exact path probabilities (χ² p > 0.01, $10^5$ samples). `log_q_of_path` equals the exact log probability within $10^{-6}$. The forward total equals the enumerated sum.
2. **Soft monotonicity:** every path has $\log q>-\infty$.
3. **Gauss–Newton:** on noise-free tasks with the true assignment and $n_A\ge J+1$, the fit recovers $\theta_{\text{true}}$ with $|\Delta|<10^{-3}$, for J = 2, 4, 8.
4. **Laplace evidence vs grid evidence** for J = 2, H0, dense-B toy: within 0.2 nats, on 100 tasks.
5. **Auxiliary importance sampling is consistent:** on the J = 2 dense-B toy with an exact likelihood in place of the decoder, the CIS predictive at $K=4096$ matches the grid PPD with KL < 0.02.
6. **Weights are finite and bounded** on 1000 random tasks; no NaNs.
7. **GP derivative** matches finite differences.

**Sweeps:**
- **S1:** U with R = 1, H0.
- **S2:** U with R = 3, H0.
- **S3:** U with R = 1 and R = 3, H1.
- Each with K ∈ {32, 128}, $n_A$ ∈ {1, 2, 3, 5, 8, 10}, 256 tasks per cell, stratified by the number of extrema of g.
- Same evaluation tasks for all, saved as `results/air/eval_tasks_cis_{H}.pt`.
- Also run AIR-flow (retrained, with the J head), IS-ref (K = 8192), G-ref (J = 2 stratum, H0), teacher and A-only, on the same tasks.

## 4. Learned operator (stage S4)

`air/cis/operator.py`: $O_\phi$, as in spec §3.8.
- **State builder:** per-point weighted mean, standard deviation and top-3 KDE modes of $T_{\theta_k}(x_a)$; the weighted residual; candidate tokens; global token (h estimate and spread, ESS/K, weighted p(J), round index).
- **Network:** 4 layers, width 128; A→candidate cross-attention; bilinear claim-logit head; transition-width head per edge; h-emission head; J-logit correction head.
- Weights shared across rounds.

`air/train_operator.py`:
- **Loss:** $\sum_{r=2}^{R}\big[-\log q^{(r)}(c^{\text{true},(r)})-\log q^{(r)}(J^{\text{true}}\mid c^{\text{true},(r)})\big]$, with $c^{\text{true}}$ = the nearest candidate to $T(x_a)$ in each round's set.
- Stop gradients across rounds. Curriculum: the first 30 % of steps use U states for round 1.
- $K_r=16$; 60k steps, batch 32; AdamW $3\cdot10^{-4}$, clip 20; 3 seeds; H0 and H1.

**Tests:**
- `log_q_of_path` gradients match finite differences.
- With $O_\phi$ outputs set to zero, L reproduces U exactly: same samples for the same seed.
- **Validity guard:** with random $O_\phi$ outputs, the auxiliary-importance-sampling consistency test 5 still passes at K = 4096.

**Sweep S5:** L with R = 3 against U with R = 3, H0 and H1, on the same grid as S1–S3.

## 5. Analysis, notebook, log

Notebook `notebooks/13_2026-11-04_air_cis.ipynb`:
1. ESS/K against $n_A$ and K, all proposals.
2. Branch hit rate and q-mass on the truth, by extrema stratum.
3. NLL gap to IS-ref and to G-ref.
4. SBC histograms (θ, J); calibration of p(J).
5. Rounds: ESS per round, U against L.
6. H1 against H0.
7. Runtime breakdown per stage.
8. The T′-feature ablation (evidence KL).

Export the report to `reports/figures/air/13_air_cis_report.html`.

Fill the log's Results section (the hypothesis table C1–C8 with values, standard errors and pass/fail), Interpretation (≤ 10 bullets, each tied to a number), Deviations, and the sweep IDs; set `status: done`. Append a summary entry to the AIR run log.

## 6. Report back

After S3, and again after S5, send a short summary:
- the hypothesis table;
- the headline numbers: branch hit rate on the ≥ 3-extrema stratum, and ESS/K of U against AIR-flow at $n_A=3$;
- any test that needed a fallback;
- paths.

## 7. Order and budget

S0 → S1 → S2 → S3 → S4/S5. Stop after S1 if C1 fails, and diagnose candidate pruning first: how often was the true branch's candidate missing from $\mathcal C_a$? d = 2 is a separate pre-registration later (spec §7).
