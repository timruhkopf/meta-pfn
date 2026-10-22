<!-- Carried over verbatim from ~/PycharmProjects/Meta_FTPFN (branch meta-pfn-task-pair-prior, 71d0430),
docs/labbook/2026-10-15-hpo-meta-task-distortions.md, on 2026-10-22. Reference catalogue of how HPO
landscapes differ across tasks; NOT a spec. Its ρ is a function-space mixture weight with ρ = 1 = fully
shared (the opposite direction of this repo's `rho`, where 0 = same). Constants in it are guesses, not
calibrated (its §2.4). What this repo builds from it is decided per experiment; see issue #3. -->

# Distortions of HPO landscapes across tasks (same search space, different datasets)

*Log, 2026-10-16*

Generative view of how task *t* relates to a reference landscape:

$$f_t(\lambda) = g_t\big(f_{\text{ref}}(\phi_t(\lambda))\big) + \varepsilon_t(\lambda)$$

- **φ_t** – input warp (where the good region sits)
- **g_t** – output warp (scale/range of the metric)
- **ε_t** – noise
- **structural changes** – see Part 4 for what these are and when the formulation can or can't express them

---

# Part 1 – Catalogue of distortions by hyperparameter family

## 1. Optimizer hyperparameters

**Learning rate**
- Main transform: **translation in log-space**. Stable shape: gentle slope on the low side, optimum, sharp divergence cliff on the high side.
- Shift drivers: loss curvature/sharpness, width (largely removable via μP-style reparametrization), batch size, gradient clipping, input scale.
- Asymmetry: the high-LR side is a hard failure boundary whose location is task-dependent.
- Plateau width varies (wide basins on easy tasks, narrow on hard ones), so there is also a **scaling** in log-space, not only a shift.

**Batch size**
- Diagonal ridge with LR (linear/sqrt scaling); the ridge slope is roughly stable, but where it breaks depends on the task.
- Critical batch size (gradient noise scale) is strongly task-dependent: a plateau in quality above it, rising cost.
- Hard truncation: bs ≤ n, plus OOM regions that depend on input size.

**Momentum / Adam β₁, β₂, ε**
- Mostly low importance and flat.
- Effective LR ≈ lr/(1−β) gives another diagonal ridge.
- β₂ and ε matter only in sparse-gradient or unstable regimes, so they are dimensions that **turn on** for some tasks.

**Schedules (warmup, decay type, final LR fraction)**
- Usually low importance; warmup becomes critical at large LR or batch size, or for transformers (conditional importance).
- Interacts with the epoch budget.

**Gradient clipping threshold**
- Irrelevant when gradients are well behaved, critical when they aren't; scales with gradient norm.

## 2. Regularization

**Weight decay / L2 / ridge λ / lasso α**
- Log-space translation driven by n (λ* ∝ 1/n for a fixed-penalty formulation) and by SNR (ridge: λ* ~ σ²/‖w*‖²).
- AdamW: lr × wd sets the effective timescale, giving a ridge whose position depends on epochs.
- Sum vs. mean loss changes how λ* scales with n, a common hidden source of "inconsistency".

**Dropout / drop-path / stochastic depth**
- **Regime change**: unimodal on small or noisy data, monotone decreasing (optimum at 0) on large data.
- Bounded domain, so the shifted optimum gets **clipped to the boundary**.

**Data augmentation**
- **Sign flips** possible: an augmentation helps if the invariance it encodes is valid and hurts otherwise.
- Optimal magnitude shrinks with n and grows with capacity; strongly modality-dependent.

**Label smoothing / mixup / cutmix**
- Optimum depends on label noise and number of classes K (smoothing ε is naturally relative to 1/K).

**Early-stopping patience**
- Matters only for noisy validation curves.

## 3. Capacity / complexity

**Width, depth, hidden size**
- Monotone saturating with large data; unimodal (overfitting) with little data; double-descent bumps possible.
- The cost landscape is task-specific (scales with n, p, sequence length).
- Deep, unnormalized configurations fall into failure regions on some tasks only.

**Tree depth / num_leaves**
- Optimum tracks the **interaction order** of the true function, a property of the data-generating process rather than of n.

**n_estimators**
- Boosting: ridge with shrinkage (η × n_trees ≈ const).
- Random forests: monotone saturating and nearly unimportant past a few hundred trees.

**kNN k, neighbors**
- Optimum scales with n (k* ∝ n^{4/(4+d)}).

**Degree, rank, number of components**
- Integer; often clipped by p or n.

## 4. Data-relative hyperparameters (largely "fake" distortion)

| Hyperparameter | Relative to | Distortion in absolute units |
|---|---|---|
| min_samples_leaf, min_child_weight | n, class balance | log-shift with n |
| max_features, colsample | p, fraction of informative features | large shifts, importance sign changes |
| RBF γ | 1/(p · feature variance) | log-shift |
| SVM C | n, noise, loss scaling | log-shift; ridge with γ |
| batch size | n, gradient noise scale | truncation, plateau shift |
| epochs | n (steps = epochs · n / bs) | step-budget shift |
| k, perplexity, n_neighbors | n | shift |
| subsample | n | matters only for small n |
| PCA components | p, intrinsic dimension | clip + shift |

Reparametrizing relative to n, p, K (and μP for width) removes much of this distortion for free.

## 5. Budget / fidelity

- Epochs/steps: monotone saturating on big data, peaked on small or noisy data; interacts with schedule and weight decay.
- **Fidelity rank correlation is itself task-dependent**: some tasks rank configurations correctly early, others show rank inversions later. This is critical for multi-fidelity transfer.

## 6. Categorical / architectural choices

- Optimizer, activation, kernel, booster, normalization, model family: the effect is a **ranking permutation**, not a continuous warp. Some choices are robustly dominant; others flip; near-ties make rankings look random.
- Conditional subspaces (e.g., γ only for the RBF kernel) switch importance on and off with the categorical's optimum.

## 7. Preprocessing / pipeline

- Imputation, missing indicators: **completely flat** without missing values.
- Scaling: irrelevant for trees, critical for distance- and kernel-based methods.
- Encoding: depends on the number and cardinality of categoricals.
- Feature selection: depends on the fraction of informative features; clipped by p.
- Class weighting / resampling: depends on imbalance *and* metric; can hurt under accuracy.
- Tokenization / sequence length: corpus-dependent; also a cost transform.

## 8. Init / normalization / precision

- Init scale: interacts with depth and activation.
- Norm momentum / epsilon: mostly flat; matter for tiny batches.
- fp16/bf16 and loss scaling: task-dependent failure regions at high LR.

## 9. Output-side distortions (g_t and friends)

- **Affine** scale/offset (error ranges differ wildly).
- **Floor effects** (Bayes error compresses the good region), **ceiling effects** (failure plateaus).
- **Monotone nonlinear warps** (metric choice, skew); rank- or copula-based transfer is robust to these.
- **Heteroscedastic noise**, depending on both configuration and task.
- **Failure / constraint regions** (OOM, timeouts, divergence) that move with n, p, sequence length.
- **Cost landscape**, usually more consistent across tasks than the loss.

## 10. Structural distortions

1. Effective dimensionality / importance changes (stable core: LR, regularization, sometimes capacity; plus a task-specific tail).
2. Interaction structure changes (coupling strong vs. separable).
3. Modality changes (unimodal ↔ monotone; one basin ↔ two).
4. Sign / direction flips.
5. Boundary optima from clipping.
6. Fidelity inconsistency.

## 11. Implications for meta-HPO

- Reparametrize first (log scales; relative to n, p, K; μP).
- Handle output warps with rank/quantile normalization or per-task standardization.
- Handle residual input warps with learned shifts/scales or Beta-CDF input warping conditioned on meta-features.
- Use portfolios for categoricals.
- Allow per-task relevance (task-specific lengthscales / ARD).
- Model feasibility separately.
- Model per-task, heteroscedastic noise.

---

# Part 2 – Compressing this into a generative meta-task prior

Goal: **broad function classes**, not a tailored simulator. Take a single-task prior (here: a BNN prior over depth, width, parameters and relevant dimensions) and add a small number of transform families that capture the dominant *trends* of how one task differs from another.

## 2.0 Core structure

For a meta-task with T tasks on the box λ ∈ [0,1]^d (log-encoded where appropriate):

```
f_shared ~ BNN prior, defined on an EXTENDED box, e.g. [-0.5, 1.5]^d
for each task t:
    h_t(λ) = sqrt(ρ_t) · f_shared(φ_t(λ)) + sqrt(1 − ρ_t) · r_t(m_t ⊙ λ)
    y_t(λ) = g_t(h_t(λ))           (failure mask applied here)
    y_obs  = y_t(λ) + ε_t(λ)
```

Standardize f_shared and r_t (e.g., over a Sobol sample of the box) before mixing, so that ρ_t is meaningful.

**Hierarchical meta-parameters**: sample the meta-level hyperparameters (typical ρ, shift scale, noise scale, etc.) **per meta-task**, then each task's parameters around them. This lets the PFN learn to *infer relatedness in context* instead of assuming a fixed degree of transfer. This is arguably the most important design decision. The prior's breadth controls how much the PFN trusts source tasks.

## 2.1 Transforms, ranked by importance

### (1) Task correlation via residual — the master knob
- `h_t = sqrt(ρ_t) · shared + sqrt(1 − ρ_t) · residual_t`, with r_t drawn from the same BNN prior family (own depth/width/relevance draw).
- This covers, in one scalar, everything a structure-preserving warp can't: new modes, interaction changes, local sign flips, categorical permutations.
- Prior: meta-level μ_ρ ~ Beta(2, 1.5) (mostly related tasks, but not always); per task ρ_t ~ Beta around μ_ρ with concentration ~20. Include a small mass (~5–10%) of essentially unrelated tasks (ρ ≈ 0), so the PFN learns to ignore misleading sources.
- Optional: give the residual a shorter lengthscale (smaller amplitude, higher frequency); real task differences tend to be smoother at large scales than at small ones, though not always.

### (2) Per-dimension shift (log-space translation)
- φ_t,i(λ) = c + (w_i(λ_i) − c) / a_t,i + δ_t,i, with c = 0.5.
- δ_t,i: heavy-tailed, δ ~ s_i · StudentT(ν = 4), with per-dimension scale s_i ~ LogUniform(0.02, 0.3) in box units (sampled per meta-task: some dims shift a lot, like LR and WD; others barely move).
- **Shifts must be allowed to push the optimum outside the box**, which is why f_shared lives on an extended domain. This alone generates boundary optima and "unimodal ↔ monotone" regime changes inside the box.

### (3) Coherent shifts via a low-dimensional task latent
- Real shifts are correlated: dataset size, noise level and difficulty move many hyperparameters at once (WD, dropout, min_leaf, epochs all shift with n).
- δ_t = B z_t + δ_t^indep, with z_t ~ N(0, I_k), k ∈ {1, 2}, and B sparse (each dim loads on a latent with probability ~0.5, loading ~ N(0, s_i²)).
- The same z_t can optionally also modulate noise level (2.3) and floor (2.2). This captures "small data ⇒ more regularization, more noise, more overfitting" without hand-tailoring.

### (4) Per-dimension basin width / monotone warp
- Scale: a_t,i ~ LogNormal(0, 0.25²) — basins widen or narrow.
- Monotone warp w_i: Kumaraswamy CDF w(x) = 1 − (1 − x^α)^β with log α, log β ~ N(0, 0.3²) (identity at α = β = 1). This gives asymmetric stretching (gentle on one side, compressed on the other).
- Keep these mild; φ must stay near-identity on average or transfer gets destroyed.

### (5) Relevance switching
- Per task and dimension, with probability p_off ~ Beta(1, 9) (meta-level), set the dimension to a constant u_i in both shared and residual input (m_t,i = 0). The dimension becomes irrelevant for that task.
- "Turning on" comes for free from the residual's independent relevance draw.
- Soft version: a_t,i ≫ 1 (very wide basin) rather than exact collapse.

### (6) Output transform g_t
- Affine: g(h) = a_t · w(h) + b_t, with log a_t ~ N(0, 1) and b_t ~ N(0, 1). If the PFN standardizes y per task this mostly matters for noise ratios, but sample it anyway so the model doesn't rely on absolute values across tasks.
- Monotone warp w: u = sigmoid(h / τ), then a Kumaraswamy CDF with log α, log β ~ N(0, 0.5²); with probability 0.5 use the identity. This produces floor and ceiling saturation (compressed good region, flat bad region).
- Keep g strictly increasing; flips belong to the residual.

### (7) Failure regions
- Fail where f_shared(φ_t(λ)) + small noise exceeds the task's (1 − q_t) quantile, with q_t ~ mixture(0 with prob 0.6, Beta(1, 6) otherwise).
- Because the threshold is applied after φ_t, the failure region automatically moves with the task's shift (the "LR cliff moves with the LR optimum" effect).
- Represent failures as a special token or as the worst observed value; don't let them pass through as Gaussian observations.

### (8) Low priority (optional)
- **Small rotations/shears** (λ' = λ + η A λ with η small): ridge slope changes. Mostly covered by the residual.
- **Reflections**: rare; prefer the residual.
- **Categoricals**: per-category additive offsets in the residual, c_t,k ~ N(0, τ²), with τ inversely related to ρ_t. This yields Mallows-like near-identity permutations without explicit permutation sampling.

## 2.2 What this covers

| Real-world trend | Mechanism |
|---|---|
| Log-space optimum shift (LR, WD, C, γ, …) | (2), (3) |
| Data-size-driven coherent shifts | (3) |
| Boundary optima, unimodal ↔ monotone | (2) + extended domain |
| Basin width, plateau asymmetry | (4) |
| Dimensions turning off / on | (5) / residual |
| Interaction changes, new modes, sign flips | (1) residual |
| Categorical ranking permutations | (1) + category offsets |
| Metric scale, floor/ceiling | (6) |
| Moving failure cliffs | (7) |
| Noise level / heteroscedasticity | Part 3 |

Not covered well: fidelity-specific transforms (they need a fidelity axis in the prior), and cost landscapes.

## 2.3 Sampler sketch

```python
import numpy as np

def kumaraswamy(x, a, b):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return 1 - (1 - x**a)**b

def sample_meta_task(d, T, rng, sample_bnn):
    # sample_bnn(d, domain) -> standardized callable f(X: [N, d]) -> [N]
    f_shared = sample_bnn(d, domain=(-0.5, 1.5))

    # ---- meta-level (per meta-task) ----
    mu_rho  = rng.beta(2.0, 1.5)
    s       = np.exp(rng.uniform(np.log(0.02), np.log(0.3), size=d))  # shift scale per dim
    k       = rng.integers(1, 3)
    B       = rng.normal(0, s[:, None], size=(d, k)) * (rng.random((d, k)) < 0.5)
    p_off   = rng.beta(1, 9)
    log_sig = rng.uniform(np.log(0.01), np.log(0.3))                  # meta noise level

    tasks = []
    for t in range(T):
        unrelated = rng.random() < 0.07
        rho = 0.0 if unrelated else rng.beta(20 * mu_rho + 1e-3, 20 * (1 - mu_rho) + 1e-3)

        z      = rng.normal(size=k)
        delta  = B @ z + s * 0.5 * rng.standard_t(4, size=d)
        a      = np.exp(rng.normal(0, 0.25, size=d))
        ka, kb = np.exp(rng.normal(0, 0.3, size=(2, d)))
        on     = rng.random(d) >= p_off
        u      = rng.random(d)
        r_t    = sample_bnn(d, domain=(0, 1))

        def phi(X):
            W = kumaraswamy(X, ka, kb)
            return 0.5 + (W - 0.5) / a + delta

        def mask(X):
            return np.where(on, X, u)

        # output warp
        ga, gb = np.exp(rng.normal(0, 0.5, size=2))
        use_w  = rng.random() < 0.5
        scale, offset = np.exp(rng.normal()), rng.normal()

        # noise (Part 3)
        sig_t  = np.exp(log_sig + 0.5 * z[0] + rng.normal(0, 0.5))
        beta_f = rng.normal(0.5, 0.5)
        h_sig  = sample_bnn(d, domain=(0, 1))
        nu     = rng.choice([3, 5, 1e9])

        def f_task(X, rho=rho, phi=phi, mask=mask, r_t=r_t, ga=ga, gb=gb,
                   use_w=use_w, scale=scale, offset=offset,
                   sig_t=sig_t, beta_f=beta_f, h_sig=h_sig, nu=nu):
            h = np.sqrt(rho) * f_shared(phi(mask(X))) + np.sqrt(1 - rho) * r_t(mask(X))
            y = h
            if use_w:
                y = kumaraswamy(1 / (1 + np.exp(-h)), ga, gb)
                y = (y - y.mean()) / (y.std() + 1e-9)      # crude re-standardization
            y = scale * y + offset
            log_s = np.log(sig_t) + beta_f * h + 0.3 * h_sig(X)
            eps = np.exp(log_s) * rng.standard_t(nu, size=len(X)) / np.sqrt(nu / (nu - 2) if nu > 2 else 1)
            out = rng.random(len(X)) < 0.02
            eps = np.where(out, eps + scale * rng.normal(0, 3, size=len(X)), eps)
            return y + eps, y

        tasks.append(f_task)
    return tasks
```

(Failure masks are omitted here for brevity; see 2.1 (7). Constants are starting guesses.)

## 2.4 Calibration

The numbers above are priors-of-convenience. To ground them: fit the transform family between task pairs on real tabular benchmarks (HPO-B, YAHPO Gym, LCBench, PD1). Specifically, fit per-dimension shifts and scales, a residual ρ, a monotone g, and a noise level per task pair, then use the empirical distributions of fitted parameters (inflated somewhat) as the meta-level priors. Check the generated meta-tasks against held-out real ones with simple statistics: rank correlation between tasks, distance between optima, fraction of boundary optima, importance overlap.

---

# Part 3 – Noise model (replacing constant Gaussian noise)

A constant small-variance Normal misses four things, listed by importance:

1. **Task-level noise scale varies by orders of magnitude** (validation set size, seed variance, model stability).
   σ_t relative to signal std ~ LogUniform(0.01, 0.3) at the meta level, with per-task jitter. Optionally coupled to the task latent z (small data ⇒ more noise).
2. **Heteroscedasticity in λ**:
   log σ_t(λ) = log σ_t + β · h(λ) + γ · s(λ), with h the standardized task function and s a smooth random function (small BNN/GP, amplitude ~0.3).
   - β > 0 most of the time (bad / near-divergence configurations are noisier). Sample β ~ N(0.5, 0.5), allowing negative values.
   - For bounded metrics near a floor, noise *shrinks* near the optimum, which is the same mechanism with the opposite sign of β.
3. **Heavy tails / outliers**: Student-t with ν ∈ {3, 5, ∞}, plus an outlier mixture (probability ~1–3%) of large deviations (seed blow-ups, partial divergence).
4. **Bounded-metric noise** (optional, physically motivated): for error rates p ∈ [0,1], use y_obs ~ Binomial(n_val, p)/n_val plus a seed term, with n_val ~ LogUniform(10², 10⁵). This gives variance ∝ p(1−p)/n_val, i.e., heteroscedasticity and task-level scale from a single meta-feature.

The PFN then has to infer the noise level from the observations, which matters for how sharply it should trust small differences.

---

# Part 4 – Why "structural changes" escape φ and g (and when they don't)

**Unrestricted, the formulation is vacuous.** If φ_t can be any map and f_ref is rich enough, then f_ref ∘ φ_t can represent essentially any function: pick φ_t(λ) as a point where f_ref takes the desired value. So "can be expressed" is only meaningful relative to a **restricted transform family**, and choosing that family *is* the definition of what is shared across tasks.

**With the natural restrictions** (φ_t a diffeomorphism such as a shift, scale or monotone per-dimension warp; g_t strictly increasing), the composition preserves invariants:
- the number and type of critical points (Morse structure), so **modality is preserved**;
- the ordering of function values among those critical points (g is monotone), so which local optimum is best is preserved;
- the topology of level sets;
- dependence on a dimension (a diffeomorphism cannot make a relevant dimension irrelevant);
- separability, when φ is per-dimension.

Structural changes are exactly the changes in these invariants, so they need transforms outside the family:

| Structural change | Requires |
|---|---|
| Dimension turns off | non-injective φ (collapse/projection) |
| Dimension turns on | dependence absent from f_ref → residual |
| New / merged modes | folding φ, non-monotone g, or residual |
| Sign flip | reflection in φ (non-monotone) or residual |
| Interaction appears / disappears | non-axis-aligned or nonlinear coupling φ, or residual |
| Failure regions, heteroscedastic noise | depend on λ directly, not only through f, so they are not of the form g(f(φ(λ))) at all |

One subtle exception: **boundary optima and "unimodal ↔ monotone" changes *are* expressible with a pure shift**, because the domain is bounded. The invariants hold on the extended domain, but the box shows a different window. Hence the extended domain for f_shared in Part 2.

**For a PFN, identifiability doesn't matter.** You only need the prior predictive distribution to be realistic. The restricted family matters for a different reason: it determines **how much information transfers**. A structure-preserving warp transfers a lot (knowing the shape from source tasks constrains the target strongly); a residual transfers nothing. Hence the decomposition in Part 2: structure-preserving warps for the dominant trends, plus a residual with a learned-in-context weight ρ for everything else.
