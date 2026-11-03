# AIR: reading A and B for T — proposal v2

Nov 3, 2026 · @Tim

## How the flow produces θ, in plain terms

The flow can only shape what the encoder's summary c tells it; if c is poor, the best flow still proposes the wrong region. There are two stages.

**Stage 1 — the encoder writes a fingerprint, c.**

- Every point of A and B becomes a token: location x, value y, from A or from B.
- Tokens talk within their own task first, then A's tokens look at B's. The hope: an A token learns "values like mine occur in B over there".
- Four learned readers each skim all A tokens and write 64 numbers. Concatenated, that is c: 256 numbers meant to describe everything (A, B) says about T.

**Stage 2 — the flow turns noise into transforms, shaped by c.**

- Start with J random numbers z from a standard bell curve. This is the dice roll that makes each sample different.
- Pass them through 5 layers. Each bends one coordinate at a time with a smooth increasing curve (a spline), like a wire kinked in about 10 places without folding back.
- A small network that sees c and the coordinates already processed decides the kinks. Steep stretches spread nearby z apart (density thins); flat ones squeeze them together (density piles up).
- After 5 layers the bell curve has the posterior's shape — ridges, several lumps. The output is θ.
- Because every step is a known bend, the flow also reports how likely any θ is, from the accumulated stretching. Importance sampling needs that.

In short: c decides what the posterior should look like, the noise decides which sample you get, and the spline layers are the clay modelling that turns a round blob into that shape.

## Why the summary is the weak spot

With dense B (g known), the set of per-point evidence maps is sufficient for θ; a fixed 256-number pooled vector is not.

```latex
p(A\mid\theta)=\prod_a \mathcal N\big(y_a;\,g(T_\theta(x_a)),\sigma^2\big)=\prod_a C_a\big(T_\theta(x_a)\big),\qquad C_a(u):=\mathcal N\big(y_a;\,g(u),\sigma^2\big)
```

Each C\_a(u) is a map over B's domain: how well A's point a would fit if it sat at location u in B. It peaks on the level set of y\_a and shows every branch. Together with the x\_a, the set of pairs (x\_a, C\_a) holds everything the data say about T; the prior supplies how T(x\_a) and T(x\_a') must relate.

The current encoder falls short in three ways:

- It must discover these maps implicitly, through attention on y-similarity.
- Softmax attention averages over the branches it finds, so each "where could I be" map collapses into one blurred location (the mode-averaging problem from the HS-PFN doc).
- It then compresses n\_A blurred maps into 256 numbers.

The fix: keep per-point evidence explicit, and do not pool before the structure is used.

## How registration methods read two signals

Every successful registration family separates local correspondence evidence, a coherence step through the warp's smoothness, and iteration around the current guess. None summarises both inputs into one global vector before matching.

| Family | How it reads the two inputs | Our analogue | Fit for us |
| --- | --- | --- | --- |
| Optical flow (RAFT) | All-pairs correlation volume; a recurrent unit looks up a local window around the current flow and updates it | C\_a(u) is the correlation volume; windows around the current T(x\_a). The run-2 transport-token loop was a version of this and registered at Bayes level at small n\_A | High: learned in context, generalises to unseen pairs |
| Point sets without correspondences (ICP, Coherent Point Drift, RPM) | Alternate soft correspondences with a smooth-warp update under a coherence penalty (EM) | Soft correspondences = C\_a times the current prior; the smooth update = the kernel solve (ops doc A5) | High: it is the structure of our refinement |
| Multimodal image registration (mutual information) | Intensities relate by an unknown function, so match statistical dependence, not values | Exactly our h problem; with sparse A, use ranks instead of MI | High for h, in the rank version |
| Elastic curve registration (DTW, square-root velocity) | Dynamic programming finds the best monotone 1-D alignment exactly | In 1-D, with A sorted by x, the posterior over locations is a chain (below) | Exact in 1-D; message passing in d |
| Keypoint matching (SuperGlue, LoFTR) | Describe distinctive points, match with attention + Sinkhorn, fit the warp robustly | B's critical points are known; A's are uncertain at small n\_A | Medium: useful near the optimum, later |
| Amortised registration (VoxelMorph) | One network outputs a field for a pair | A pooled-summary approach like ours: one estimate, at best unimodal uncertainty | Low: what we should move away from |

**The 1-D chain.** Sort A's points by x. With dense g, the joint posterior over their B-locations (u\_1, …, u\_n) is approximately a Markov chain:

- emissions: C\_a(u\_a);
- transitions: p(u\_{a+1} | u\_a) from the transform prior — monotone, with u\_{a+1} − u\_a ≈ T′ · (x\_{a+1} − x\_a).

Forward–backward on a B grid of G points costs n\_A · G² and gives exact per-point marginals, all modes, and coherent joint samples of the correspondences — soft DTW with a probabilistic prior. In d ≥ 2 there is no ordering, so the chain becomes a graph over neighbouring A points with message passing, which is what within-slot attention approximates.

## Proposal v2: replace the pooled encoder

Build the proposal from explicit correspondences, so that it is multimodal by construction, as sharp as the data, and keeps an exact density for importance sampling.

1. **Evidence volumes.** For each A point, compute C\_a(u) on B's grid from the B model's predictive (handles noisy B). A band of values, or ranks, covers unknown h. A fixed operator, not learned.
2. **Coherence.** In 1-D, exact forward–backward along sorted A. In d ≥ 2, a few rounds of learned message passing between neighbouring A points, on location beliefs (distributions over u, or particles), not single vectors.
3. **Proposal from correspondences.** Draw M coherent correspondence sets from step 2. For each, solve the regularised least-squares fit for θ (Gauss–Newton or kernel solve, with the prior), giving a mean and a Laplace covariance. The proposal is the mixture below.
4. **Optional refinement.** A few MCMC moves or learned Gauss–Newton steps per sample. Or keep the flow, but condition its knot networks by cross-attention on the per-point tokens instead of the pooled c.

```latex
q(\theta)=\frac1M\sum_{m=1}^{M}\mathcal N\big(\theta;\,\hat\theta_m,\,(\Lambda+F_m)^{-1}\big)
```

Why it should beat the current proposal:

- No branch averaging: one component per branch combination.
- No global bottleneck.
- Correct width: Laplace widths instead of a flow that underfits wide.
- Learning is confined to judgement calls: emissions under unknown h, message passing in d ≥ 2, a correction of the mixture weights.

**Cost in d.** The evidence volume over B's domain grows like G^d. Mitigations: local windows around current guesses (RAFT-style), restricting to the important axes (low intrinsic dimension), and representing each level set by particles on its contour. That is the honest scaling limit, tied to the relevant dimension k rather than d.

This also answers "being about right first is paramount": the explicit correspondence posterior explores branch combinations directly, so importance sampling only has to correct, not search.

## Sample J by default

J becomes a latent of every task, so the model has to infer the transform's complexity; current pilot results assume it is known.

- **Training prior:** draw J per task from the hierarchy, including J = 0 (identity).
- **Proposal v2:** J enters at the fit step. Fit each sampled correspondence set under each J and score the fit by its Laplace evidence. The proposal becomes a mixture over (J, correspondence set), and Occam's razor comes from the evidence automatically.
- **Flow variant:** a categorical head over J, as in the complexity plan.
- **Weights:** include p(J) in both cases.

## The experiment that decides this

Compare four proposals at equal K; if the chain + Laplace mixture wins, the d ≥ 2 work becomes learned message passing over location beliefs on the important axes.

**Setup:** 1-D, h = id, J sampled, ECDF prior with BNN-sampled architecture, K ∈ {32, 128}.

| Variant | Proposal |
| --- | --- |
| (a) | Current pooled encoder E0 + flow |
| (b) | E1 transport-token loop + flow |
| (c) | Chain + Laplace mixture (v2, no learned parts) |
| (d) | v2 with learned emissions |

**Metrics:** ESS/K, calibration (SBC), coverage of the true mode, and NLL gap to the importance-sampling reference — the last as a function of n\_A and of the number of branches of g.

**Expected:** (c) beats (a) clearly at n\_A ≥ 2 and on wiggly g; (a) stays competitive only at n\_A = 1, where the posterior is close to the prior.
