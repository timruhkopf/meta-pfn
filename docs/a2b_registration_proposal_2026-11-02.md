# A↔B transfer by in-context registration: proposal and verification plan

*Version 2026-11-02. Standalone hand-off spec, meant to be executed by Claude Code and kept as the log of record.*

*Background derivations live in the project doc `a2b_bayes_tree_verbose_2026-11-02.md`, cited as "tree §x". Everything needed to implement and check the proposal is in this document.*

---

## 0. Summary

**Goal.** Predict A's held-out values, $p(y^*\mid x^*,A,B)$, by transferring information from a densely observed related task B. A is a transformed copy of B: $f_A=h\circ g\circ T$, with B observing $g$ itself.

**Proposal: amortised importance sampling over an explicit transform latent.** It has three learned components, each trained against an *exact factor* of the Bayesian solution with a *proper scoring rule*, and an exact rule for combining them:

1. **Transform latent.** The transform is $T=T_\vartheta$, the time-1 flow of a box-preserving velocity field with coefficients $\vartheta\in\mathbb R^J$. The prior density $p(\vartheta)$ is known, and the inverse $T_\vartheta^{-1}$ is exact.
2. **Transported PFN $D_\psi$** (the "decoder").
   - Input: A, plus B *pulled into A's coordinates* by a given $\vartheta$: $B_{\text{in}A}(\vartheta)=\{(T_\vartheta^{-1}(x_b),y_b)\}$.
   - It outputs the predictive $p(y^*\mid x^*,A,B,\vartheta)$ and, via a causal mask, the evidence $p(A\mid B,\vartheta)$.
   - $h$ is never estimated: $D_\psi$ marginalises it in context.
   - It is trained on prior simulations with the **true** $\vartheta$.
3. **Amortised proposal $q_\phi(\vartheta\mid A,B)$.** A conditional normalising flow on top of an in-context encoder of $(A,B)$. It is trained by neural posterior estimation (forward KL to the true $\vartheta$). **This is the registration operator:** it learns to propose transforms near the posterior, so the number of samples needed scales with how complex the posterior is, not with the volume of the prior.
4. **Exact combination by self-normalised importance sampling.**
   - Weights: $w_k=p(\vartheta_k)\,\hat p(A\mid B,\vartheta_k)/q(\vartheta_k)$, with $\vartheta_k\sim q$.
   - Plus an A-only "null" branch and the relatedness posterior $P(z\mid A,B)$.

**Why this is the principled choice.**
- Every learned part targets a well-defined Bayesian object, so it can be checked separately against exact references.
- Errors decompose additively into decoder error, proposal error and finite-sample error (§7.1).
- With a correct decoder the estimator is **consistent** for any proposal with adequate support. The proposal only affects efficiency, never correctness.
- No component is trained on its own outputs, so there is **no train/test mismatch** by construction.

**Verification.** A staged plan (V0–V8, §8) in d = 1 with an exact grid-Bayes reference, then scaling in the transform dimension J and in d = 2. Each stage has numeric pass criteria.

---

## 1. Problem setting (generative prior)

One training example is produced by this program. Defaults follow the earlier runs, so results stay comparable; changes are marked **[new]**.

| Step | Variable | Definition |
|---|---|---|
| 1 | base function $g$ | random 2-layer tanh MLP (width 32, random input and weight scales), min–max normalised to $[0,1]$ on a 256-point grid over $[0,1]^d$ |
| 2 | transform $T_\vartheta$ **[new]** | §3.1: flow of a box-preserving velocity field, $\vartheta\sim\mathcal N(0,\Lambda^{-1})$ |
| 3 | output map $h$ | phase H0: identity. H1: affine $a+bv$, $b\in[0.3,1]$, $a\in[0,1-b]$. H2: $a+b\,\text{Kumaraswamy}_{\kappa}(v)$, $\log\kappa\sim\mathcal N(0,0.3^2)$ |
| 4 | relatedness $z$ | phase Z0: $z=1$ always. Z1: $z\sim\text{Bern}(0.8)$; if $z=0$, A uses an independent $g'$ but **the same** $h\circ\cdot\circ T$ construction (choice C1, tree §1.3) |
| 5 | designs | $X_B$: 128 uniform points. $X_A$: $n_A\sim\mathcal U\{1..10\}$ uniform points. $X^*$: 32 uniform query points (16 in training) |
| 6 | noise | convention C2′ (as in runs 1–2): $y_b=g(x_b)+\varepsilon_b$, $y_a=h(g_\bullet(T(x_a)))+\varepsilon_a$, with $\sigma_A,\sigma_B\sim\mathcal U[0.005,0.03]$ independently. Optional switch to C2 (noise before $h$) for check V7b |

**Privileged quantities** (training only): $\vartheta$, $T$, $T^{-1}$, $h$, $g$, $z$, noise-free values. In particular:
- $A_{\text{in}B}=\{(T(x_a),h^{-1}(y_a))\}$;
- $B_{\text{in}A}=\{(T^{-1}(x_b),h(y_b))\}$.

---

## 2. Target and exact decomposition

Notation: $p(A\mid\cdot)$ means $p(y_{1:n_A}\mid x_{1:n_A},\cdot)$. The inputs $x$ are always conditioned on, never modelled: designs are ignorable (tree §3).

**(R0) Root.** Sum over $z$, using $A\perp B\mid z{=}0$:

$$
p(y^*\mid x^*,A,B)=P_0\;p(y^*\mid x^*,A)\;+\;P_1\;p_1(y^*),\qquad P_1=P(z{=}1\mid A,B).
$$

**(R1) Relatedness.**

$$
P_1=\frac{\bar\rho\,Z_1}{\bar\rho\,Z_1+(1-\bar\rho)\,Z_0},\qquad Z_1=p(A\mid B,z{=}1),\quad Z_0=p(A).
$$

Under C1, the null evidence $Z_0$ is the A-only marginal (tree §4.2).

**(R2) Related branch as an integral over the transform.** $h$ (and $g$) are marginalised *inside* the conditional terms:

$$
p_1(y^*)=\int p(\vartheta\mid A,B)\;p(y^*\mid x^*,A,B,\vartheta)\,d\vartheta,\qquad
p(\vartheta\mid A,B)=\frac{p(\vartheta)\,p(A\mid B,\vartheta)}{Z_1}.
$$

**(R3) Evidence of a transform**, by the chain rule. It is exact in any order, because the product telescopes to a symmetric integral (tree §5.2):

$$
p(A\mid B,\vartheta)=\prod_{a=1}^{n_A}p(y_a\mid x_a,A_{<a},B,\vartheta).
$$

**(R4) Given the transform, B is A-frame data.**
- Given $\vartheta$, each B point is an observation of $f_A$'s *base* function at $T_\vartheta^{-1}(x_b)$: $y_b=g(x_b)+\varepsilon_b=(g\circ T_\vartheta)(T_\vartheta^{-1}(x_b))+\varepsilon_b$, seen *without* $h$.
- So $p(y^*\mid x^*,A,B,\vartheta)$ is the posterior predictive of a single A-frame regression problem with **two data sources**:
  - A's points observe $h\circ\tilde g$;
  - B's transported points observe $\tilde g$ itself, where $\tilde g:=g\circ T_\vartheta$.
- $h$ is the only unknown linking the two sources. It is one-dimensional and monotone, and it is identified by the A points alone.
- A network trained on simulations of exactly this two-source problem **computes this predictive, with $h$ integrated out**. This is the A-frame counterpart of the tree's pooling identity (tree §8.2), and it holds for either noise convention, because the network learns the observation model.

**(R5) Importance sampling.** For any proposal $q$ whose support covers the posterior:

$$
w_k=\frac{p(\vartheta_k)\,p(A\mid B,\vartheta_k)}{q(\vartheta_k)},\quad \vartheta_k\sim q,\qquad
\hat Z_1=\tfrac1K\textstyle\sum_k w_k,\qquad
\hat p_1(y^*)=\frac{\sum_k w_k\,p(y^*\mid x^*,A,B,\vartheta_k)}{\sum_k w_k}.
$$

- $\hat Z_1$ is unbiased.
- $\hat p_1$ is consistent, with $O(1/K)$ bias.
- **$q$ may depend on $A$ and $B$** (in fact it should). Validity only requires that $q(\vartheta_k)$ can be evaluated. Evaluating the evidence at a $\vartheta$ chosen with A is *not* double counting, because the division by $q$ corrects for the choice.

---

## 3. Components

### 3.1 Transform family: box-preserving flows with an exact inverse

**d = 1.**

$$
v_\vartheta(x)=x(1-x)\sum_{j=1}^{J}\vartheta_j\,\phi_j(x),\qquad \phi_j(x)=\cos\big((j-1)\pi x\big),\qquad \vartheta_j\sim\mathcal N\big(0,\ s_T^2\,j^{-2}\big),
$$

$$
T_\vartheta=\Phi^{v_\vartheta}_{1}\ \ (\text{time-1 flow of }\dot x=v_\vartheta(x)),\qquad T_\vartheta^{-1}=\Phi^{-v_\vartheta}_{1}.
$$

**Properties:**
- $v(0)=v(1)=0$, so $T$ maps $[0,1]$ onto itself and fixes the endpoints.
- 1-D flows preserve order, so $T$ is monotone.
- The inverse is exact: integrate the negated field.

**Calibration and integration:**
- Choose $s_T$ so that $\mathbb E|T(x)-x|\approx0.13$ under the prior. That matches the identity-registration error of runs 1–2, so the transform is about as strong as before.
- Defaults: $J\in\{2,4,8,16\}$, starting at $J=2$, where an exact grid reference exists.
- Integrate with RK4, 16 steps. Check that $|T^{-1}(T(x))-x|<10^{-4}$.

**d = 2 (stage V6).**
- $v_i(x)=x_i(1-x_i)\sum_{p,q<m}\vartheta_{i,pq}\cos(p\pi x_1)\cos(q\pi x_2)$, with $m=3$, so $J=18$.
- Prior variance $\propto(1+p^2+q^2)^{-2}$.
- The box is preserved because $v_i$ vanishes on the faces $x_i\in\{0,1\}$.

**Why this family:** a known prior density $p(\vartheta)$ (needed by R5), an exact inverse (needed by R4), and a dimension $J$ we can turn up to test scaling.

### 3.2 Transported PFN $D_\psi$ (decoder; learns R3 and R4)

**Tokens** (embedding: linear or Fourier features of each field, summed):
- A context: $(x_a,\ y_a,\ \text{src}{=}A)$;
- B context: $(T_\vartheta^{-1}(x_b),\ y_b,\ \text{src}{=}B)$, where $y_b$ stays in **B's scale**;
- queries: $x^*_j$, plus "evidence queries" $x_a$ for $a=1..n_A$.

**Attention mask:**
- context tokens attend to all context tokens they are allowed to see (standard PFN);
- query $x^*_j$ attends to all of A and B;
- **evidence query $a$** attends to all B tokens and to A tokens $<a$ only, in a random permutation drawn per task.

**Output.** A bar distribution over $y$ in A's scale. Reuse the run-1 bucket borders.

**Loss** (on prior simulations, with the true $\vartheta$, $z=1$):

$$
\mathcal L_D=-\sum_{a}\log D_\psi(y_a\mid x_a,A_{<a},B_{\text{in}A}(\vartheta))\;-\;\sum_j\log D_\psi(y^*_j\mid x^*_j,A,B_{\text{in}A}(\vartheta)).
$$

- Use the noise-free soft targets for the $y^*$ term where available (variance reduction, consistent: tree LUPI I.4.1).
- Each term is a proper score for a conditional of the joint prior. The minimisers are exactly the factors of (R3) and the predictive of (R4).

**Null head.** The same network with all B tokens removed. Train it on every task (all $z$, under C1), as a second forward pass with B dropped: this gives $p(y\mid x,A_{<a})$ and $p(y^*\mid x^*,A)$, hence $Z_0$.

**Size.** 6 layers, $d_{\text{model}}=128$, 4 heads. Train for about 200k tasks, or until validation NLL plateaus.

### 3.3 Proposal $q_\phi(\vartheta\mid A,B)$ (the registration operator)

**Encoder $E$.** Two variants; the comparison is check V5.
- **E0 (baseline).**
  - 2 self-attention layers per task (shared weights, src embedding);
  - 2 layers in which A tokens cross-attend to B;
  - pooling by multi-head attention with 4 learned seeds, giving a context vector $c\in\mathbb R^{256}$.
- **E1 (registration-aware).** E0, plus the run-2 transport-token registration module (`embed` variant: iterated location beliefs $(\hat u,s)$ per A and A_test token, R = 4).
  - Its final $(\hat u_a,s_a)$, together with B features read at $\hat u_a$, are appended to the A tokens before pooling.
  - Keep its dense LUPI loss (cross-entropy of each lookup row against the oracle plan $P^*$, a Gaussian kernel around $T(x)$ with width 0.02) as an **auxiliary** loss.

**Flow.** A conditional neural spline flow over $\vartheta\in\mathbb R^J$ (`zuko.flows.NSF`, 5 transforms, hidden size 128), conditioned on $c$.

**Loss: neural posterior estimation.**

$$
\mathcal L_q=-\log q_\phi\big(\vartheta_{\text{true}}\mid E(A,B)\big),\qquad\text{on prior simulations with }z=1.
$$

**Why it is proper.** Its expectation over the prior equals $\mathbb E_{A,B}\,\mathrm{KL}\big(p(\vartheta\mid A,B)\,\|\,q_\phi\big)$ plus a constant. So the minimiser is the exact posterior. Forward KL is **mass-covering**, which is what an IS proposal needs, and the flow can represent several modes.

**What it does not have to estimate:**
- $h$: the posterior targeted is the marginal over $h$ (and $g$);
- the directions of $\vartheta$ that don't matter: those stay at prior width automatically.

**Defensive mixture at inference.** Sample from and evaluate $q_{\text{mix}}=(1-\epsilon)\,q_\phi+\epsilon\,p(\vartheta)$ with $\epsilon=0.1$. This bounds the weights by $p\cdot\text{lik}/(\epsilon p)$, so no single sample can blow up the estimate.

### 3.4 Inference: fixed cost, no optimisation loop

```
c        = E(A, B)
θ_1..K   ~ q_mix(·|c)                    ; log q_mix(θ_k)            K = 32 by default
for k (batched):
    B_k  = {(T_{θ_k}^{-1}(x_b), y_b)}
    ev_k = Σ_a log D(y_a | x_a, A_<a, B_k)       # one forward, evidence queries
    P_k  = D(· | x*, A, B_k)                     # same forward, query tokens
log w_k  = log p(θ_k) + ev_k − log q_mix(θ_k)
log Ẑ1   = logsumexp(log w) − log K
p̂1       = Σ_k softmax(log w)_k · P_k
log Z0   = Σ_a log D_null(y_a | x_a, A_<a)
P1       = σ( logit ρ̄ + log Ẑ1 − log Z0 )
output   = (1 − P1) · D_null(· | x*, A) + P1 · p̂1
```

Do all arithmetic in log space. Average $\text{ev}_k$ over 2 random A permutations to reduce order dependence.

### 3.5 Training schedule

The three nets are trained **independently**, on fresh prior simulations; no stage consumes another's outputs.
1. $D_\psi$ and its null head (§3.2).
2. $q_\phi$ with its encoder (§3.3).
3. Nothing joint.

**Optional later step:** distil the whole estimator into a single-forward PFN, with the IS predictive as a soft target.

---

## 4. Design rationale

| Decision | Reason |
|---|---|
| Explicit finite latent $\vartheta$ with known prior | Needed for exact IS weights. Earlier slot designs refined hypotheses without a tractable proposal density, so their weights could only be patched by a learned correction. |
| Learned proposal instead of prior draws | Prior IS degenerates as $J$ grows. NPE concentrates samples where the posterior is, and its cost tracks posterior complexity. |
| Pull B into A's frame (not A into B) | B's raw points each enter once (no double counting). The evidence is a density over the observed $y_a$, so no Jacobian of $h$ is needed (tree §1.5). $h$ is marginalised in context. A future residual between tasks lives in A's coordinates (tree §16). |
| $h$ not in $\vartheta$ | It has low dimension and maps values to values. Leaving it to $D_\psi$ is an implicit Rao–Blackwellisation (tree §10.2). |
| Decoder trained only with the true $\vartheta$ | It then computes $p(\cdot\mid A,B,\vartheta)$ for the $\vartheta$ it is given, which is exactly what IS needs. No exposure bias, because no training input is a model output. |
| Mixture weights from evidence, not learned logits | They are exact up to decoder accuracy. The learned parts are checkable in isolation. |
| **Rejected** | OT/GW objectives (the input distribution carries no information about $T$; GW assumes an isometry); a pooled point estimate of $T$ (no posterior); MSE on correspondences (averages modes); prior-sample slots without weights correction; cycle-consistency losses (invertibility is built in). |

---

## 5. Exact references

**G-ref: grid Bayes** (d = 1, $J=2$; reuse `eval_regonly.py` logic from run 2).
- Dense-B limit: $g$ known exactly.
- Grid of $101^2$ points over $\vartheta\in[-4,4]^2$ prior standard deviations.
- Posterior $\propto p(\vartheta)\prod_a\mathcal N(y_a;h(g(T_\vartheta(x_a))),\sigma_A^2)$ with the true $\sigma_A$.
- For h = affine (H1): marginalise $(a,b)$ on a $32^2$ grid per $\vartheta$.
- PPD: grid mixture of $\mathcal N(y^*;h(g(T_\vartheta(x^*))),\sigma_A^2)$, integrated over the bar buckets.
- **Caveat:** it knows $g$ exactly, while the models see 128 noisy B points. That alone is worth about 0.1 nats at large $n_A$ (run 2). Report it as a floor, not a target.

**IS-ref: large-K prior IS with the learned decoder.**
- R5 with $q=p(\vartheta)$, $K=8192$ (or $q_\phi$ with $K=4096$ once V3 has passed).
- This is "our estimator with a perfect proposal". It separates decoder error from proposal error.

**IS-exact for $J>2$:**
- dense-B likelihood, $q_\phi$ as proposal with the defensive mixture, $K=10^4$;
- check $\text{ESS}>500$ before trusting it;
- in d = 2, add tempered SMC as a cross-check.

**Context baselines** (recompute under the new transform prior):
- **A-only:** null head.
- **Teacher:** $D_\psi$ given the true $\vartheta$.
- **Oracle:** an A-frame PFN with the true $B_{\text{in}A}$ *and* the true $h$ applied to B's values. This is the ceiling.
- **Run-2 `embed`**, retrained on the new prior if time allows; otherwise compare *gaps to G-ref* (run 2: gap 0.25 nats at $n_A$ = 1, 0.36 at $n_A$ = 10).

---

## 6. Metrics

All per query, in nats, stratified by $n_A\in\{1,2,3,5,8,10\}$, with 256 tasks per $n_A$ and 32 queries each.

| Metric | Definition | Measures |
|---|---|---|
| NLL | $-\log\hat p(y^*)$ on the bar distribution | the target |
| gap | NLL − G-ref NLL (or − IS-ref) | distance to Bayes |
| ESS/K | $(\sum w)^2/(K\sum w^2)$ | proposal efficiency |
| SBC | rank of $\vartheta_{\text{true},j}$ (and of $T(x^*)$) among 255 samples of $q$; uniformity via a KS test | calibration of $q$ |
| HPD coverage | fraction of true $\vartheta$ inside the 50% and 90% HPD regions of $q$ | calibration of $q$ |
| KL$(p_{\text{grid}}\,\|\,q)$ | on the $J=2$ grid | posterior fidelity |
| evidence fidelity | KL(grid posterior ‖ $p(\vartheta)\,\hat p(A\mid B,\vartheta)$ normalised on the grid) | correctness of the decoder's likelihood |
| permutation spread | std of $\text{ev}$ over 8 permutations of A | chain-rule consistency |
| prediction-relevant error | $\lvert g(T_{\hat\vartheta}(x^*))-g(T(x^*))\rvert$ vs $\lvert T_{\hat\vartheta}(x^*)-T(x^*)\rvert$, under samples of $q$ | whether uncertainty sits only where it doesn't matter |
| $P(z)$ calibration | reliability diagram, ECE | relatedness gate (Z1) |

---

## 7. Error decomposition (the main diagnostic)

$$
\underbrace{\text{NLL}_{\text{ours}}-\text{NLL}_{\text{G-ref}}}_{\text{total gap}}
=\underbrace{\big(\text{NLL}_{\text{IS-ref}}-\text{NLL}_{\text{G-ref}}\big)}_{\text{decoder error + finite-B}}
+\underbrace{\big(\text{NLL}_{q\text{-IS},K}-\text{NLL}_{\text{IS-ref}}\big)}_{\text{proposal / finite-}K\text{ error}} .
$$

Report both terms at every stage from V4 on. If the first term dominates, improve $D_\psi$. If the second dominates, improve $q_\phi$ or the encoder.

---

## 8. Verification plan

Run the stages in order; each is a go/no-go. Every stage writes `results/V*.json` and appends a table to `results/LOG.md` (format in §10).

### V0: prior and references (no learning)
- Implement §1 and §3.1. Tests:
  - $T$ is monotone;
  - $T(0)=0$ and $T(1)=1$;
  - inverse error $<10^{-4}$;
  - histogram of $|T(x)-x|$, mean ≈ 0.13.
- Implement G-ref ($J=2$, H0).
- Report the G-ref NLL against $n_A$.
- **Pass:** tests green. G-ref NLL lies between the A-only and oracle curves.

### V1: decoder $D_\psi$ (H0, Z0)
1. **Predictive with the true $\vartheta$** (teacher) against the oracle and A-only.
   - **Pass:** monotone improvement with $n_A$; at $n_A=10$, within 0.1 nats of the oracle (with H0 the teacher and the oracle differ only by having to infer $h=\text{id}$ in context).
2. **Evidence fidelity on the $J=2$ grid** (§6).
   - **Pass:** median KL < 0.1 nats at every $n_A$.
   - Also plot $\log\hat p(A\mid B,\vartheta)$ against the exact value for $\vartheta$ *far from the truth*: the decoder must not over-rate inconsistent transforms. Report the 95th percentile of over-rating.
3. **Permutation spread.** **Pass:** median < 0.05 nats.
- **If 2 fails** (out-of-distribution conditioning on wrong transforms): add a small residual to the training prior (tree §16, amplitude $\eta\sim$ half-normal with scale 0.02), so that incoherent (A, B_inA) sets become in-distribution. Re-test.

### V2: perfect-proposal estimator (IS-ref)
- Prior IS with $K=8192$ and $D_\psi$. Compare to G-ref.
- **Pass:** gap ≤ 0.12 nats at every $n_A$. This is the irreducible "decoder + finite-B" term.

### V3: proposal $q_\phi$ (E0 encoder, $J=2$)
- SBC, HPD coverage, KL to the grid posterior.
- **Pass:**
  - SBC KS test $p>0.01$ for each coordinate;
  - 50%/90% coverage within ±5 percentage points;
  - median KL < 0.2 nats.
- Also report the prediction-relevant error ratio. **Expected:** $q$ stays wide along directions where $g\circ T$ is flat.

### V4: full estimator ($J=2$, H0, Z0)
- $q$-IS with $K\in\{8,32,128\}$.
- **Pass:**
  - $\text{NLL}_{q\text{-IS},32}-\text{NLL}_{\text{IS-ref}}\le0.03$;
  - ESS/K ≥ 0.3;
  - total gap to G-ref ≤ 0.15 at every $n_A$, i.e. better than run-2 `embed`'s gaps (0.25 to 0.36).
- Report the error decomposition (§7).

### V5: scaling in the transform dimension (the central claim)
- $J\in\{2,4,8,16\}$ in d = 1. At $K=32$, compare:
  - prior IS;
  - $q$-IS with E0;
  - $q$-IS with E1.
- Metrics: ESS/K, NLL gap to IS-exact, SBC.
- **Pass:**
  - ESS/K of $q$-IS ≥ 0.2 at $J=16$ while prior IS collapses (expected below 0.01);
  - NLL gap grows by ≤ 0.05 from $J=2$ to $J=16$.
- **E1 vs E0:** does the explicit registration loop with dense LUPI improve $q$ at small $n_A$ or large $J$? Decide whether to keep it.

### V6: d = 2 ($J=18$)
- Repeat V3–V5 metrics, with IS-exact and SMC as references.
- **Frankenstein check:** joint samples of $q$ across queries vs the nearest posterior mode. Because $\vartheta$ is global, coherence is guaranteed per sample. Verify that the mixture covers the multimodal cases (stratify by the number of posterior modes, which SMC can count).
- **Pass:** criteria as in V5, with tolerances ×1.5.

### V7: output map $h$
- **V7a.** H1 (affine), then H2 (Kumaraswamy shape). G-ref with $(a,b)$ marginalised (H1 only).
  - **Pass:** total gap ≤ 0.2 at all $n_A$.
  - Expect larger gaps at $n_A\le2$. That is Bayes-correct, since $h$ is prior-dominated there; check it against G-ref, not against the oracle.
- **V7b.** Convention check: generate under C2 and C2′, train on one, test on both.
  - Expected: the decoder is robust when trained on a mixture of both. Record the cost of a convention mismatch.

### V8: relatedness (Z1, $\bar\rho=0.8$)
- **Pass:**
  - ECE of $P_1$ < 0.05;
  - on unrelated tasks, NLL within 0.05 of A-only;
  - on related tasks, no loss larger than 0.03 relative to Z0.

### Later (out of scope here)
Residual between tasks (tree §16), BO-shaped designs, $d\ge3$.

---

## 9. Risks and fallbacks

| Risk | Symptom | Fallback |
|---|---|---|
| Decoder over-rates wrong transforms | V1.2 fails off-posterior | residual-noise training prior (V1); or temper the evidence with $\beta<1$, chosen on validation |
| $q$ too narrow (mode dropping) | SBC ∩-shaped, ESS small, IS-ref ≫ $q$-IS | larger $\epsilon$; train $q$ with a slightly inflated posterior target (e.g. noise-augmented $\vartheta$); ensemble of flows |
| $q$ uninformative at small $n_A$ | ESS fine, but $q\approx$ prior | **correct behaviour** when the posterior ≈ prior (check against G-ref); not a failure |
| Flow family can't reach the true transforms | only relevant if real data needs non-box T | compose 2 fields; or switch $T$ to map into $U\supset[0,1]$ and handle partial $B_{\text{in}A}$ (tree §12) |
| Cost of $K$ decoder passes | slow inference | $K=8$ once ESS is high; cache B's per-layer K/V per $\vartheta$ is not possible (B moves), so keep $D_\psi$ small |
| Finite-B vs dense-B reference | constant gap ≈ 0.1 at large $n_A$ | report it; use IS-ref as the practical target |

---

## 10. Implementation notes for Claude Code

**Environment.**
- Python with `torch`, `numpy`, `scipy`, `zuko` (`pip install zuko --break-system-packages` if needed).
- CPU is fine for d = 1. Keep each training run ≤ 3 h on 2 cores, by reducing tasks per run if necessary.

**Reuse.** If the run-1/2 code is available (`prior.py`, `model.py`, `train.py`, `eval.py`, `eval_regonly.py`), reuse its BNN prior, bar buckets, grid-Bayes code and the `embed` registration module. Otherwise implement from §1 and §5.

**Suggested layout.**

```
a2b_air/
  prior.py        # g, T_θ (flow, inverse), h, z, sampling of tasks; tests (V0)
  reference.py    # G-ref grid Bayes (J=2, h grid), IS-exact (dense-B), SMC (d=2)
  decoder.py      # D_ψ transported PFN + null head + masks
  proposal.py     # encoders E0/E1, conditional NSF q_φ
  estimator.py    # §3.4 inference, error decomposition
  train_decoder.py, train_proposal.py
  eval.py         # stages V1–V8 → results/V*.json, results/LOG.md
  tests/          # T monotone/inverse; chain-rule permutation invariance on toy data
```

**Determinism.** Fixed seeds per stage. Evaluation tasks are generated once and saved, `eval_tasks_{stage}.pt`, so every model is scored on identical tasks.

**`results/LOG.md` format, per stage.**
- Date, git hash (or file hashes), config.
- One NLL table (columns: $n_A$; A-only, ours, IS-ref, G-ref, teacher, oracle).
- One diagnostics table (ESS/K, SBC p-values, coverage, evidence KL).
- Pass/fail per criterion.
- Three sentences of interpretation.
- **Never overwrite an earlier entry; append.**

**Stop rule.** If a stage fails its pass criterion, stop. Report the failure with the diagnostics, apply the listed fallback once, and re-test before moving on.

---

## Appendix A: why each objective is proper

1. **Decoder.** For any conditioning set $C$ (here $(x,A_{<a},B,\vartheta)$ or $(x^*,A,B,\vartheta)$), $\mathbb E\big[-\log D(y\mid C)\big]$ under the joint prior is minimised by $D=p(y\mid C)$. Taking $\vartheta$ as an input means $D$ learns the conditional *given* $\vartheta$.
2. **Proposal.** $\mathbb E_{p(\vartheta,A,B)}[-\log q(\vartheta\mid A,B)]=\mathbb E_{A,B}\big[\mathrm{KL}(p(\vartheta\mid A,B)\,\|\,q)\big]+H$, with $H$ constant in $q$. The minimiser is the posterior.
3. **Soft targets.** Replacing $y^*$ by its conditional law given the privileged latents $\lambda$ leaves the minimiser unchanged, because $\mathbb E_{\lambda\mid A,B}\,p(y^*\mid x^*,\lambda)=p(y^*\mid x^*,A,B)$ (tower property). It lowers gradient variance.
4. **IS.** $\mathbb E_q[w]=\int p(\vartheta)\,p(A\mid B,\vartheta)\,d\vartheta=Z_1$. The self-normalised predictive converges to $p_1$ as $K\to\infty$. Data-dependent $q$ is allowed because $w$ divides by $q$.

## Appendix B: glossary

| Term | Meaning |
|---|---|
| $A_{\text{in}B}$, $B_{\text{in}A}$ | a task's data moved into the other task's coordinates and scale (privileged) |
| PFN | prior-fitted network: a transformer trained on prior simulations to output posterior predictives in context |
| NPE | neural posterior estimation: train a conditional density on (latent, data) simulations |
| SBC | simulation-based calibration: posterior ranks of the true latent must be uniform |
| ESS | effective sample size of importance weights |
| dense-B limit | reference that knows $g$ exactly |
| C1, C2, C2′ | prior choices: unrelated branch uses the same $h\circ\cdot\circ T$; noise before / after $h$ |
