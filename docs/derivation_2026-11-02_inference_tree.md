# From the generative model to the inference model: a derivation tree

*Research-log note, 2026-11-02. The derivation (§0–7) was written in a separate chat from the user's description of
the generative model and pasted here verbatim (without its closing question). Its "formulation from this morning"
reference points to a document from that chat, which is not in the repo. The appendix maps the derivation onto this
repo's priors and experiments; its facts were checked against the code on 2026-11-02.*

Below is the full derivation, written as a tree of numbered equations. Each node is split until the leaves are
densities you can write down in closed form. Three modelling choices were made that the description didn't state; they
are marked ⚠ because they change the leaves.

---

## 0. Generative model, written out

**Latents**

| Symbol | Meaning | Prior |
|---|---|---|
| $\omega=(\text{depth},\text{width})$ | architecture (discrete) | $p(\omega)$ |
| $\theta$ | weights | $p(\theta\mid\omega)=\mathcal N(0,\Sigma_\omega)$, with $\Sigma_\omega$ built from $\sigma_w^2/H_{\ell-1}$ and $\sigma_b^2$ |
| $F$ | ECDF of BNN outputs over the whole prior | **fixed and known**, so nothing to infer |
| $\tau=(T,h)$ | input field and output map | $p(\tau)=p(\vartheta_T)\,p(\vartheta_h)$ |
| $z$ | related ($1$) or unrelated ($0$) | $\text{Bern}(\bar\rho)$ |
| $(\theta',\omega')$ | fresh draw, only used when $z=0$ | same prior as $(\theta,\omega)$ |
| $\sigma_A,\sigma_B$ | noise levels | $p(\sigma)$, or fixed |
| $X_A,X_B,n_A,n_B$ | designs and sizes | $\mathcal U[0,1]^d$, $p(n)$ |

**Normalised base function**

$$g_{\theta,\omega} := F\circ \text{BNN}_{\theta,\omega}.$$

**Tasks**

$$f_B = g_{\theta,\omega},\qquad
f_A=\begin{cases} h\circ g_{\theta,\omega}\circ T & z=1\\[2pt] h\circ g_{\theta',\omega'}\circ T & z=0\end{cases}$$

$$y_b=f_B(x_b)+\varepsilon_B,\qquad y_a=f_A(x_a)+\varepsilon_A,\qquad y^*=f_A(x^*)+\varepsilon_A .$$

**⚠1. Pass the unrelated branch through $h\circ\cdot\circ T$ as well.** If $z=0$ draws a plain $g_{\theta'}$, then the
*marginal* look of A (its y-histogram, its smoothness) differs between $z=0$ and $z=1$. A single task would then leak
$z$ without any comparison to B. Under this construction, $\tau$ is just a nuisance in the $z=0$ branch, and $p(f_A)$
is identical in both branches.

**⚠2. Noise placement.** $F$ is applied before the noise, and the noise is added after $h$, in A's own scale. Section 6
gives the alternatives.

**⚠3. No residual $r$.** The description has none, unlike the v2 doc. Section 6 explains what this costs.

**Design ignorability.** $X$ and $n$ are independent of every latent, so

$$p(A,B\mid\cdot)=p(X_A)p(X_B)\,p(Y_A,Y_B\mid X,\cdot).$$

The design factors cancel from every ratio below, and the $x$'s are only ever conditioned on.

**Conditional-independence structure** (used throughout):

$$B\perp \tau \mid (\theta,\omega), \qquad A\perp B\mid(\theta,\omega,\tau,z{=}1), \qquad A\perp B \mid z{=}0 .$$

---

## 1. The tree at a glance

```
(E0) p(y*|x*,A,B) = Σ_z P(z|A,B) · p(y*|x*,A,B,z)
├── (E1) P(z|A,B) ∝ p(z) · p(A|B,z)                         ← Bayes factor
│     ├── p(A|B,z=1)  → (E3.2)
│     └── p(A|z=0)    → (E2.2)
├── (E2) z=0: p(y*|x*,A)                                    ← single-task PPD
│     ├── (E2.1) Σ_ω' P(ω'|A) ∫ p(y*|x*,θ',ω',τ) p(θ',τ|A,ω')
│     └── (E2.2) p(A) = Σ_ω' p(ω') ∫∫ p(A|θ',ω',τ) p(θ'|ω') p(τ)
└── (E3) z=1: Σ_ω ∫∫ p(y*|x*,θ,ω,τ) · p(θ,ω,τ|A,B)
      │     p(θ,ω,τ|A,B) ∝ p(θ,ω|B) · p(τ) · p(A|θ,ω,τ)        ← master posterior
      ├── Route α: τ outermost  ("register A against B's posterior")
      │     ├── (E3.1) p(τ|A,B) ∝ p(τ) · p(A|B,τ)
      │     ├── (E3.2) p(A|B,τ) = Π_a p(y_a|x_a,A_<a,B,τ)
      │     └── (E3.3) p(y*|x*,A,B,τ) = ∫ p(y*|x*,g,τ) p(g|B,A,τ)
      ├── Route β: function outermost ("align each candidate function")
      │     ├── (E3.4) p(g|A,B) ∝ p(g|B) · ∫ p(A|g,τ) p(τ) dτ
      │     └── (E3.5) p(τ|g,A)     ← B drops out given g
      └── τ split: T = warp∘scale∘shift, h = a + b·H
            ├── (E4.1) p(ϑ_T|A,B)
            └── (E4.2) p(ϑ_h|ϑ_T,A,B)  ← 1-D calibration, partly analytic
Leaves (L1–L4): Gaussian likelihoods · p(g|B) · p(τ) · p(ω), p(θ|ω)
```

---

## 2. Root and model posterior

$$
p(y^*\mid x^*,A,B)=P(z{=}0\mid A,B)\;p(y^*\mid x^*,A,z{=}0)\;+\;P(z{=}1\mid A,B)\;p(y^*\mid x^*,A,B,z{=}1) \tag{E0}
$$

$$
P(z\mid A,B)=\frac{p(z)\,p(A\mid B,z)}{\sum_{z'}p(z')\,p(A\mid B,z')} \tag{E1}
$$

E1 holds because $p(A,B\mid z)=p(B)\,p(A\mid B,z)$, and $p(B\mid z)=p(B)$: B's law does not depend on $z$, so $p(B)$
cancels. In the $z=0$ branch, $p(A\mid B,z{=}0)=p(A)$.

The posterior log-odds is

$$\log\frac{\bar\rho}{1-\bar\rho}+\log p(A\mid B,z{=}1)-\log p(A).$$

It is a *predictive* comparison: does B help predict A better than the prior does?

---

## 3. Unrelated branch ($z=0$): a single-task problem with a slightly richer prior

$$
p(y^*\mid x^*,A,z{=}0)=\sum_{\omega'}P(\omega'\mid A)\iint p(y^*\mid x^*,\theta',\omega',\tau)\;p(\theta',\tau\mid A,\omega')\,d\theta'\,d\tau \tag{E2.1}
$$

$$
p(\theta',\tau\mid A,\omega')\propto p(\theta'\mid\omega')\,p(\tau)\,p(A\mid\theta',\omega',\tau),\qquad
P(\omega'\mid A)\propto p(\omega')\,p(A\mid\omega')
$$

$$
p(A)=\sum_{\omega'}p(\omega')\iint p(A\mid\theta',\omega',\tau)\,p(\theta'\mid\omega')\,p(\tau)\,d\theta'\,d\tau
=\prod_a p(y_a\mid x_a,A_{<a}) \tag{E2.2}
$$

Here $\tau$ is pure nuisance: it only reshapes the function prior. **This branch is exactly the A-only PFN**, provided
that PFN was trained on $h\circ g\circ T$ draws (⚠1).

---

## 4. Related branch ($z=1$)

### 4.1 Master posterior

$$
p(\theta,\omega,\tau\mid A,B)\;\propto\;\underbrace{p(\omega)\,p(\theta\mid\omega)\,p(B\mid\theta,\omega)}_{\propto\,p(\theta,\omega\mid B)}\;\cdot\;p(\tau)\;\cdot\;p(A\mid\theta,\omega,\tau) \tag{E3}
$$

B enters only through the B-posterior over the shared function, and $\tau$ enters only through A. Everything below
comes from factoring this one expression in different orders.

### 4.2 Collapse to function space (do this before anything else)

Every likelihood depends on $(\theta,\omega)$ only through $g=g_{\theta,\omega}$. So

$$\sum_\omega\int d\theta\;p(\theta,\omega\mid\cdot)\,(\dots) \quad\longrightarrow\quad \int dg\;p(g\mid\cdot)\,(\dots).$$

This removes all weight-space symmetries: unit permutations, sign flips, and the **affine absorption**
$g_\theta(Ax+c)=g_{\theta'}(x)$ with $W_1'=W_1A$, $b_1'=b_1+W_1c$.

In weight space, an affine $T$ is therefore unidentifiable against $\theta$. In function space it is identifiable,
because B pins $g$ itself, not $\theta$. From here on, write $p(g\mid B)$. **This is the object the frozen B-PFN
amortises.**

### 4.3 Route α: $\tau$ outermost (the current design docs)

$$
p(y^*\mid x^*,A,B,z{=}1)=\int p(\tau\mid A,B)\;p(y^*\mid x^*,A,B,\tau)\,d\tau
$$

**Transform posterior:**

$$
p(\tau\mid A,B)\propto p(\tau)\,p(A\mid B,\tau) \tag{E3.1}
$$

**Evidence**, in two equivalent forms:

$$
p(A\mid B,\tau)=\int \prod_a \mathcal N\big(y_a;\,h(g(T(x_a))),\sigma_A^2\big)\,p(g\mid B)\,dg
=\prod_{a}p(y_a\mid x_a,A_{<a},B,\tau) \tag{E3.2}
$$

- The chain rule is exact in any order of A's points.
- The cheap B-only product $\prod_a p(y_a\mid x_a,B,\tau)$ is *not* this quantity. It treats A's points as
  independent given B, and so double-counts the parts of $g$ that are shared between them.

**Conditional predictive:**

$$
p(y^*\mid x^*,A,B,\tau)=\int \mathcal N\big(y^*;\,h(g(T(x^*))),\sigma_A^2\big)\;p(g\mid B,A,\tau)\,dg,\qquad p(g\mid B,A,\tau)\propto p(g\mid B)\,p(A\mid g,\tau) \tag{E3.3}
$$

With no residual and invertible $h$, $A$ pulled through $\tau$ *is* B-domain data:

$$A_{\text{in}B}=\{(T(x_a),\,h^{-1}(y_a))\}.$$

So E3.3 equals pushing the B-domain prediction through $h$ (⚠2 matters here; see §6):

$$h_\#\,p\big(\cdot\mid T(x^*),\,B\cup A_{\text{in}B}\big).$$

### 4.4 Route β: function outermost

$$
p(g\mid A,B)\propto p(g\mid B)\int p(A\mid g,\tau)\,p(\tau)\,d\tau \tag{E3.4}
$$

$$
p(\tau\mid g,A,B)=p(\tau\mid g,A)\propto p(\tau)\,p(A\mid g,\tau) \tag{E3.5}
$$

**B drops out of E3.5**, because $\tau\perp B\mid g$. Given a candidate function, aligning A to it is a pure
registration problem with $n_A$ points and no B anywhere.

Routes α and β are algebraically identical, but they lead to different estimators:
- **α:** samples over $\tau$, with B cached. This is IS or SMC over transforms.
- **β:** samples over functions from the B-posterior, then registers each one. This is natural when $p(g\mid B)$ is
  narrow, because dense B means few distinct $g$'s.

### 4.5 Splitting $\tau$

Parameterise

$$T=W_{\vartheta_w}\circ S_s\circ \text{Sh}_c,\qquad h(u)=a+b\,H_{\vartheta_H}(u)\quad (a\ge0,\ b>0,\ a{+}b\le1).$$

Then

$$
p(\tau\mid A,B)=\underbrace{p(\vartheta_T\mid A,B)}_{\text{E4.1}}\;\underbrace{p(\vartheta_h\mid \vartheta_T,A,B)}_{\text{E4.2}}
$$

$$
p(\vartheta_h\mid\vartheta_T,A,B)\propto p(\vartheta_h)\int\prod_a\mathcal N\big(y_a;\,a+bH(u_a),\sigma_A^2\big)\,p(u_{1:n_A}\mid B,\vartheta_T)\,du,\qquad u_a=g(T(x_a)).
$$

**Given $T$, $h$ is a 1-D monotone regression of $y_a$ on the transported B-values $u_a$.** For a sample of $u$ and
affine $h$ (i.e. $H=\text{id}$), this is linear-Gaussian in $(a,b)$, so it is conjugate up to the box constraint. It can
be integrated analytically instead of sampled, which is Rao–Blackwellisation.

$\vartheta_T$ further splits into shift, scale and warp. Only the warp is high-dimensional.

---

## 5. Leaves (everything is closed-form or a cached network)

| Leaf | Expression | Source |
|---|---|---|
| L1, B likelihood | $\prod_b\mathcal N(y_b;\,g(x_b),\sigma_B^2)$ | closed form; amortised into $p(g\mid B)$ by the B-PFN |
| L2, A likelihood | $\prod_a\mathcal N(y_a;\,h(g(T(x_a))),\sigma_A^2)$ | closed form given $(g,\tau)$ |
| L3, query | $\mathcal N(y^*;\,h(g(T(x^*))),\sigma_A^2)$ | closed form given $(g,\tau)$ |
| L4, priors | $p(\omega)$, $p(\theta\mid\omega)$, $p(\vartheta_T)$, $p(\vartheta_h)$, $\bar\rho$, $p(\sigma)$ | known, because they are the user's prior |
| L5, function posterior | $p(g\mid B)$ | not closed form; the one place a network is unavoidable |

**Out-of-window caveat.** If $T(x)$ leaves the region B covers, $p(g(T(x))\mid B)$ reverts to the prior there.
Uncertainty then grows correctly, as long as the B-PFN is trained to query outside its data.

---

## 6. Branch points that change the leaves

**(a) Where the noise sits.**
- Noise after $h$, as above: L2 is Gaussian in A's scale.
- Noise before $h$, i.e. $y_a=h(g(T(x_a))+\varepsilon)$: then
  $$p(y_a\mid\cdot)=\mathcal N\big(h^{-1}(y_a);g(T(x_a)),\sigma^2\big)\,\big|(h^{-1})'(y_a)\big|.$$
  The Jacobian is mandatory; without it, the evidence rewards a compressive $h$. In this case B-domain pooling becomes
  *exact*.
- If the noise is added before $F$ (i.e. $F(\text{BNN}+\varepsilon)$), the noise is non-Gaussian in $[0,1]$. Pick one
  convention and make sure the PFN training prior matches it.

**(b) No residual.** With $z=1$ and $r=0$, A's points are exact (noisy) observations of $g\circ T$. E3.2 is then
extremely peaked in $\tau$ when $\sigma_A$ is small, which has two consequences:
- The $\tau$-posterior becomes a set of near-delta ridges, and IS weights degenerate.
- The only way A can discredit B is all-or-nothing, through $z$; there is no local discrediting.

Adding $f_A=h\circ g\circ T+r$ adds one integral, $\int p(r)\,dr$, inside L2 and L3, and that softens both problems.

**(c) Unknown $\sigma$.** $\sigma_B$ is well identified from B, but $\sigma_A$ is poorly identified from $n_A$ points.
It trades off against $h$'s slope $b$, which is another axis of the identifiability ridge.

---

## 7. What this tree suggests testing next (cheapest first)

1. **z-leak check.** Train a tiny classifier on A alone (and on B alone) to predict $z$. It must be at chance. If it
   isn't, the unrelated branch violates ⚠1, and every "B helps" number is inflated.
2. **Route α vs. route β on $d=1$, against grid or HMC ground truth.** Both are the same integral, so their
   predictives must agree. A disagreement points to which factor is broken: the E3.2 chain rule, or the E3.5
   registration.
3. **Rao–Blackwellised $h$.** Compare IS over $(T,h)$ with IS over $T$ plus analytic affine $(a,b)$. Report the ESS
   ratio and NLL at fixed compute. This is a cheap win if it holds, and it shrinks the search to $T$ only.
4. **Peakiness sweep (no residual).** Plot ESS of prior-IS against $n_A$ and $\sigma_A$. The resulting curve tells
   whether the residual $r$ (or a heavier-tailed noise) is needed purely for the estimator to work, separately from
   whether it is realistic.

---

## Appendix: how the tree maps onto this repo (A→B session, checked against the code 2026-11-02)

**The ⚠ choices against the code.**
- **⚠1 holds.** In the ladder prior (`experimental/ladder.py`) and the S3/S4 prior (`experimental/a2b/prior.py`), the
  unrelated stratum keeps A as $h\circ f\circ T$ and gives *B* the fresh function. A's marginal law is therefore the
  same in both strata, which is the symmetric form of ⚠1.
- **⚠2 differs: noise sits before $F$ and before $h$.** pfn-repro's `BNNPrior` adds its noise (pre-activation and
  output) in the raw BNN space, then applies the ECDF $F$. `a2b/prior.py` applies $h$ to that noisy, $F$-normalised
  observation. So the repo uses both alternatives of §6(a), not the default:
  - the noise is non-Gaussian on $[0,1]$, so the leaves L1–L3 are not closed-form Gaussians and are only available
    through the PFN;
  - B-domain pooling with the Jacobian is exact. The S3/S4 oracle and teacher already use
    $\log p_A(y)=\log p_B(h^{-1}(y))+\log|(h^{-1})'(y)|$;
  - the frozen PFN was trained on this same convention, so the A-only PFN and the B-PFN match the prior.
- **⚠3 holds: no residual** in the ladder or S3/S4 priors. Two observations are consistent with §6(b):
  - S1's posteriors over the shift are near-delta from $n_A\approx5$;
  - the importance-sampling reference's ESS collapses at larger $n_A$ in the S3 smoke runs.

  The frozen-PFN K/V pre-check (`scripts/kv_precheck_a2b.py`) found that adding $A_{\text{in}B}$ to B's context
  improves prediction by only 0.01–0.04 nats/query. That matches "no local discrediting without $r$".

**The tree's nodes in our experiments.**

| node | where it already is |
|---|---|
| E0, E1 (Bayes factor over $z$) | `bayes_null` in notebook 08; the null particle in `a2b/model.py` |
| E2, E2.2 (A-only PPD, chain-rule evidence) | the frozen A-only PFN; `A2B.null_particle` (tested to match the chain rule) |
| E3.1 + E3.2 (route α, exact chain rule) | `grid_exact` in S1 (grid over the shift) |
| the cheap B-only product (not E3.2) | `grid_pseudo` in S1, and the particles' weights in `a2b` |
| E3.3 ($h_\#$ PPD on $B\cup A_{\text{in}B}$) | `teacher_b` in S0 and S3/S4; the `context` variant |
| route β (E3.4, E3.5) | not implemented |
| E4.2 (analytic affine $h$) | not implemented; S4's `affine` stratum is where it applies |

**The §7 tests against what exists.**
- (1) The z-leak check has not been run. Our priors satisfy ⚠1 by construction, but the check is cheap.
- (2) Route α vs. route β: route α's ground truth exists for the shift family (S1's grid), but route β does not.
- (3) The Rao–Blackwellised affine $h$ is new and fits S4.
- (4) The peakiness sweep is partly covered: S1's entropy and mass vs $n_A$ at the PFN's own noise. The $\sigma_A$
  axis is not swept; the BNN prior draws its own noise levels.
