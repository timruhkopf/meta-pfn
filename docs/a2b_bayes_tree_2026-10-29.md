# A→B: the Bayesian equation tree, from generative model to estimators (2026-10-29, afternoon)

*This doc derives the posterior predictive $p(y^*\mid x^*,A,B)$ from the generative model, splitting every node until its leaves are closed-form densities or one cached network. It refines the morning docs (`a2b_math_formulation_2026-10-29.md`, `a2b_design_v2_2026-10-29.md`) and records the answers to the questions raised in chat. Where this doc and the morning docs disagree, this doc wins (see §14).*

**Guiding principle (user).** We never estimate $(\theta,\omega)$ explicitly; their posterior is highly multimodal (permutations, sign flips, absorbed transforms). Only the **function** they encode matters. The derivation still *starts* in $(\theta,\omega)$ because $\omega$ (depth, width) encodes functional complexity, i.e. the capacity of the prior. It then collapses to function space at the first point where that is legitimate (§7.2).

---

## 0. Conventions

| Symbol | Meaning |
|---|---|
| $A=A_{\text{train}}=\{(x_a,y_a)\}_{a=1}^{n_A}$ | observed target data, in **A's coordinates** |
| $A_{\text{test}}=\{(x^*_j,y^*_j)\}_j$ | queries: $x^*\in[0,1]^d$ in **A's coordinates**; $y^*$ unknown, to be predicted |
| $B=\{(x_b,y_b)\}_{b=1}^{N_B}$ | related data, in B's coordinates, with $N_B\gg n_A$ |
| $u$ | a location in **B's / the base function's** coordinates, e.g. $u=T(x)$ |
| $A_{<a}$ | the first $a-1$ points of A, in a fixed order (acquisition order if A is a BO trajectory) |

Each $x^*_j$ is predicted **independently**: we want the marginal PPD per query, not a joint over $A_{\text{test}}$. This is what a PFN outputs.

---

## 1. Generative model

### 1.1 Latents and priors

| Latent | Meaning | Prior |
|---|---|---|
| $\omega=(\text{depth},\text{width})$ | architecture, i.e. capacity / functional complexity | $p(\omega)$, discrete |
| $\theta$ | weights and biases | $p(\theta\mid\omega)=\mathcal N(0,\Sigma_\omega)$, with $\Sigma_\omega$ from $\sigma_w^2/H_{\ell-1}$ and $\sigma_b^2$ |
| $F$ | ECDF of BNN outputs pooled over the *whole prior* | fixed and known, so never inferred |
| $\tau=(T,h)$ | input field $T$ with params $\vartheta$; output map $h$ with params $\varphi$ (§10) | $p(\tau)=p(\vartheta)\,p(\varphi)$, independent of everything else |
| $z\in\{0,1\}$ | unrelated / related | $\text{Bern}(\bar\rho)$ |
| $(\theta',\omega')$ | fresh, independent draw, used only when $z=0$ | same prior as $(\theta,\omega)$ |
| $\sigma_A,\sigma_B$ | noise levels | $p(\sigma)$, or fixed |
| $X_A,X_B,n_A,n_B$ | designs and sizes | $\mathcal U[0,1]^d$ and $p(n)$ for now; §3 covers BO trajectories |

**Base function.** $g_{\theta,\omega}:=F\circ\text{BNN}_{\theta,\omega}$. Because $F$ is the prior ECDF, $g(x)$ is marginally $\approx\mathcal U[0,1]$ over (prior × random $x$). This fact matters for C1.

### 1.2 Tasks

$$
f_B=g_{\theta,\omega},\qquad
f_A=\begin{cases}h\circ g_{\theta,\omega}\circ T & z=1\\ h\circ g_{\theta',\omega'}\circ T & z=0\end{cases}
$$

Observation model (convention C2, §1.4):

$$
y_b=g(x_b)+\varepsilon_b,\qquad y_a=h\big(g_\bullet(T(x_a))+\varepsilon_a\big),\qquad y^*=h\big(g_\bullet(T(x^*))+\varepsilon^*\big),
$$

with $\varepsilon\sim\mathcal N(0,\sigma^2)$, and $g_\bullet=g$ or $g'$ depending on $z$.

### 1.3 DAG

```
 ω ──► θ ──► g ──────────────► B
              │
              ▼   (z=1)
   τ=(T,h) ──► f_A ──► A, A_test
              ▲   (z=0)
 ω'──► θ'──► g'
```

$A$ is a **collider** between $g$ (or $g'$) and $\tau$. Every coupling between B and $\tau$ runs through A.

### 1.4 Modelling choices

**C1. The unrelated branch is also passed through $h\circ\cdot\circ T$.**

*What it does.* Under C1 the *marginal* law of $f_A$ is identical for $z=0$ and $z=1$: in both cases it is $h\circ g\circ T$, with $g$ drawn from the BNN prior independently of $\tau$. Consequences:

$$P(z\mid A)=P(z),\qquad P(z\mid B)=P(z).$$

**Neither task alone carries information about relatedness. Only the joint $(A,B)$ does.**

*What goes wrong without it.* Suppose $z=0$ drew a plain $g'$.
- $g'(x)$ is marginally uniform on $[0,1]$, because $F$ is the prior ECDF.
- $h(g)$ under $z=1$ is not uniform. Its range is $[a,a+b]$ and its shape is set by $H$.
- So A's y-histogram alone would reveal $z$. The same goes for T's stretching, which changes the apparent lengthscale.
- A PFN would learn the shortcut "A looks uniform ⇒ unrelated". Real unrelated tasks carry no such signature, so the shortcut fails at deployment, and the in-prior transfer gains would be inflated.

*Why $\tau$ is a "nuisance" under $z=0$.* $\tau$ still exists in the $z=0$ generative path, but nothing it touches is shared with B. It only reshapes the *prior over $f_A$*. In the $z=0$ branch it is integrated against its prior (updated by A alone), and nothing about it transfers. Formally, it lives inside the A-only marginal $p(A)$ (§6).

*Requirement.* The A-only PFN (the null particle) must be trained on the prior of $h\circ g\circ T$, not on plain $g$.

**C2. Noise placement and the box $[0,1]^{d+1}$.** The noise is added in the **base scale, before $h$**. Then $h^{-1}(y_a)=g(T(x_a))+\varepsilon_a$ is an ordinary noisy observation of $g$, which makes B-domain pooling exact (§8.2).

The price is the Jacobian $\lvert(h^{-1})'(y)\rvert$ in every A-likelihood, together with the requirement that $h$ is defined on the support of $g+\varepsilon$, which lies slightly beyond $[0,1]$.

To keep $y\in[0,1]$ strictly, there are two options:
- **(i)** Extend $h$ monotonically outside $[0,1]$ and clip/tolerate a tiny overflow in the bar distribution.
- **(ii)** Put the noise in logit space: $y=\text{sigm}(\text{logit}(\cdot)+\varepsilon)$, which adds a factor $1/(y(1-y))$ to the Jacobian.

The alternative is noise added *after* $h$. Then pooling is only approximate, because $h^{-1}(h(u)+\varepsilon)$ has heteroscedastic, non-Gaussian noise. **[decide; C2 is assumed below]**

**C3. Design ignorability** (§3).

**C4. No residual $r$ for now** (deferred). Consequence: under $z=1$, A carries exact noisy observations of $g\circ T$, so the $\tau$-posterior can be very peaked (§12.4).

---

## 2. Conditional independences, and what they say

The DAG with d-separation gives the following.

| Statement | Reading | Used in |
|---|---|---|
| $B\perp\tau$ (**marginally**) | B alone says nothing about the transform, not even without knowing $g$. $\tau$ is independent of $g$ a priori, and B depends only on $g$. | $p(\tau\mid B)=p(\tau)$ |
| $B\not\perp\tau\mid A$ | Conditioning on the collider A couples them. **B informs $\tau$ only *through* A** (explaining away: given A, knowing $g$ tells you which $T$ fits). | the whole related branch |
| $\tau\perp B\mid g, A$ | Once the function is known, B adds nothing about $\tau$. The path $\tau\to A\leftarrow g\to B$ is blocked at $g$. | route β (§9) |
| $A\perp B\mid g,\tau,z{=}1$ | All coupling between the tasks goes through the shared function. Given it and the transform, the noises are independent. | factorisation of the likelihood |
| $A\perp B\mid z{=}0$ | A's function comes from an *independent* draw $g'$. Even numerically equal weights would be independent draws. | $p(A\mid B,z{=}0)=p(A)$ |
| $B\perp z$ and, under C1, $A\perp z$ | single tasks do not reveal relatedness | §4 |

*Correction to the chat reading:* $B\perp\tau$ holds without conditioning on $(\theta,\omega)$ as well. Conditioning on $g$ matters only once A is also conditioned on.

---

## 3. Design ignorability: when do $X$ and $n$ depend on the latents?

**Uniform designs.** $X$ and $n$ are independent of everything, and $p(X)$ cancels trivially.

**BO trajectories (the realistic HPO case).** $x_t\sim\pi(x_t\mid D_{<t})$ depends on the previously *observed* $y$'s, so $X$ is **not** independent of $g$. The joint density of a trajectory is

$$
p(D\mid g)=\prod_t \pi(x_t\mid D_{<t})\;p(y_t\mid x_t,g).
$$

The $\pi$-factors are functions of the data only, with no latent inside. They are therefore identical in every term of every posterior ratio and **cancel**:

$$p(g\mid D)\propto p(g)\prod_t p(y_t\mid x_t,g),$$

exactly as for a fixed design. The same holds for a stopping rule $n$ that depends only on observed data ("stop when no improvement for 20 steps"). And it holds for A's warm start at B's optimum, *provided B is in the context*.

**When ignorability breaks** (non-ignorable selection, where the policy used information not in the context):
- logs truncated or filtered by value: top-k kept, crashed or diverged configs dropped (a crash depends on the unseen $f$);
- B's run warm-started from tasks we don't see;
- runs merged or subsampled by an unknown rule;
- a budget or $n$ chosen from information outside the data.

In these cases the $x$-density itself carries information about $f$, and the prior has to model the mechanism.

**What remains even when ignorable:** an *amortisation* shift. A PFN trained on uniform designs sees clustered contexts as out of distribution. This is a training-distribution issue, not a Bayes issue.

---

## 4. Root and model posterior

**(E0) Root**

$$
p(y^*\mid x^*,A,B)=P(z{=}0\mid A,B)\,p(y^*\mid x^*,A,z{=}0)+P(z{=}1\mid A,B)\,p(y^*\mid x^*,A,B,z{=}1)
$$

**(E1) Model posterior.** By Bayes' rule on $z$,

$$
P(z\mid A,B)=\frac{p(z)\,p(A,B\mid z)}{\sum_{z'}p(z')\,p(A,B\mid z')},\qquad p(A,B\mid z)=p(B\mid z)\,p(A\mid B,z)=p(B)\,p(A\mid B,z),
$$

since $B\perp z$. The factor $p(B)$ is common and cancels:

$$
P(z\mid A,B)=\frac{p(z)\,p(A\mid B,z)}{\sum_{z'}p(z')\,p(A\mid B,z')},\qquad p(A\mid B,z{=}0)=p(A).
$$

**Log-odds.** Divide the $z=1$ and $z=0$ cases (the denominators cancel) and take the log:

$$
\log\frac{P(z{=}1\mid A,B)}{P(z{=}0\mid A,B)}=\underbrace{\log\frac{\bar\rho}{1-\bar\rho}}_{\text{prior odds}}+\underbrace{\log p(A\mid B,z{=}1)-\log p(A)}_{\text{log Bayes factor}}.
$$

Via the chain rule (§5), the Bayes factor is $\sum_a\big[\log p(y_a\mid x_a,A_{<a},B,z{=}1)-\log p(y_a\mid x_a,A_{<a})\big]$. It is a **sequential predictive comparison**: point by point, did the B-informed model predict A's next point better than the prior did?

---

## 5. The chain rule: why products over $A_{<a}$

A PFN gives **marginal predictives** $p(y\mid x,D)$, not joint densities over several $y$'s. The evidence needs the *joint* $p(y_1,\dots,y_{n}\mid x_{1:n})$. The chain rule of probability converts one into the other:

$$
p(y_{1:n}\mid x_{1:n})=\prod_{a=1}^{n}p(y_a\mid x_a,A_{<a}).
$$

**It is exact in any order.** Let $Z_t:=\int p(g)\prod_{s\le t}p(y_s\mid x_s,g)\,dg$. Then

$$
p(y_t\mid x_t,A_{<t})=\int p(y_t\mid x_t,g)\,p(g\mid A_{<t})\,dg=\frac{Z_t}{Z_{t-1}},
$$

so the product telescopes to $Z_n/Z_0=Z_n=\int p(g)\prod_a p(y_a\mid x_a,g)\,dg$. That expression is symmetric in the points. **Every ordering gives the same number.**

- A PFN's factors break this symmetry slightly. Averaging over a few permutations reduces the error; permutation variance is a diagnostic.
- Under a BO trajectory, the acquisition order is the natural one, and it is the one the SMC filter (§8.4) follows.

**Why the B-only product is wrong.**

$$
\prod_a p(y_a\mid x_a,B,\tau)=\prod_a\int p(y_a\mid g,\tau)\,p(g\mid B)\,dg\;\neq\;\int\prod_a p(y_a\mid g,\tau)\,p(g\mid B)\,dg=p(A\mid B,\tau).
$$

The left side never updates $g$ after seeing $y_1$. Information that A's points share about $g$ (two nearby points both "explained" by the same uncertain bump) is counted once per point, i.e. **double-counted**.

- The gap vanishes only where $p(g\mid B)$ is a point mass at all $T(x_a)$ (dense B, no coverage gap).
- It is largest in coverage gaps, which is exactly where weights matter most.
- **Consequence:** the B-only pseudo-likelihood is acceptable as a *screening* score, never as a weight. The v1 design was biased by using it as a weight.

---

## 6. Unrelated branch ($z=0$)

**(E2.1) Predictive**

$$
p(y^*\mid x^*,A,z{=}0)=\sum_{\omega'}P(\omega'\mid A)\iint p(y^*\mid x^*,\theta',\omega',\tau)\,p(\theta',\tau\mid A,\omega')\,d\theta'\,d\tau,
$$

with

$$
p(\theta',\tau\mid A,\omega')\propto p(\theta'\mid\omega')\,p(\tau)\,p(A\mid\theta',\omega',\tau),\qquad P(\omega'\mid A)\propto p(\omega')\,p(A\mid\omega').
$$

**(E2.2) Evidence**

$$
p(A)=\sum_{\omega'}p(\omega')\iint p(A\mid\theta',\omega',\tau)\,p(\theta'\mid\omega')\,p(\tau)\,d\theta'd\tau=\prod_a p(y_a\mid x_a,A_{<a}).
$$

In function space this is the PPD and evidence of a single-task problem under the prior $\text{Law}(h\circ g\circ T)$. **This is exactly the A-only PFN trained under C1.** No search is needed, and $\tau$ is marginalised implicitly by amortisation.

---

## 7. Related branch ($z=1$): master posterior and collapse to function space

### 7.1 Master posterior

$$
p(\theta,\omega,\tau\mid A,B)\;\propto\;\underbrace{p(\omega)\,p(\theta\mid\omega)\,p(B\mid\theta,\omega)}_{\propto\;p(\theta,\omega\mid B)}\;p(\tau)\;p(A\mid\theta,\omega,\tau).
$$

**(E3) Predictive**

$$
p(y^*\mid x^*,A,B,z{=}1)=\sum_\omega\iint p(y^*\mid x^*,\theta,\omega,\tau)\,p(\theta,\omega,\tau\mid A,B)\,d\theta\,d\tau.
$$

### 7.2 Collapse to the functional

Every likelihood in the tree depends on $(\theta,\omega)$ **only through $g=g_{\theta,\omega}$**. Let $p(g)$ denote the push-forward of $p(\omega)p(\theta\mid\omega)$. Then for any integrand of the form $\phi(g_{\theta,\omega})$,

$$
\sum_\omega\int \phi(g_{\theta,\omega})\,p(\theta,\omega\mid\cdot)\,d\theta=\int\phi(g)\,p(g\mid\cdot)\,dg,\qquad p(g\mid B)\propto p(g)\,p(B\mid g).
$$

- All weight-space multimodality disappears: unit permutations, sign symmetries, and the *affine absorption* $g_\theta(Mx+c)=g_{\theta'}(x)$ with $W_1'=W_1M$, $b_1'=b_1+W_1c$.
- An affine $T$ is unidentifiable against $\theta$ in weight space. In function space it is identifiable, because B pins $g$ itself.
- $\omega$ survives only as part of what $p(g)$ is (a mixture over capacities), and inside $p(g\mid B)$ as an implicit complexity posterior.
- **$p(g\mid B)$ is the single object we need from B, and it is what the frozen B-PFN amortises**, as marginals $Q(\cdot\mid u,B)$.

From here on:

$$
p(\tau,g\mid A,B)\propto p(g\mid B)\,p(\tau)\,p(A\mid g,\tau),\qquad p(A\mid g,\tau)=\prod_a \mathcal N\big(h^{-1}(y_a);g(T(x_a)),\sigma_A^2\big)\,\big|(h^{-1})'(y_a)\big| .
$$

There are two orders in which to factor this.

---

## 8. Route α: $\tau$ outermost ("register A against B's posterior")

### 8.1 The three equations

**(E3.1) Transform posterior**

$$
p(\tau\mid A,B)\propto p(\tau)\,p(A\mid B,\tau)
$$

**(E3.2) Evidence**, exact in any order (§5):

$$
p(A\mid B,\tau)=\int p(A\mid g,\tau)\,p(g\mid B)\,dg=\prod_a p(y_a\mid x_a,A_{<a},B,\tau)
$$

**(E3.3) Conditional predictive** at a query $x^*$ of $A_{\text{test}}$:

$$
p(y^*\mid x^*,A,B,\tau)=\int p(y^*\mid x^*,g,\tau)\,p(g\mid A,B,\tau)\,dg,
$$

where

$$
p(g\mid A,B,\tau)\propto p(g\mid B)\,p(A\mid g,\tau).
$$

### 8.2 Pooling: given $\tau$, A *is* B-domain data

Under C2 (noise before $h$, $r=0$, $h$ invertible),

$$
\tilde y_a:=h^{-1}(y_a)=g(T(x_a))+\varepsilon_a .
$$

So the transported set

$$
\tilde A:=A_{\text{in}B}=\{(T(x_a),\tilde y_a)\}
$$

is an ordinary noisy dataset on $g$, with the same observation model as B (if $\sigma_A=\sigma_B$; otherwise the B-PFN needs per-token noise levels). Hence:

$$
p(g\mid A,B,\tau)=p(g\mid B\cup\tilde A).
$$

With $Q(\cdot\mid u,D)$ the B-domain PPD (the B-PFN), the two quantities become:

**Predictive**, a push-forward through $h$:

$$
p(y^*\mid x^*,A,B,\tau)=Q\big(h^{-1}(y^*)\mid T(x^*),B\cup\tilde A\big)\,\big|(h^{-1})'(y^*)\big|=\big[h_\#\,Q(\cdot\mid T(x^*),B\cup\tilde A)\big](y^*).
$$

For a bar distribution, the push-forward is exact and needs no explicit Jacobian. Map the bucket borders $\beta_k\mapsto h(\beta_k)$ and keep each bucket's mass.

**Evidence factor**, one PFN call:

$$
\ell_a(\tau):=p(y_a\mid x_a,A_{<a},B,\tau)=Q\big(\tilde y_a\mid T(x_a),B\cup\tilde A_{<a}\big)\,\big|(h^{-1})'(y_a)\big|.
$$

- The Jacobian is mandatory, because the evidence must be a density in A's scale. Without it, a compressive $h$ (small $b$) is rewarded spuriously.
- $T$ needs no Jacobian, since the $x$'s are conditioned on, not modelled.

**Cost.** The context is $B\cup\tilde A_{<a}$ and is *particle-specific*, because $\tilde A$ depends on $\tau$. In practice, freeze B's K/V cache and let the $\le n_A$ transported tokens attend to it one-way. The error of this approximation is the measured ΔKV (0.01–0.04 nats).

### 8.3 Importance sampling (reference estimator; consistent as $M\to\infty$)

Draw $\tau_m\sim q(\tau)$ for $m=1..M$.

$$
\log w_m=\log p(\tau_m)-\log q(\tau_m)+\sum_a\log\ell_a(\tau_m),\qquad
\hat Z_1=\tfrac1M\textstyle\sum_m w_m\ \ (\text{unbiased for }p(A\mid B,z{=}1)),
$$

$$
Z_0=p(A)=\prod_a p_{\text{A-only}}(y_a\mid x_a,A_{<a}),\qquad
\hat P_1=\frac{\bar\rho\hat Z_1}{\bar\rho\hat Z_1+(1-\bar\rho)Z_0},
$$

$$
\hat p(y^*\mid x^*,A,B)=(1-\hat P_1)\,p_{\text{A-only}}(y^*\mid x^*,A)+\hat P_1\sum_m\bar w_m\,\big[h_{m\#}Q(\cdot\mid T_m(x^*),B\cup\tilde A^{(m)})\big](y^*),
$$

where $\bar w_m=w_m/\sum w$.

- With a prior proposal $q=p(\tau)$, the prior ratio drops out.
- The bias is $O(1/M)$, from self-normalisation.
- Diagnostic: $\text{ESS}=(\sum w)^2/\sum w^2$.

### 8.4 SMC: the chain rule *is* the weight sequence

Process A's points in acquisition order, $t=1..n_A$, with particles $\tau_k$ and normalised weights $W_k^{(t-1)}$.

1. **Reweight.** The incremental weight is $\alpha_k^t=\ell_t(\tau_k)$, which is exactly one chain-rule factor. Set $W_k^{(t)}\propto W_k^{(t-1)}\alpha_k^t$.
2. **Evidence.**
   - Related branch: $\hat Z_1^{t}=\hat Z_1^{t-1}\sum_kW_k^{(t-1)}\alpha_k^t$, the standard unbiased SMC evidence.
   - Null branch, tracked in closed form: $Z_0^{t}=Z_0^{t-1}\,p_{\text{A-only}}(y_t\mid x_t,A_{<t})$.
   - $\hat P_1^t$ follows as in §8.3.
3. **Resample** if $\text{ESS}<K/2$.
4. **Rejuvenate** with a kernel invariant for $\pi_t(\tau)\propto p(\tau)\prod_{s\le t}\ell_s(\tau)$.
   - MALA on $\vartheta$, with $\nabla\log\pi_t$ by autograd through $Q\circ T_\vartheta$. Each step re-evaluates $t$ factors per particle.
   - Learned moves are allowed only if MH-corrected.

This matches deployment: A grows by one point per BO step, B is static, and particles carry over.

---

## 9. Route β: function outermost ("align each candidate function")

### 9.1 Derivation

Start from $p(\tau,g\mid A,B)\propto p(g)\,p(B\mid g)\,p(\tau)\,p(A\mid g,\tau)$.

**(E3.4) Marginal over $g$.** Integrate out $\tau$:

$$
p(g\mid A,B)\propto p(g)\,p(B\mid g)\int p(\tau)\,p(A\mid g,\tau)\,d\tau\;\propto\;p(g\mid B)\;p(A\mid g),\qquad p(A\mid g):=\int p(\tau)\,p(A\mid g,\tau)\,d\tau .
$$

**(E3.5) Conditional over $\tau$.** Divide the joint by the marginal, i.e. read the joint as a function of $\tau$ for fixed $g$:

$$
p(\tau\mid g,A,B)=\frac{p(g)p(B\mid g)\,p(\tau)p(A\mid g,\tau)}{p(g)p(B\mid g)\,p(A\mid g)}=\frac{p(\tau)\,p(A\mid g,\tau)}{p(A\mid g)}=p(\tau\mid g,A).
$$

The factor $p(B\mid g)$ does not contain $\tau$ and cancels. This is the algebraic form of the d-separation $\tau\perp B\mid g,A$ (§2). **Given a candidate function, aligning A to it is pure registration: $n_A$ points, a closed-form likelihood, no B.**

**Predictive**

$$
p(y^*\mid x^*,A,B,z{=}1)=\iint p(y^*\mid x^*,g,\tau)\,p(\tau\mid g,A)\,p(g\mid A,B)\,d\tau\,dg,
$$

with

$$
p(y^*\mid x^*,g,\tau)=\mathcal N\big(h^{-1}(y^*);g(T(x^*)),\sigma_A^2\big)\,\big|(h^{-1})'(y^*)\big|.
$$

### 9.2 Estimator: nested IS

Use the proposal $p(g\mid B)\,q(\tau\mid g)$.

$$
g_j\sim p(g\mid B),\quad \tau_{jm}\sim q(\tau\mid g_j),\quad
w_{jm}=\frac{p(\tau_{jm})\,p(A\mid g_j,\tau_{jm})}{q(\tau_{jm}\mid g_j)},\quad \hat Z_1=\tfrac1{JM}\textstyle\sum_{jm}w_{jm}.
$$

The outer weights are pure likelihood-weighting, because $g_j$ is drawn from its B-posterior.

**Advantages over α:**
- $p(A\mid g,\tau)$ is a **closed-form product of Gaussians**, with no chain rule and no order dependence.
- No PFN runs inside the registration loop, and gradients in $\vartheta$ are cheap.
- When B is dense, $p(g\mid B)$ is narrow on the window, so small $J$ suffices.

**Price:** it needs **joint function samples** $g_j$, which a PFN does not give directly (it gives marginals). The options:
- **(i)** Autoregressive sampling with the B-PFN on a grid, then interpolation. Fine for $d\le2$.
- **(ii)** *Lazy* sampling: draw $g_j$ at a new location conditioned on B plus $g_j$'s previously sampled values. This is exact, but not differentiable in $T$.
- **(iii)** Weight-space BNN posterior samples. This brings back exactly the $\theta$-multimodality we want to avoid; usable only as a check.

Within the window, a plug-in of the posterior mean with $J=1$ is nearly exact for dense B. The spread of $g$ matters in coverage gaps.

---

## 10. Transform parametrisation (names), and Rao–Blackwellisation

### 10.1 Input field $T_\vartheta$, with $\vartheta=(\vartheta_{\text{sep}},\vartheta_{\text{flow}})$

$$
T_\vartheta=T_{\text{flow}}\circ T_{\text{sep}}.
$$

**$T_{\text{sep}}$: separable, per-dimension monotone map** ("where does the optimum move along each axis, and how stretched is each axis").
- $T_{\text{sep},i}(x_i)=\text{Kumaraswamy/Beta-CDF}(x_i;\alpha_i,\beta_i)$, or a few-knot RQ spline.
- Parameters: $\vartheta_{\text{sep}}=\{(\alpha_i,\beta_i)\}_{i\le d}$, about $2d$ numbers.
- It fixes the box: $0\mapsto0$, $1\mapsto1$.

**$T_{\text{flow}}$: non-separable residual distortion** (interactions, local warps).
- Flow of a stationary velocity field, $T_{\text{flow}}=\exp(v)$, with $v_i(x)=x_i(1-x_i)\,w_i(x)$ and $w=\sum_j\vartheta_{\text{flow},j}\beta_j(x)$.
- Prior: $\vartheta_{\text{flow}}\sim\mathcal N(0,\Lambda^{-1})$, with $\Lambda$ growing with frequency. The decay of $\Lambda$ is the "warp complexity / degrees-of-freedom" knob.
- Overlap with $T_{\text{sep}}$: low-frequency separable flows duplicate $T_{\text{sep}}$. Remove the separable lowest band from the flow basis, or accept the ridge.

**Option, if T must not preserve the box** (§12): $T_{\text{aff}}(x)=s\odot x+c$ maps into $U\supset[0,1]^d$, giving $\vartheta_{\text{aff}}=(s,c)$.

### 10.2 Output map $h_\varphi$, with $\varphi=(a,b,\kappa)$

$$
h(u)=a+b\,H_\kappa(u),\qquad a\ge0,\ b>0,\ a+b\le1 .
$$

- $a$ = **floor** (offset).
- $b$ = **dynamic range**; harder tasks compress the range.
- $H_\kappa$ = **shape**, a monotone bijection of $[0,1]$, e.g. a Beta-CDF with $\kappa=(\alpha_h,\beta_h)$.

The image $h([0,1])=[a,a+b]\subseteq[0,1]$ respects the box.

### 10.3 Rao–Blackwellising $\varphi$

**Principle.** If part of the latent can be integrated *exactly* conditional on the rest, then sampling only the rest and integrating that part analytically or numerically never increases estimator variance:

$$
\text{Var}\big(E[\cdot\mid\vartheta]\big)\le\text{Var}(\cdot)\quad\text{(law of total variance).}
$$

**Applied here.** Given $\vartheta$ (and $g_j$ in route β, or the transported marginals in route α), A's likelihood as a function of $\varphi$ is a low-dimensional closed form. Replace sampling $\varphi$ by quadrature:

$$
p(A\mid g,\vartheta)=\int p(\varphi)\,p(A\mid g,\vartheta,\varphi)\,d\varphi\quad(\text{2–4-D grid, e.g. }32^2\text{ over }(a,b)\text{ per }\kappa),
$$

$$
p(y^*\mid x^*,A,g,\vartheta)=\int p(y^*\mid x^*,g,\vartheta,\varphi)\,p(\varphi\mid g,\vartheta,A)\,d\varphi .
$$

- **Special case, fully conjugate.** With noise after $h$ and $H=\text{id}$, $y_a=a+b\,u_a+\varepsilon$ is linear-Gaussian in $(a,b)$, so the integral is analytic (up to the box constraint). Under C2 the noise scales with $b$, so use the grid.
- **What it buys.**
  - IS/SMC only has to explore $\vartheta$, so the dimension drops by $\dim\varphi$.
  - The $T$–$h$ trade-off ridge (each A point is one scalar shared by $T$ and $h$) is integrated exactly per particle instead of having to be covered by particles.
  - $\sigma_A$ can be Rao–Blackwellised the same way, since it trades off against $b$.
- **Route α caveat.** $p(A\mid B,\vartheta,\varphi)$ goes through the chain rule with PFN calls, so the grid multiplies PFN cost. With the frozen B-K/V, the $\varphi$-grid enters only through $\tilde y_a=h_\varphi^{-1}(y_a)$, which batches. Route β has no such cost.

---

## 11. How $\vartheta$ is estimated (inventory)

All options target $p(\vartheta\mid A,B)$ (route α), or $p(\vartheta\mid g_j,A)$ per function (route β), with $\varphi$ Rao–Blackwellised.

| Method | Mechanism | Unbiased? | Role |
|---|---|---|---|
| Prior IS | §8.3 / §9.2 with $q=p$ | consistent | reference; V0 |
| Sample-and-screen | rank prior draws by a cheap score (B-only product, or the β-likelihood at the posterior-mean $g$), keep the top $K$ **with their IS weights** | bias = dropped mass (estimable) | cheap starts |
| SMC + MALA | §8.4; gradients via autograd through $Q\circ T_\vartheta$ (α) or the interpolant $g_j\circ T_\vartheta$ (β) | consistent | deployment, along BO |
| Gauss–Newton / Laplace | linearise $\tilde y_a\approx\mu_B(T(x_a))$; each point constrains only $\nabla\mu_B^\top\delta T(x_a)$; solve the regularised normal equations with $\Lambda$; weight modes by $p\cdot\text{lik}\cdot\lvert H\rvert^{-1/2}$ | biased (Gaussian modes) | cheapest; preconditioner; template for the learned optimiser |
| Learned optimiser | amortised Gauss–Newton (evidence tokens → field) | valid only as an MH proposal or inside Laplace with $\lvert H\rvert$ | speed |
| NPE proposal $q_\lambda(\vartheta\mid A,B)$ | trained on the true $\vartheta$ (LUPI); IS weights $p/q_\lambda$ keep $\hat p$ unbiased | variance only | **user decision** (conflicts with "no warp-parameter heads") |

The degrees-of-freedom rule: each A point gives **one scalar** constraint. While $n_A\lesssim\dim_{\text{eff}}(\vartheta)$, the posterior is prior-dominated, which is correct. The decay of $\Lambda$ sets $\dim_{\text{eff}}$.

---

## 12. Window and coverage: implications

### 12.1 Box-bijective $T$ (current decision)

$T$ never leaves $[0,1]^d$, so out-of-window queries cannot happen. The costs:

- **Boundary pinning and leakage.** $T(0)=0$ and $T(1)=1$ per axis. Near the edges $T\approx$ identity in the separable part, so A ≈ $h\circ$B there, and **the model can transfer confidently near edges for a structural reason, not because of evidence**. This is the bounding-box leakage of the context doc §3.
- **Shift artefacts.** Moving the optimum is only possible by stretching one side and squeezing the other. The prior then couples "optimum moved" with "lengthscale changed asymmetrically".
- **HPO realism.** The search space really is shared across datasets, so edge correspondence is partly realistic (lr = 1e-5 means the same in both). Whether the response is pinned at the edges is an empirical question (§15, X5).

### 12.2 $T$ into $U\supset[0,1]^d$

Shifts are realistic, but $T(x)$ can leave the region where B has data. Then:

- **Predictive.** $p(g(T(x^*))\mid B)\to$ the prior marginal there, so the related branch predicts $h_\#(\text{prior})$ updated by nearby A points, i.e. roughly A-only. This is correct, graceful degradation, *if* the B-PFN was trained with queries outside its context region (extrapolation). Otherwise it is out of distribution and possibly overconfident.
- **Evidence.** A points whose $T(x_a)$ fall out of window have flat factors in $\tau$, so they carry almost no information about the transform. ESS stays high, but the posterior stays broad.
- **Gradients.** $\nabla\mu_B\approx0$ out of window, so Gauss–Newton and MALA lose signal and moves become prior-driven.
- **Identifiability.** A transform family that can push A's region out of B's support has a cheap "explain-away" mode: map the awkward A points out of the window, where anything goes. The prior over $(s,c)$ must make that costly. Otherwise the related branch mimics the null branch with spurious weight.

### 12.3 Coverage gaps (also under bijective $T$)

With BO-biased B, much of the window is sparse. The bullets of §12.2 apply wherever $T(x)$ lands in a sparse region. This is also where the B-only product double-counts most (§5).

### 12.4 Peakiness without a residual (C4)

With $r=0$ and small $\sigma_A$, $p(A\mid B,\tau)$ concentrates on thin ridges in $\vartheta$. Prior IS degenerates, and the only way to discredit B is all-or-nothing through $z$. Track ESS vs. $(n_A,\sigma_A)$ before deciding on $r$.

---

## 13. Leaves

| Leaf | Expression | Status |
|---|---|---|
| B likelihood | $\prod_b\mathcal N(y_b;g(x_b),\sigma_B^2)$ | closed form; amortised into $p(g\mid B)$ |
| A likelihood | $\prod_a\mathcal N(h^{-1}(y_a);g(T(x_a)),\sigma_A^2)\lvert(h^{-1})'(y_a)\rvert$ | closed form given $(g,\tau)$ |
| Query | $\mathcal N(h^{-1}(y^*);g(T(x^*)),\sigma_A^2)\lvert(h^{-1})'(y^*)\rvert$ | closed form given $(g,\tau)$ |
| Priors | $p(\omega),p(\theta\mid\omega)$ (→ $p(g)$), $p(\vartheta),p(\varphi),\bar\rho,p(\sigma)$ | known (our prior) |
| $p(g\mid B)$ | marginals $Q(\cdot\mid u,B)$ | **the only learned component strictly needed** (frozen B-PFN) |
| $p(A)$, $p(y^*\mid x^*,A)$ | A-only PFN, trained on $h\circ g\circ T$ (C1) | learned |

---

## 14. Corrections to the morning docs and to the first chat answer

1. $B\perp\tau$ holds **marginally**, not only given $(\theta,\omega)$. B informs $\tau$ only through the collider A.
2. B-domain pooling (route α, §8.2) is exact only under C2 (noise before $h$) **and** with matching noise levels or per-token noise in the B-PFN.
3. Route β needs **joint** function samples; PFN marginals are not enough (§9.2).
4. Rao–Blackwellisation of $\varphi$ is analytic only in the conjugate special case. In general it is low-dimensional quadrature (§10.3).
5. The B-only product is a screening score only (§5). The v1 weights were biased by it; v2's prequential $R_\psi$ weights avoid it.
6. The A-only baseline and null particle must be trained on $\text{Law}(h\circ g\circ T)$ (C1). Otherwise $z$ leaks and transfer gains are inflated.
7. Box-bijective $T$ introduces edge pinning (§12.1); this is not neutral.

---

## 15. Next experiments (cheapest first)

- **X1. z-leak test.** Train a small classifier to predict $z$ from A alone, and separately from B alone. **Pass:** chance level, AUC ≤ 0.52. Run it with and without C1 to show the leak is real.
- **X2. Chain rule vs. B-only product.** $d=1$, B-PFN pooling with frozen K/V.
  - Measure the permutation variance of $\sum_a\log\ell_a$.
  - Measure the gap $\log p(A\mid B,\tau)-\sum_a\log p(y_a\mid x_a,B,\tau)$ as a function of B's posterior std at $T(x_a)$.
  - Expected: the gap is about 0 for dense coverage and grows in gaps.
- **X3. α vs. β vs. grid truth ($d=1$).** Route β with autoregressive function samples on a 256-grid. **Pass:** the predictives agree within 0.02 nats, and both match grid Bayes.
- **X4. Rao–Blackwellised $\varphi$.** IS over $(\vartheta,\varphi)$ vs. IS over $\vartheta$ plus a $\varphi$-grid, at equal compute. Report ESS, NLL, and $P(z\mid A,B)$ calibration.
- **X5. Window decision.** Box-bijective $T_{\text{sep}}$ vs. $T_{\text{aff}}$ into $U$.
  - Compare the NLL of the transfer model vs. A-only, stratified by distance of $x^*$ to the box edge. An edge-only gain under bijective $T$ is a leakage signature.
  - Also check on real HPO pairs whether responses agree more near the edges than in the interior. This decides whether pinning is realistic.
- **X6. Peakiness.** ESS of prior IS vs. $(n_A,\sigma_A)$ under C4. This sets the case for $r$ purely on estimator grounds.
