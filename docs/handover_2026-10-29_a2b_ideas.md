# Handover, 2026-10-29: A→B architecture state, for the point-cloud-registration / unbalanced-Sinkhorn session

*Written by the A→B session (issues #16, #18). The user asked that session to fold these ideas into one concise
architecture and to flag conflicts with its registration / unbalanced-Sinkhorn ideas. It was not running when this was
written, so this file is the message to hand it.*

**Request.** Write a one-page architecture summary covering:
- the modules and their data flow;
- what is frozen and what is learned;
- the losses (LUPI only);
- where A's evidence enters: the weights, the null particle, the cache residual, the final PFN;
- the open design choices, and which experiment decides each.

Flag any conflicts with the point-cloud-registration / unbalanced-Sinkhorn ideas.

## Sources

- `docs/design_2026-10-28_A_to_B_architecture.md`: the user's design note (§1–9) plus appendices (Q&A, deviations).
- `notebooks/08_2026-10-28_ladder.ipynb` (S0/S1 results and a Findings section; HTML in `reports/figures/ladder/`) and
  log `research_log/entries/2026-10-28_ladder.md` §1.
- `notebooks/09_2026-10-29_a2b.ipynb` (S3/S4 design; running) and code in `src/meta_pfn/experimental/a2b/`.
- Branches: `exp/16-ladder-s0s1` → `exp/18-a2b-s3s4` (stacked on `exp/14-fieldonly`).

## Core design (the note)

- **Direction:** move A's train points and queries *into* B's domain; never move B.
- **Reading B:** B is read through the frozen single-task PFN (`pfn_large`). B's per-layer K/V are cached once, and the
  query path $Q_B(x)$ is differentiable in $x$.
- **Hypotheses:** $K$ particles hold hypotheses $(T_k, h_k)$, under $f_A = h \circ f_B \circ T + r$.
- **Weights:** explicit Bayes weights, $\log w_k = \pi(P_k) + \sum_a \log p_k(y_a)$, with $p_k$ = $h_k$ pushed through
  $Q_B(T_k(x_a))$.
- **Learned moves:**
  - a Perceiver summary $Z_B$;
  - an A-encoder;
  - DETR-style particle init;
  - $R$ weight-tied evidence blocks (likelihood, PIT, B's hidden state, a lookup into $Z_B$).
- **Prediction:** a Bayes mixture over the particles plus a null particle (A-only PFN), then a final PFN on
  $[A, A_{\text{test}}]$ with zero-initialised injections.

## Results so far

Frozen PFN only, no training; $N_B = 80$ because `pfn_large`'s context is $n \le 100$.
- **S0 PASS.** B queried at the true $T(x^*)$ beats A-only by 2.8–5.4 nats/query at $n_A \le 2$. It equals both privileged
  teachers to within 0.07.
- **S1 GO.** A grid posterior over the shift closes 95% ($d = 1$) and 88% ($d = 2$) of the A-only → oracle gap at
  $n_A = 5$.
- **Multimodality:** unimodal posteriors are 56–84% at $n_A \le 2$ and 97–98% at $n_A = 5$.
- **Search, not likelihood, is the bottleneck.** Gradient descent from $s = 0$ finds the MAP in 43–61% of pairs; the best
  of 8 starts finds it in 89–99%. The landscape is rugged.
- **Partial identifiability at $d = 2$:** ridges along flat directions of $f$, while predictions stay fine.
- **Bayes-null** (model comparison against the A-only PFN) reaches A-only within 0.38 nats at $n_A = 1$, and within 0.03
  from $n_A = 5$, on unrelated pairs.
- **Pseudo-likelihood:** A's points are treated as independent given B, because test tokens never attend to each other.
  - It costs ≤ 0.04 nats against the exact chain rule here.
  - It double-counts evidence where B is uncertain.
  - Its predictive cannot use A.

## The user's firm decisions

- Privileged information as LUPI only: teacher distillation, or latent targets. No warp-parameter heads, and no
  location or function-space supervision on particles.
- No β-tempering. No retrained causal PFN.
- $T$ is a full-box monotone bijection: no sub-windows.
- B must be discounted as evidence against it accumulates. At the hypothesis level this is the null particle; locally it
  is a correction of B's cache.
- Local correction: a **low-rank K/V residual** on B's cache, applied only to the **top-k B tokens by attention** from
  the transported A tokens, zero-initialised.

## Ideas to integrate

1. **Cache-residual target.**
   - Teacher: $\Delta K^*_l, \Delta V^*_l = KV_l(B \mid B \cup A_{\text{in}B}) - KV_l(B \mid B)$ for layers $l \ge 1$ (layer 0
     is unchanged).
   - Student: the low-rank top-k residual computed from $(T_k(x_a), h_k^{-1}(y_a))$, plus the transported A tokens as
     extra keys.
   - Pre-check with the frozen PFN alone: the share of $\Delta K^*$'s energy in the top-k rows, and its effective rank.
     This sets $k$ and $r$.
2. **Latent LUPI (JEPA).** Pull the particles' hidden states at $T_k(x^*)$ toward the frozen PFN's hidden state at $T(x^*)$
   under the context $B \cup A_{\text{in}B}$.
   - Under a multimodal posterior an MSE averages. Put the weights inside the log, as a mixture critic:
     $-\log \sum_k w_k\, e^{s(h_k, h^*)/\tau}$, with contrastive negatives in the InfoNCE version.
   - Hard negatives: teacher latents under other prior-sampled transforms for the same pair.
3. **Output LUPI:** Rao–Blackwell distillation from PFN($B \cup A_{\text{in}B}$) on related pairs, and from the A-only
   PFN on unrelated pairs.
4. **With a null particle,** add the particles-only NLL on related pairs. Otherwise the null dominates at init and the
   particles get no gradient (gradient norm ~1e-20 in a smoke run).

## Running (ulysses, notebook 09)

- **S3** ($h = \mathrm{id}$, monotone warp):
  - $d = 1$: `full`, `r0`, `k1`, `nosup`, `context`, `jepa`;
  - $d = 2$: `full`, `r0`, `context`.
- **S4** (affine and monotone $h$): after S3.
