# AIR pilot 1: review and pilot-3 plan (2026-11-03)

*This reviews the results of pilot 1 (`results/air/`, notebook 10, run log `2026-11-02_air.md`) and Claude Code's implementation summary, against the spec `a2b_registration_proposal_2026-11-02.md`. It gives the changes for the next pre-registration (pilot 3). Pilot 2 (ECDF normalisation of g) is running and should finish **unchanged**; everything below goes into pilot 3.*

---

## 1. What pilot 1 established

- **Transfer is real.** At J = 2, n_A = 1, the NLL is −0.38 for A-only, −1.04 with K = 32, −1.30 with K = 128, and −1.48 for G-ref.
- **The decoder is close to the floor.** Its predictive plus finite-B effects sit only 0.07–0.09 above G-ref at every n_A, and the decoder's evidence adds another 0.01–0.08. Most of what remains is proposal sharpness and finite K, not the decoder.
- **The gap to IS-exact does not grow from J = 2 to J = 16** at K = 2048, for the n_A values where the reference can be trusted. So the central scaling claim is not contradicted, but the K = 32 test was too weak to show it.

**The verdict "NO-GO as pre-registered, idea not refuted" is correct.** But several checks failed for reasons that lie in the spec, not in the method (§2).

## 2. Spec errors (mine) that made checks fail for the wrong reason

1. **V1.2 and V3 compare against the dense-B posterior.**
   - G-ref knows g exactly. The decoder and q target the posterior given **finite, noisy** B.
   - So the KL of either to the grid posterior can't go to 0, even for perfect networks. That is consistent with "noise-free B removes 21–43 % of the evidence KL".
   - **Fix:**
     - replace these KLs with calibration checks that need no exact finite-B posterior: SBC of the decoder-implied grid posterior $\propto p(\vartheta)\,\hat p(A\mid B,\vartheta)$, and SBC of q (already passing);
     - measure q against the **decoder-implied** posterior, which is what IS actually needs: report $\mathrm{KL}(\hat\pi_D\,\|\,q)$ on the J = 2 grid.
2. **Thresholds that assumed a 0.1-nat finite-B offset** (V2 ≤ 0.12, V4 total gap ≤ 0.15) were guesses.
   - The measured decoder-plus-finite-B offset is 0.07–0.09 for the predictive, and up to about 0.16 including the evidence.
   - **Fix:** state proposal-side criteria relative to IS-ref, not G-ref, and use G-ref only to report the total.
3. **V1.3 permutation spread** is measured on the sum over n_A points, so it naturally grows with n_A. ESS was also unchanged with 8 orders.
   - **Fix:** a per-point criterion (≤ 0.02 nats per point) and a functional criterion (ESS change < 5 %).
4. **K = 32 was too small for the predictive at n_A = 1** (§3).
5. **V5 at K = 32 measures ESS at an arbitrary K.** ESS/K naturally falls as K grows (0.054 at K = 2048).
   - **Fix:** report ESS/K together with the NLL gap to IS-ref as a function of K ∈ {8, 32, 128, 512, 2048}, and pre-register on the NLL gap at K = 128.

## 3. The real structural finding: sharp per-sample predictives

"Exact draws with exact predictives lose 0.66 at n_A = 1 with K = 32."

- Given θ, the predictive $p(y^*\mid x^*,A,B,\theta)$ is almost as sharp as the noise.
- At n_A = 1 the posterior over θ is broad, so the posterior predictive (PPD) is broad.
- A mixture of 32 sharp components leaves holes, and the log score punishes holes hard.
- This is a property of the estimator, not of the networks: **Monte Carlo variance in log space.**

**Principled fix: Rao–Blackwellise through an auxiliary noisy view of the transform.** Augment the generative model with

$$
\theta_c\mid\theta\sim\mathcal N(\theta,\Sigma),\qquad \Sigma\ \text{fixed and known (in }\tilde\theta\text{ units)}.
$$

$\theta_c$ is a new latent that the data never see. So *exactly*:

$$
p(y^*\mid x^*,A,B)=\int p(\theta_c\mid A,B)\,p(y^*\mid x^*,A,B,\theta_c)\,d\theta_c,\qquad
p(\theta_c\mid A,B)\propto \mathcal N(\theta_c;0,I+\Sigma)\;p(A\mid B,\theta_c).
$$

| Piece | How it is learned / computed | Why it is exact |
|---|---|---|
| decoder $D_\psi(\cdot\mid A,B_{\text{in}A}(\theta_c),\Sigma)$ | train on prior data generated with the **true** θ, while the decoder is given $\theta_c=\theta+\text{noise}$ (B moved by $T_{\theta_c}^{-1}$) and Σ as an input token | it learns the conditional given $\theta_c$, which is a proper conditional of the augmented joint. No bias |
| evidence | $p(A\mid B,\theta_c)$ by the same causal mask | same |
| proposal | NPE with target $\theta_c$ (draw noise per training example): $q_\phi(\theta_c\mid A,B,\Sigma)$ | forward KL to the posterior of $\theta_c$ |
| weights | $w=\mathcal N(\theta_c;0,I+\Sigma)\,\hat p(A\mid B,\theta_c)/q(\theta_c)$ | ordinary IS |

**What Σ does:**
- Each per-sample predictive now averages over a Σ-neighbourhood of transforms, so it is smoother. Fewer samples are needed, and the posterior over $\theta_c$ is wider, so q is easier to fit.
- **Σ = 0** is the current AIR.
- **Σ → ∞** makes $\theta_c$ uninformative, so the decoder becomes a **plain joint PFN on [A, B]**: no registration latent, no IS.

So **a single sweep over Σ answers the question "is the explicit registration latent worth anything?"** If the best NLL at fixed compute (K) lies at an interior Σ, below both endpoints, the latent earns its place. If Σ = ∞ wins, a direct PFN is enough.

**Side benefit.** In training the decoder now sees transforms that are off by a known amount. That reduces the "over-rates wrong transforms" risk from V1.2.

## 4. Implementation fixes

1. **Gradient clipping.** The proposal's gradient norms were 6–36 against a clip of 1.0, so every step was shrunk 6–36×. That is the most likely cause of the underfit q.
   - Use clip 20 for q, or clip with an adaptive per-step quantile.
   - Train to a real plateau: stop when the validation NLL improves by < 0.01 over 5k steps. Report the curves.
2. **Bar borders.** The uniform 400-bin borders are fine. Note in the log that NLL values are **not comparable** to runs 1–2. Within AIR they are consistent.
3. **32 training queries.** Harmless. Record it as a deviation.
4. **Seeds.** Use 3 seeds for every headline number. Pilot 1 had one.
5. **Sampling.** Try quasi-Monte Carlo: Sobol points pushed through the flow's base distribution, compared with i.i.d. draws, at equal K. It is cheap and directly attacks the "holes" problem.

## 5. Missing baselines, under the same prior and the same bins

- **Direct joint PFN on [A, B] → y\*.** Same parameter budget as decoder + proposal (≈ 1.8M), same number of steps. This is the Σ = ∞ endpoint, but train it separately too as a sanity check. **It is the most important missing control:** AIR is only worth its complexity if it beats this.
- **Teacher** (decoder with the true θ) and **oracle**, as before.

## 6. Pilot-3 pre-registration (proposed)

**Scope:** d = 1, H0 (h = id), Z0 (always related), ECDF prior if pilot 2 shows no regression, J ∈ {2, 16}, n_A ∈ {1, 2, 3, 5, 10}, 256 tasks per cell, 3 seeds.

| ID | Check | Pass |
|---|---|---|
| P3.1 | q trained to plateau (fixed clipping) | SBC KS p > 0.01 per coordinate; 50/90 % coverage within ±5 pp; $\mathrm{KL}(\hat\pi_D\,\|\,q)$ median < 0.1 (J = 2) |
| P3.2 | decoder-implied posterior calibration | SBC of $\hat\pi_D$ on the J = 2 grid passes (KS p > 0.01) |
| P3.3 | K-curve at Σ = 0 | NLL gap to IS-ref at K = 128 ≤ 0.03 for n_A ≥ 2, ≤ 0.07 at n_A = 1 |
| P3.4 | **Σ sweep** Σ ∈ {0, 0.1, 0.3, 1, ∞}·I, at K ∈ {8, 32} | **primary:** the minimum NLL is at an interior Σ and beats the direct PFN by ≥ 0.05 nats at n_A ≥ 2 (mean over 3 seeds, > 2 standard errors) |
| P3.5 | J scaling at the best Σ | NLL gap to IS-ref grows by ≤ 0.05 from J = 2 to 16 at K = 32 |
| P3.6 | QMC vs i.i.d. | reported, no threshold |

**Decision rule.**
- **P3.4 passes →** continue to h (V7) and d = 2 with AIR at the chosen Σ.
- **Σ = ∞ (direct PFN) wins, or the gain < 0.05 →** the explicit latent isn't paying in d = 1. Before abandoning it, run the J = 16 and d = 2 cells, where a direct PFN should suffer most.
- **Otherwise:** keep the direct PFN plus soft-target LUPI as the main model.

## 7. What to log

The same `results/LOG.md` format as before, plus:
- a Σ-by-K NLL heatmap per n_A;
- the K-curves;
- the validation-NLL curves for q, showing the plateau.
