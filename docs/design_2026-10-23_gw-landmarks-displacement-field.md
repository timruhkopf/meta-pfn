<!-- User design note, received 2026-10-23 10:30 (~/Downloads, author @Tim), copied verbatim. Logged as option M3 in
docs/brainstorm_2026-10-23_registration.md §5.7. The "[embedded content]" placeholder was a figure not included in the export. -->

# MetaPFN — Registration Design: GW Landmarks + Displacement Field

Oct 23, 2026 · @Tim

## Core idea

GW does not subsume the displacement field: they answer different questions at different resolutions, and the design needs both.

| Object | Question it answers | Resolution | Built from |
| --- | --- | --- | --- |
| Coupling π (GW) | Which B points do my few A points correspond to? | Sparse (n\_A points), coarse, possibly multimodal | Within-task order relations only |
| Displacement field x̂ (refinement) | Where does every B point land in A's frame? | Dense (all N\_B points), metric | Iterative, distance-based evidence |

GW finds the landmarks. The field is an elastic sheet pinned at those landmarks and then relaxed toward the evidence. Landmarks alone say nothing about where B points between the A points go, and those are the points A\_test needs. A sheet without landmarks has no idea where to start.

## Running example (d = 1)

f\_A has two bumps: a tall one at x = 0.3 and a smaller one at x = 0.7. B is the same landscape shifted right and stretched, so its tall bump sits at u = 0.5 and its small one at u = 0.9. Output values are rescaled. A has 4 observed points; B has 500.

## Forward pass, step by step

Six steps take raw A and B tokens to a mixture prediction on A\_test; only step 3 loops.

&#91;embedded content: forward pass · 6 steps, 1 loop\]

Steps 0–2 run once; step 3 repeats T times with shared weights; steps 4–5 turn the refined slots into one prediction.

### Step 0: encode each task on its own

A and B each get intra-task self-attention, with Fourier features for x and within-task y-ranks. B's encoding does not depend on A, so it can be cached across BO iterations.

### Step 1: GW finds candidate landmark matches (coupling π)

Each A point spreads attention over B: π is n\_A × N\_B, row-softmax. A few GW-style layers then sharpen π by checking consistency.

The check uses only within-task order relations: which of two points is further left, and which has the higher y. These survive any monotone warp and any monotone output rescaling, so no learned descriptors are needed for it.

In the example, A's highest point (x ≈ 0.3, rank 1 in A) initially matches plausibly to both B bumps, because ranks alone are ambiguous. The consistency check asks whether each assignment keeps A's other three points on the correct sides with the correct value order. Matching to the tall B bump passes. Matching to the small one may also pass if A's points don't cover the second bump. Two consistent matchings means two hypotheses.

Output: a few distinct, internally consistent matchings of the 4 A points into B.

### Step 2: turn each matching into an initial field (seed a slot)

Each matching gives a displacement at a few B locations, e.g. "the B point near u = 0.5 sits at A's x = 0.3". Spread these to all 500 B points by smoothing, i.e. distance-biased attention in B's own coordinates. Each slot gets an initial dense field x̂\_i^k for every B point i. The values ŷ\_i^k are seeded the same way from the matched y pairs.

Slots: one per distinct matching, plus the identity (B\_inA = B), plus the null slot (ignore B).

### Step 3: refine each field iteratively (weight-tied, RAFT / AlphaFold style)

Each iteration, for every B point in every slot:

1. **Look up evidence.** The B point sits at x̂ in A's frame. It looks at A points near x̂, with attention biased by distance in A's frame, and sees the residual: what A observed minus what this slot predicts there.
2. **Propose a correction** to x̂ and ŷ through a GRU-style update shared across iterations.
3. **Smooth the corrections** over B neighbours, with attention biased by distance in B's own frame. This is the coherence regularizer: a B point cannot move independently of its neighbours, so the sheet stays a sheet.
4. **Exchange between slots** so they do not collapse onto the same hypothesis.

GW never computed anything between A points. The field does: it handles the metric fit (how much stretch), the regions between landmarks, and the values.

### Step 4: read out per slot

Under slot k, A\_test tokens attend to A\_train plus B rendered at its slot-k positions, with distance bias in A's frame. A query at x\* = 0.4 finds the B points whose x̂ ≈ 0.4 and reads their ŷ. Each slot outputs its own bar distribution.

### Step 5: mix

The final prediction is Σ\_k w\_k · p\_k(y\* | x\*), null slot included.

## Where distance-biased attention lives

Distance-biased attention appears three times, always in a specific frame; GW uses no distances at all.

| Where | Frame | Purpose |
| --- | --- | --- |
| Field smoothing (steps 2, 3) | B's own coordinates u | Keep the map coherent; neighbours in B move together |
| Evidence lookup (step 3) | A's frame, at the current x̂ | B point checks A's data where it is hypothesized to sit |
| Read-out (step 4) | A's frame, at the current x̂ | A\_test reads the rendered B |

GW uses only order, which makes it robust to warps but not metrically precise. The two mechanisms complement each other: order finds coarse correspondences robustly, distance gives precision once you are in roughly the right frame.

## Training signals

The mixture NLL is the task; three LUPI terms shape the slots, their weights and the descriptors. None of them exists at deployment.

- **Mixture NLL on A\_test.** The main task. It makes the mixture a correct posterior predictive.
- **LUPI on the field.** Pull x̂ and ŷ toward the true B\_inA at every refinement step, but only for slots near the truth (soft, responsibility-weighted).
- **LUPI on the weights.** Cross-entropy between w and "which slot is actually right", with the null slot winning if none is close. This makes w a calibrated posterior over hypotheses.
- **LUPI on descriptors (optional).** A contrastive loss on true A↔B partners, which sharpens step 1 beyond what order relations alone give.

## In one sentence

GW gives each hypothesis its starting landmarks, robustly and coarsely, from order alone; the displacement field turns landmarks into a full map of B in A's frame and refines it precisely from distance-based evidence; the mixture over these maps, plus the null slot, is the posterior over registrations that the NLL measures.
