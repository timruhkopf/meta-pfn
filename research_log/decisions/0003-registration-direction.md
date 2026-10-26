# 0003 — Meta-PFN before an E0 GO, and the registration direction (M3, simplified)

Status: accepted, 2026-10-26 (user, in conversation; the scope exception was given 2026-10-22).

## Context

- CLAUDE.md gates Meta-PFN training on an E0 GO. E0 and E0b are IMPRECISE, not decided, yet E0 / sampler_check
  already show B is worth 1.4–2.1 nats per query at n_A = 0 and a good sampler keeps 43–88% of the known-T headroom
  (brainstorm §1). The user directed model work anyway (2026-10-22: two-context PFN / registration prototype).
- The user has trained the plain two-context PFN (M0, brainstorm §5.1) many times; it never learned to use a
  warped B (2026-10-26). Meta_FTPFN v0 failed the same way (decision 0002: copy detection, prior mismatch).
- Phase 1 (notebook 04) tested the M3 back half with oracle / perturbed seeds: the learned read-out is a weak
  reader (E3 FAIL, m* 8–41 of 256), refinement fixes y- but not x-errors, the null weight drifts up.

## Decision

1. Model work proceeds before an E0 GO, in `src/meta_pfn/experimental/`, as a user-directed exception.
2. The direction is M3 (GW landmarks + weight-tied displacement-field operator, LUPI via B_inA; design doc
   `docs/design_2026-10-23_gw-landmarks-displacement-field.md`), built and tested **component by component**:
   - **K = 1** hypothesis slot (the best GW matching) + the null slot (+ the identity field as a free fixed slot);
   - **h = identity first** (cells C00, C10, plus the unrelated stratum), h added once x works;
   - the GW front half and the field back half are each scored against B_inA before they are composed.
3. M0 is a control only, never a candidate.

## Consequences

- No slot collapse, slot exchange or slot-identity loss to deal with yet; multimodal posteriors over T are not
  represented (the null slot hedges). Deferred: K > 1 / continuous hypotheses, h ≠ id (`deferred.md`).
- Numbers from this line are pilots against references from the same prior (A-only PFN, effective-data curve);
  they are not an E0 verdict.
