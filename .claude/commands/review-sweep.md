---
description: Review a finished sweep against its notebook description
argument-hint: <sweep dir>
---
Review the sweep `$ARGUMENTS`:

1. `just sweep-status`: how many members are DONE / FAILED / STARTED / MISSING? For FAILED, read the
   tracebacks (`<member>/FAILED`) and group them by cause. Results are preliminary unless all are DONE.
2. Provenance: `sweep.json` commit and tag; `allow_dirty` must be false for anything reportable.
3. Snapshot it (`just snapshot`), then execute its notebook (`just notebook <nb>`).
4. Evaluate the GO / NO-GO criterion from the notebook's description, against baselines and null controls,
   with the CIs from `summarize` / `paired_diff`. If the result was selected as the best of many, say whether
   a held-out estimate exists.
5. Report: verdict you would propose, with numbers and the snapshot path; what would change your mind.
   Do not run `just decide` yourself unless I say so.
