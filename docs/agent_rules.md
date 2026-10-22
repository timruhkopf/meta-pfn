# Rules for coding agents (template-owned; imported by CLAUDE.md)

Read `docs/process.md`, the current phase in `CLAUDE.md`, and the latest entries in `research_log/` before changing anything.

## Environment and code
- Use `just` recipes, or `uv run --extra <cpu|cu126>`. Never a bare `uv sync`, never `pip`, never edit `uv.lock` by hand.
- Run `just lint` and `just test` before proposing a commit.
- A new component is a class plus a config; a new pipeline is a function in `tasks/` registered in `TASKS`.
  Never a new entry point. A replicate is a sweep axis (`seed=0,1,2`), never a new experiment file.
- New ideas go to `src/meta_pfn/experimental/` unless told otherwise.
- Frozen components (carried-over models, priors) stay verbatim; add behaviour in new modules.

## Experiments
- Tracked work goes through `just run` / `just sweep` only; `uv run meta_pfn ...` is scratch.
- A new experiment starts with `just new-experiment <name> "<title>"`, then the notebook's description
  (question, setup, what to look for, GO/NO-GO, limits) is written **before** the first run.
- Never weaken provenance (no `--allow-dirty` for anything that may be reported), never bypass a stage gate,
  never edit a decided log section or a log entry's header by hand.
- Aggregates use DONE members; always report how many members are failed / started / missing.
- Never invent references or numbers. Every number in a log entry or the track record names the snapshot
  or artifact it comes from. References need a DOI or arXiv id; otherwise `status: to-verify`.
- Never read `.env.secrets`.

## Long-running and remote work
- Launch once. An ssh timeout often means the launch succeeded: verify with `just remote-ps` or
  `ps -eo pid,etime,args | grep "[p]attern"` before relaunching.
- Never `pkill -f` / `pgrep -f` (they can match their own command line); kill exact PIDs.
- Before `just sweep-resume`, check `just sweep-status`: STARTED may still be running.

## Autonomous sessions
- First track-record entry: the mandate (goal, deadline, allowed launches). Then one entry per milestone
  with its commit and what was verified. Corrections are new entries. Last entry: deliverables and open questions.

## Workflow
- Use the `workflow` skill (`.claude/skills/workflow/`) at natural checkpoints: work started or finished,
  before a launch, when results arrive. `just status` shows the state; name the exact next command, briefly.
- Issues for ideas/experiments/bugs, a branch per issue (`exp/<N>-<slug>`), draft PRs that close the issue.
- When a design choice, pre-registration or verdict narrows the scope, add an entry to
  `research_log/deferred.md` (what, why deferred, when to revisit); append status lines, never rewrite.
