# 0001: Start from research-template

**Context.** A new research project needs tracked, reproducible experiments without growing infrastructure.

**Decision.** Generated from research-template (see `.copier-answers.yml` for the options:
compute=remote, mlflow=True, optuna=False, training=False).
The process is `docs/process.md`; template improvements arrive with `uvx copier update`.

**Consequences.** Template-owned files (docs/process.md, justfile, tracking/, analysis/, scripts/,
tests/template/) are changed in the template, not here.
