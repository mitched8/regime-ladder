# First session (paste once, on day one)

Before this session the owner has pasted `templates/INSTRUCTIONS_ADDENDUM.md` into the repository
instruction file. If that file does not contain those rules, stop and say so.

```
This is the first session on this repository. Do only work unit WU-00, the agent part.

Read, in this order, and nothing else:
  1. The repository instruction file (the standing rules)
  2. HANDOFF.md
  3. The WU-00 row of PLAN.md

Then:
  1. Run `pytest -q`. If anything fails, paste the failure into HANDOFF.md › uncertain and stop.
  2. Run `python -m regime_ladder demo` and `python -m regime_ladder validate`.
  3. Write docs/ENVIRONMENT.md: available data APIs and their access pattern, installed packages
     and versions, paths for caches and outputs, any existing retrieval code that could become
     an adapter, and existing feature computations that could be wrapped as `asof` functions.
     Facts only, one line each. This file is untracked and stays on this machine.
  4. Run `git check-ignore adapters/x configs/local.yaml docs/ENVIRONMENT.md` and confirm all
     three paths print. If one does not, stop and say so.
  5. Update HANDOFF.md: done (the three commands and their result), next ("owner: review
     ENVIRONMENT.md, finish the owner steps of WU-00"), uncertain.

Do not start WU-01 or WU-06. Do not do the steps marked Owner. Commit nothing under adapters/,
configs/local.yaml or docs/ENVIRONMENT.md.
```

From the second session on, use `templates/BUILDER_PROMPT.md` with the unit number filled in (the
separate model's prompt writer in `CHALLENGER_PROMPT.md` drafts it from the card and HANDOFF).
