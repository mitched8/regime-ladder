# Builder session prompt (coding agent, inside the repository)

```
You are implementing work unit WU-{NN} of the regime-ladder research scaffold.

Read, in this order, and nothing else unless the card lists it:
  1. SPEC.md
  2. The WU-{NN} row of PLAN.md
  3. HANDOFF.md
  4. The files named under "Reads" in the card

Obey the standing rules in the repository instruction file. In particular: no exploring,
no costs, cash units only, no edits to configs/gates.yaml, stop on an unresolved DECISIONS item.

Deliver:
  - the files named under "Produces" in the card
  - `pytest -q` passing (paste the last three lines of its output into HANDOFF.md)
  - HANDOFF.md updated: done / next / uncertain / parked
  - the results table or CSV the acceptance criterion needs, saved under out/ with the as-of date

When the acceptance criterion is met, stop. Do not improve anything the card did not ask for.
```

Replace `{NN}`. One unit per session. If the unit is not done when the session ends, say so in
`HANDOFF.md › next` with the exact remaining step; the next session starts from there.
