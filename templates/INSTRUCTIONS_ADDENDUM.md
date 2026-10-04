# Standing rules for the coding agent — paste into the repository instruction file

1. Build from `SPEC.md` and the current work-unit card in `PLAN.md`. Never from any longer document, and never from memory of an earlier session.
2. Read only the files the card names. Do not explore the repository. Do not re-read the proposal.
3. Features are plain functions with an `asof` argument and trailing normalisation only. Every new feature is added to `FEATURES` and must pass `tests/test_pit_truncation.py` before it is used anywhere.
4. The ladder measures cumulative P&L of fresh entries at fixed horizons. No daily rate is ever rolled forward or scaled. No transition or switch term is ever added to horizon outcomes.
5. Transaction costs are out of scope. Do not model, estimate, mention or display them.
6. Units are cash per unit of standard notional. Do not express any result as a percentage of premium.
7. `configs/gates.yaml` is read-only for you. Thresholds change only through `DECISIONS.md`, by the owner.
8. If an item in `DECISIONS.md` is unresolved and blocks the unit: stop, write the question into `HANDOFF.md › uncertain`, and end the session. Do not choose a default.
9. End every session by updating `HANDOFF.md` (done / next / uncertain / parked) and running `pytest -q`. A unit is done only when its acceptance test passes.
10. Results go to files under `out/` with the data as-of date in the filename. Numbers reported only in chat do not exist.
11. Keep it small. No new abstractions, base classes, registries or frameworks. If a change needs more than about 150 lines, stop and propose a split in `HANDOFF.md`.
12. Nothing committed to this repository names or describes the data owner's internal systems. Site-specific code lives in `adapters/` and `configs/local.yaml`, which are untracked.
13. One commit per work unit. The commit body is the `HANDOFF.md › done` section. Tag at each gate.
14. When a unit is marked **packet** in `PLAN.md › Route`, finish it by running `python -m regime_ladder inspect ...`, then `python -m regime_ladder view --src out/inspect` (the owner's page), then `python -m regime_ladder pack --stage <stage> --src out/inspect [--market ...] [--td ...]`, write the packet and page paths into `HANDOFF.md › next` as "packet ready for review", and stop. The next unit starts only after the owner has pasted the reviewer's requests into `HANDOFF.md › next`.
15. Steps marked **Owner** in a card or in `HANDOFF.md` are not yours. Do not attempt them, do not mark them done, and do not treat them as blocking your part of the unit.
