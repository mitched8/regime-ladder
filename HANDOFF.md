# HANDOFF

Updated at the end of every session. The next session starts by reading this file.

## Current unit
WU-00 — Localise — status: todo

## Done (this session)
- Screen plumbing (`leading.py`, `transitions.py`): pooled screen across pairs (pair dummies, folds at common dates, one tilt β), fold-local z-scoring in the tilt, permutation p-value for the outcome gain, hold-out guard, diagnostic horizons (k = 10, 21; outcome at 21d), low-band interaction (`delta_r2_low`), per-feature targets (signed features → RR), event-study `lift`, `gap_z_signed` and `drift_t` registered, pressure index z-scored on a trailing window. Synthetic external-leader world (`external_beta`, `external_lead`); calibration world D. Gates keys recorded in DECISIONS.
- WU-16p card (provenance audit and daily snapshots for proprietary series) added to PLAN; the detailed checklist lives with the owner's local decisions.
- `pack` command (`regime_ladder/pack.py`): one review packet per stage (data, states, ladder, leading) for a reviewer without the code; `inspect` now also writes gates 2/3/4, walk-forward and `meta.json`.
- `templates/CHALLENGER_PROMPT.md`: results challenger with stage checklists, plus the prompt writer.
- `docs/calibration/`: packets on three synthetic worlds (genuine desk series, useless one, planted timestamp bug) with an answer key.
- Fix: Gate 4's pressure test now uses the same footing as the screen (k-step likelihood, composite baseline); before, a genuinely leading series could pass the screen and fail the gate.
- Fix: CUSUM recorded its statistic after the reset, so the alarm day showed zero pressure and the shock score dropped exactly when the detector fired. Found by the challenger dry runs.
- Fix: `shock.lead_profile` ranked states in display order (stressed→normalising counted as an escalation); now by stress rank. `docs/shock.html` regenerated.
- `PLAN.md › Route`: market side and trade side in parallel, joined at Gate 2; deferrals listed.

## Next (exact next step, one line if possible)
- WU-00: pull, `pytest -q`, `python -m regime_ladder demo`, paste `templates/INSTRUCTIONS_ADDENDUM.md` into the repo instruction file, calibrate the challenger on `docs/calibration/`, write `docs/ENVIRONMENT.md`. Send CR-2 the same day.

## Uncertain (questions for the owner; blocking items reference DECISIONS.md)
- D17 (which pairs get an intervention-risk tag and with what thresholds), D20 (which desk series to screen as leading features) — both can wait until Phase 4.

## Parked (ideas that are not the current unit)
- Labels during the features' warm-up default to carry (first ~6 months of the market frame). Harmless while the trade table starts years later; blanking them needs NaN-safe tags/episodes. The states packet prints the warm-up date.
- D22 (retention rule: both tests vs split by product; declared k): the external-leader world is the evidence; decide before WU-16 runs on real data.
- CUSUM raises 0–1 alarms in ten synthetic years (surprises there never shift in variance): check the false-alarm budget (D18) on real data at WU-14.
- Power: with noisy forward P&L, ten years of one pair did not retain even the true synthetic pressure series. Expect Gate 4 to be conservative; pooling pairs or the realised-minus-implied target (D21) are the levers.
- Shock blend into the state probability vector is a heuristic until the §14 calibration test exists.
- Phase 5 path model: regime-conditional surface-move bootstrap through a repricer; consistency test against the ladder first.

## Last test run
```
110 passed (pytest -q, ~5 min)
```

## Data as-of
none yet (synthetic only)
