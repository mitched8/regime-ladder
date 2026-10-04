# PLAN — work units through the Phase 3 go/no-go

Conventions: one work unit (WU) is one agent session, two at most. Each WU names the only files
the builder reads, produces named outputs, and is done when its acceptance test passes and
`HANDOFF.md` is updated. A separate reviewer session (see `templates/REVIEWER_PROMPT.md`) passes
or fails it before the next WU starts. Gates are the research owner's review points.

**Model allocation.** *Routine* = the smaller model; *judgement* = the larger model: schema
discovery, point-in-time machinery, statistics, anything failed twice in review.

Status legend: `todo` · `doing` · `review` · `done` · `blocked (see HANDOFF)`

---

## Phase 0 — Localise

| WU | Goal | Reads | Produces | Acceptance | Model | Status |
|---|---|---|---|---|---|---|
| 00 | Pull, run, inventory | README, this card | `pytest` and `python -m regime_ladder demo` green locally; `docs/ENVIRONMENT.md` (untracked) listing available APIs, packages, paths, existing retrieval plumbing and feature sub-components | both commands succeed; inventory reviewed by owner | routine | todo |

## Phase 1 — Data, trade definition, integrity

| WU | Goal | Reads | Produces | Acceptance | Model | Status |
|---|---|---|---|---|---|---|
| 01 | Trade-day adapter | SPEC, `regime_ladder/schema.py`, `checks.py`, `docs/LOCALISATION.md` | `adapters/trade_days.py` (untracked): source API → schema frame, cached to parquet with as-of date; one pair, straddle archetype, full backfill; schema dump of the source | `python -m regime_ladder check --td …` passes; row count and date range logged in HANDOFF | judgement | todo |
| 02 | Vertical slice | SPEC, `ladder.py`, `report.py` | One pair, straddle, unconditional, h=5: cumulative → ladder ALL row → interval → plot, on real data | numbers reconciled against a direct pandas computation for one month of entries; plot saved | routine | todo |
| 03 | Archetype table | SPEC §4, source documentation | `configs/archetypes.yaml`: per archetype — legs, ratio rule, hedge rule, standard notional unit, P&L currency, as the source defines them | reviewed by owner; `DECISIONS.md` entries for notional unit and currency closed | routine | todo |
| 04 | Golden trades | `templates/golden_trades.csv`, WU-01 adapter | owner names 4–5 trade IDs (calm; entered just before a stress episode; through a stress episode; expiring near the money; a risk reversal whose strikes drifted far); agent extracts their daily rows to `tests/golden/` and writes `tests/test_golden.py` | test passes; owner has eyeballed each CSV and signed `golden_trades.csv` | routine | todo |
| 05 | Extend adapter | WU-01 adapter, `checks.py` | RR and fly archetypes, second pair; integrity report per archetype | all checks pass; `info_` lines reviewed | routine | todo |
| 06 | Market adapter | `features.py`, `docs/LOCALISATION.md` | `adapters/market.py` (untracked): point-in-time market frame (ATM by tenor, RR, fly, spot, realised vols from hourly spot), cached; cross-check of entry-date marks against the marks carried in the trade-day table | every registered feature passes the truncation test on the real frame; cross-check differences tabulated | judgement | todo |

**Gate 1 (owner + a quant, 45 min):** integrity report, golden trades, mark cross-check. Pass → Phase 2.

**In parallel, owner-driven, not on the critical path:**
- CR-1 to the backtester owners: second-order Greek buckets (gamma, vanna, volga), daily Greeks, remaining tenor and moneyness per leg, explicit residual.
- CR-2: extend the backfill to cover the earliest stress episode available for two pairs.

## Phase 2 — Labels

| WU | Goal | Reads | Produces | Acceptance | Model | Status |
|---|---|---|---|---|---|---|
| 07 | Features on real data | `features.py`, WU-06 adapter | plots and descriptive stats per feature per pair in `out/features/`; existing feature sub-components wrapped as functions with `asof` and added to `FEATURES` | all registered features pass `test_pit_truncation`; no feature uses a centred or full-sample statistic | routine | todo |
| 08 | Labeller calibration | `labels.py`, `configs/default.yaml` | composite + EWMA; boundaries by regression kink **inside walk-forward training windows** against forward realised vol; hysteresis chosen on false-switch vs lag; label history per pair; durations, transition matrix, episodes | label history saved with fold boundaries logged; durations vs geometric reported; episodes per regime ≥ 5 or flagged | judgement | todo |
| 09 | Separation report | `ladder.py`, `gates.py` | ladder at h ≤ 10 with real labels; `out/gate_phase2.json`; plots | gate evaluated and written; sign stability across sub-periods tabulated | routine | todo |

**Gate 2 (owner, 30 min):** separation. Pass → Phase 3. Fail → first look at detection lag (compare label-switch dates with the mark moves), then at feature choice; do not loosen `gates.yaml`.

## Phase 3 — The ladder product

| WU | Goal | Reads | Produces | Acceptance | Model | Status |
|---|---|---|---|---|---|---|
| 10 | Full ladder | `ladder.py`, `report.py` | all archetypes, pairs, horizons, with labels; increments; persistence split; κ by leave-one-fold-out MSE; cards and plots in `out/ladder/` | identity tests (`tests/test_ladder.py`) pass on real data; cards generated | routine | todo |
| 11 | Benchmarks | `evaluate.py`, `gates.py` | layer 0 (unconditional), layer 1 (ridge on the feature set, walk-forward), layer 2 (ladder) — `walk_forward.csv` for each; `gate_phase3.json` for layer 2 vs 0 **and** vs 1 | gate evaluated against both; per-horizon table of improvement, rank IC, calibration slope | judgement | todo |
| 12 | Age-dependence diagnostics | WU-10 outputs, trade-day table | exposure-normalised daily P&L by h, by T−h, by h/T pooled across tenors (raw-dollar version if Greeks not yet available); residual by remaining tenor; conditional on cumulative move since entry | three collapse plots per archetype + one page naming which panel collapses | routine | todo |
| 13 | Robustness and report | WU-10/11 outputs | leave-one-episode-out; sensitivity to halflife, δ, κ, block length; `docs/PHASE3_REPORT.md` | tables in `out/robustness/`; report drafted; no single episode flips a sign at h ≤ 10 or it is stated | routine | todo |

**Gate 3 (owner + traders, 60 min): go/no-go.** The regime layer is kept only if it beats layers 0 and 1 at h ≤ 10 out of sample. Separate decision: build the Phase 4 path model (needs CR-1 delivered and a surface-override capability from the backtester owners).

---

## Owner time

Five fixed points: golden trades (1 h, WU-04), Gate 1, Gate 2, Gate 3, and one 30-minute pass over
`DECISIONS.md` before WU-01. Everything else is builder + reviewer. If the owner is reading code,
the plan has failed.

## Change control

- `configs/gates.yaml` changes only by the owner, with a dated line in `DECISIONS.md`.
- A WU still open after three sessions was mis-scoped: split it, do not push on.
- New ideas go to `HANDOFF.md › parked`, not into the current unit.
