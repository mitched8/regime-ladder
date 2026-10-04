# PLAN — work units through the Phase 3 go/no-go and the Phase 4 leading-feature gate

The phases below are the dependency order; the **Route** section is the order to work in.

Conventions: one work unit (WU) is one agent session, two at most. Each WU names the only files
the builder reads, produces named outputs, and is done when its acceptance test passes and
`HANDOFF.md` is updated. A separate reviewer session (see `templates/REVIEWER_PROMPT.md`) passes
or fails it before the next WU starts. Gates are the research owner's review points.

**Model allocation.** *Routine* = the smaller model; *judgement* = the larger model: schema
discovery, point-in-time machinery, statistics, anything failed twice in review.

Status legend: `todo` · `doing` · `review` · `done` · `blocked (see HANDOFF)`

## Route: two tracks, joined at Gate 2

Everything on the market side (states, transitions, shocks, leading features) needs only the market
frame, so it runs on the full surface history (10y+) without waiting for the backtester. The trade
side needs the backtester API and runs alongside. They join when the finder is rerun on forward P&L.

| step | track A — market side | track B — trade side |
|---|---|---|
| 1 | WU-00 localise (both tracks) · send CR-2 (backfill extension) on day one | |
| 2 | WU-06 market adapter, one pair, full history | WU-01 trade-day adapter (hardest unit: per-day vega, components, cut times) |
| 3 | WU-07 `inspect` on the real frame | WU-02 vertical slice |
| 4 | WU-08 states, finder on the forward-surface fallback target · **packet: states** | WU-03 legs and packages · WU-04 golden trades · **packet: data** |
| 5 | WU-15 constant matrix and fan (comes from `inspect`) | |
| 6 | WU-14 shocks · WU-16 leading screen (realised-minus-implied target) · **packet: leading** | |
| join | WU-08 rerun with `--td` (finder on forward P&L) → WU-09 → **Gate 2** · **packet: states, ladder** | |
| then | WU-10, WU-11 → **Gate 3** · **packet: ladder** | WU-16/17 rerun on forward P&L → **Gate 4** · **packet: leading** |

**Deferred** until the stage they serve shows signal: WU-06b (cross-asset frame; only profiles and
sub-states need it), WU-08d (tags), WU-05 (second pair; add once the first is clean), WU-12 and
WU-13 (age-dependence and robustness; only if Gate 3 is promising).

## Review loop

After each bold **packet** step: `python -m regime_ladder inspect ...` then
`python -m regime_ladder pack --stage <stage> --src out/inspect [--market ...] [--td ...]`. The packet
(`out/packets/<stage>/packet.md`) goes to a separate model session with
`templates/CHALLENGER_PROMPT.md`; its REQUESTS go into `HANDOFF.md › next`. The same separate model
drafts each builder prompt (prompt writer, same file), so coding-agent sessions spend their budget on
execution. Calibrate the challenger once on `docs/calibration/` before the first real packet. Code
review (`templates/REVIEWER_PROMPT.md`) still covers the diff; the challenger covers the results.

---

## Phase 0 — Localise

| WU | Goal | Reads | Produces | Acceptance | Model | Status |
|---|---|---|---|---|---|---|
| 00 | Pull, run, inventory | README, this card | **Agent:** `pytest -q`, `python -m regime_ladder demo` and `python -m regime_ladder validate` green locally; `docs/ENVIRONMENT.md` (untracked) listing available APIs, packages, paths, existing retrieval plumbing and feature sub-components; `git check-ignore adapters/x configs/local.yaml docs/ENVIRONMENT.md` prints all three. **Owner (not the agent):** repo instruction file pasted from `templates/INSTRUCTIONS_ADDENDUM.md`; CR-2 sent; the challenger calibrated on `docs/calibration/` | agent part: the three commands succeed and ENVIRONMENT.md exists; owner part: inventory reviewed and the owner marks the unit done in HANDOFF | routine | todo |

## Phase 1 — Data, trade definition, integrity

| WU | Goal | Reads | Produces | Acceptance | Model | Status |
|---|---|---|---|---|---|---|
| 01 | Trade-day adapter | SPEC, `regime_ladder/schema.py`, `checks.py`, `docs/LOCALISATION.md` | `adapters/trade_days.py` (untracked): source API → schema frame, cached to parquet with as-of date; one pair, straddle archetype, full backfill; schema dump of the source | `python -m regime_ladder check --td …` passes; row count and date range logged in HANDOFF | judgement | todo |
| 02 | Vertical slice | SPEC, `ladder.py`, `report.py` | One pair, straddle, unconditional, h=5: cumulative → ladder ALL row → interval → plot, on real data | numbers reconciled against a direct pandas computation for one month of entries; plot saved | routine | todo |
| 03 | Base legs and combinations | SPEC §3b–4, `combine.py`, source documentation | `configs/archetypes.yaml` filled: the five base legs' hedge rule, notional unit and currency as the source defines them; the source's own RR and fly packages reconciled against `combine` of legs (agreement up to the ratio convention tabulated); per-day `vega` confirmed available for `to_vega_units` or D1 falls back to notional | reviewed by owner; D1, D2, D14 closed; reconciliation table saved | routine | todo |
| 04 | Golden trades | `templates/golden_trades.csv`, WU-01 adapter | owner names 4–5 trade IDs (calm; entered just before a stress episode; through a stress episode; expiring near the money; a risk reversal whose strikes drifted far); agent extracts their daily rows to `tests/golden/` and writes `tests/test_golden.py` | test passes; owner has eyeballed each CSV and signed `golden_trades.csv` | routine | todo |
| 05 | Extend adapter | WU-01 adapter, `checks.py` | all five base legs, second pair, tenors in `tenors_phase3`; integrity report per leg and tenor | all checks pass; `info_` lines reviewed | routine | todo |
| 06 | Market adapter | `features.py`, `docs/LOCALISATION.md` | `adapters/market.py` (untracked): point-in-time market frame (ATM by tenor, RR, fly, spot, realised vols from hourly spot), cached; cross-check of entry-date marks against the marks carried in the trade-day table | every registered feature passes the truncation test on the real frame; cross-check differences tabulated | judgement | todo |
| 06b | Cross-pair and cross-asset adapter | `configs/profile.yaml`, `profile.py` | extend the market frame with the G10 spot panel and the cross-asset series named in `configs/profile.yaml` (equities, front- and long-end yields, oil, gold, dollar index, credit, equity and rates vol), point-in-time; `profile.characteristics` runs on the real frame | characteristics frame built for 10 years; truncation test on `characteristics` passes; coverage per series tabulated | routine | todo |

**Gate 1 (owner + a quant, 45 min):** integrity report, golden trades, mark cross-check. Pass → Phase 2.

**In parallel, owner-driven, not on the critical path:**
- CR-1 to the backtester owners: second-order Greek buckets (gamma, vanna, volga), daily Greeks, remaining tenor and moneyness per leg, explicit residual.
- CR-2: extend the backfill to cover the earliest stress episode available for two pairs.

## Phase 2 — States

| WU | Goal | Reads | Produces | Acceptance | Model | Status |
|---|---|---|---|---|---|---|
| 07 | Features on real data | `features.py`, `docs/COMPONENTS.md`, WU-06 adapter | `python -m regime_ladder inspect` run on the real frame (every derived input as a file, in `out/inspect/<pair>/`); plots and descriptive stats per feature per pair; existing feature sub-components wrapped as functions with `asof` and added to `FEATURES` | all registered features pass `test_pit_truncation`; no feature uses a centred or full-sample statistic; owner has looked at `features.csv`, `composite.csv` and `finder_report.csv` | routine | todo |
| 08 | Labeller calibration | `labels.py`, `configs/default.yaml` | composite + EWMA; `calibrate_states` **inside walk-forward training windows**: level bounds step-fitted on forward ATM (whole history), extreme bound as tail quantile, partition 3 / 6 / 6x and hysteresis chosen on forward P&L under the switch budget; label history per pair; durations, transition matrix, episodes; extreme merged or kept by the episode rule | label history saved with fold boundaries logged; durations vs geometric reported; episodes per state ≥ 5 or flagged; states packet passed by the challenger; D11 closed (first pass on the forward-surface target; rerun on forward P&L at the join) | judgement | todo |
| 08d | Tags | `tags.py`, `configs/default.yaml › tags` | corr-sign, pinned, event-window (calendar from the adapter) and, for the configured pairs, intervention-risk tags on the real frame; episode counts per (state, tag) cell; which cells split | all tags pass the truncation test; cell table saved; intervention thresholds per pair recorded in `configs/local.yaml` | routine | todo |
| 09 | Separation report | `ladder.py`, `gates.py` | ladder at h ≤ 10 with real labels; `out/gate_phase2.json`; plots | gate evaluated and written; sign stability across sub-periods tabulated | routine | todo |

**Gate 2 (owner, 30 min):** separation. Pass → Phase 3. Fail → first look at detection lag (compare label-switch dates with the mark moves), then at feature choice; do not loosen `gates.yaml`.

## Phase 3 — The ladder product

| WU | Goal | Reads | Produces | Acceptance | Model | Status |
|---|---|---|---|---|---|---|
| 10 | Full ladder | `ladder.py`, `report.py` | all archetypes, pairs, horizons, with labels; increments; persistence split; κ by leave-one-fold-out MSE; cards and plots in `out/ladder/` | identity tests (`tests/test_ladder.py`) pass on real data; cards generated | routine | todo |
| 11 | Benchmarks | `evaluate.py`, `gates.py` | layer 0 (unconditional), layer 1 (ridge on the feature set, walk-forward), layer 2 (named-state ladder), layer 2b (sub-state ladder where discovery found stable sub-states) — `walk_forward.csv` for each; `gate_phase3.json` for layer 2 vs 0 **and** vs 1; 2b vs 2 | gate evaluated against both; per-horizon table of improvement, rank IC, calibration slope; a sub-state enters the card's number only if 2b beats 2 | judgement | todo |
| 12 | Age-dependence diagnostics | WU-10 outputs, trade-day table | exposure-normalised daily P&L by h, by T−h, by h/T pooled across tenors (raw-dollar version if Greeks not yet available); residual by remaining tenor; conditional on cumulative move since entry | three collapse plots per archetype + one page naming which panel collapses | routine | todo |
| 13 | Robustness and report | WU-10/11 outputs | leave-one-episode-out; sensitivity to halflife, δ, κ, block length; `docs/PHASE3_REPORT.md` | tables in `out/robustness/`; report drafted; no single episode flips a sign at h ≤ 10 or it is stated | routine | todo |

**Gate 3 (owner + traders, 60 min): go/no-go.** The regime layer is kept only if it beats layers 0 and 1 at h ≤ 10 out of sample. Separate decision: build the Phase 5 path model (needs CR-1 delivered and a surface-override capability from the backtester owners).

## Phase 4 — Shocks, transitions, leading features

Runs after Gate 2 (needs labels); does not wait for Gate 3. Everything here is scored against the
labels the market actually produced and a baseline that already knows the continuous state.

| WU | Goal | Reads | Produces | Acceptance | Model | Status |
|---|---|---|---|---|---|---|
| 14 | Shock detectors on real data | `shock.py`, `configs/leading.yaml › shock` | surprise, CUSUM (h calibrated on the first year, ARL per config), BOCD, HAR forecast, shock score per pair in `out/shock/`; lead profile against the labels; alarm dates listed next to the label-switch dates | all four series pass the truncation test on the real frame; `lead_profile` shows `before_up` above `all` or the detector is flagged as coincident, not leading | routine | todo |
| 15 | Constant matrix and fan | `transitions.py`, `configs/leading.yaml › transitions` | per pair: matrix with sticky prior, implied vs observed durations, the 21-day fan from today's state; `python -m regime_ladder transitions` | duration ratios within 0.6–1.5 or the first-order approximation is flagged for that state | routine | todo |
| 16p | Proprietary series: provenance and snapshots | `docs/LOCALISATION.md`, the owner's local decisions file | for each proprietary source the owner admits (D20): history start, known breaks, timestamp semantics and `lag_days`, restatement risk, coverage against the aggregation rule, and the count of high-band entries inside its history; a daily append-only point-in-time snapshot job for the quantities that cannot be reconstructed later | audit table filed next to `configs/local.yaml`; snapshot job running with a named owner; sources with too few in-sample episodes routed to the profile slot, not the screen | routine | todo |
| 16 | Leading-feature screen | `leading.py`, `configs/leading.yaml › leading`, WU-06/06b frames, WU-16p audit | the six generic candidates plus any desk series the owner supplies (flow, dealer gamma, barrier book — wrapped as `asof` functions and registered), each screened one at a time: `out/leading/leading_screen.csv` with both tests, the lagged check, and the retain flag; event calendar filled per pair | every candidate passes the truncation test; leading packet passed by the challenger (alignment check clean for every desk series); no candidate added to the pressure index that fails either test | judgement | todo |
| 17 | Pressure index and Gate 4 | WU-16 outputs, `gates.py` | pressure from the retained features; tilt β per pair; fan with and without the tilt under the declared covariate paths; `out/gate_phase4.json`; the card's leading line | gate evaluated and written; covariate path policy per feature recorded in D19 | routine | todo |

**Gate 4 (owner, 30 min): leading information has incremental value.** Pass → the pressure index enters the card and the Phase 5 path model. Fail → the constant matrix stands and the card says so; do not loosen `gates.yaml`.

---

## Owner time

Five fixed points: golden trades (1 h, WU-04), Gate 1, Gate 2, Gate 3, and one 30-minute pass over
`DECISIONS.md` before WU-01. Everything else is builder + reviewer. If the owner is reading code,
the plan has failed.

## Change control

- `configs/gates.yaml` changes only by the owner, with a dated line in `DECISIONS.md`.
- A WU still open after three sessions was mis-scoped: split it, do not push on.
- New ideas go to `HANDOFF.md › parked`, not into the current unit.
