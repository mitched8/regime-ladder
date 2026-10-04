# DECISIONS

Choices only the research owner makes. The agent stops when it hits an open one. Each line is
dated when closed. Gate-threshold changes are recorded here with the reason.

| # | Decision | Options | Status | Closed on | Note |
|---|---|---|---|---|---|
| D1 | Per-unit denominator for base legs | vega at inception (default, needs per-day `vega`); source standard notional (fallback) | open | | see `configs/archetypes.yaml`; D14 covers combinations |
| D2 | P&L currency | source currency; converted to a single reporting currency | open | | |
| D3 | Horizons | {1,3,5,10,20,T} (default) | closed | | from SPEC |
| D4 | Primary horizon for the vertical slice | 5d (default, market-making book) | closed | | |
| D5 | Labeller smoothing half-life and hysteresis band | 3 / 3 (defaults); chosen inside training folds | open | | WU-08 |
| D6 | Scoring targets for the state finder | level bounds: forward ATM 5d ahead on the whole history (default) or forward realised vol; partition/direction/hysteresis: forward 5d archetype P&L where the backtester covers (default) | open | | WU-08; two targets, two jobs (SPEC §3a) |
| D7 | Shrinkage κ method | leave-one-fold-out MSE (default); fixed | open | | WU-10 |
| D8 | Pairs in Phase 1–3 | one liquid pair first, second pair at WU-05 | open | | |
| D9 | Data as-of date for the Phase 3 snapshot | | open | | pinned in `configs/local.yaml` |
| D10 | Interval method | block bootstrap (default); HAC | closed | | validated on synthetic data |
| D11 | Named states | 6 (carry, rising, agitated, stressed, normalising, settling; default); 6x (plus extreme, merged into stressed when < 5 episodes); 3 (level only) | open | | decided by the finder on forward P&L at WU-08, confirmed at Gate 2; extreme is a flag unless it has episodes |
| D12 | Cross-asset series in the profile | defaults in `configs/profile.yaml`; mapped to source tickers in `configs/local.yaml` | open | | expand or change freely; nothing else depends on the list |
| D14 | Leg scaling for combinations | one unit of vega at inception per base leg (default: weights in vega units, zero-sum = smile-vega-neutral); equal notional; premium-neutral; equal ES contribution | open | | WU-03 |
| D15 | Tenors in Phase 3 | 1W, 1M, 3M (default `tenors_phase3`); any of 1W–1Y available | open | | WU-05 |
| D16 | Override of any `configs/gates.yaml` threshold | | — | | record each change as a new line below |
| D17 | Tags in use and their per-pair thresholds | corr_sign, pinned, event_window (defaults); intervention_risk for the pairs the desk names, with direction, move threshold, level and the official-comment flag column | open | | WU-08d; thresholds live in `configs/local.yaml` |
| D18 | CUSUM false-alarm budget | one alarm per ~4 years (ARL₀ 1000, default); per year (250) | open | | WU-14; `configs/leading.yaml` |
| D19 | Future-path policy per leading covariate | pressure index: decaying with a 10-day half-life (default) or frozen; calendar features: calendar; anything else: zero | open | | WU-17; a frozen path is a scenario, not a forecast |
| D20 | Desk-proprietary leading series to screen | none (default); own-book risk measures, aggregated flow statistics, positioning — each wrapped as an `asof` function and registered | open | | WU-16; which series are admissible, and under what aggregation and null rules, is decided by the owner and recorded in the local decisions file alongside `configs/local.yaml`, never here; the tracked repo sees them only as registered functions |
| D22 | Leading-screen retention rule | (a) both tests must pass (current: outcome ΔR² at 5d AND transition gain at k=5, each in ≥ 3/4 folds, plus the permutation null and the hold-out guard); (b) split by product — outcome test admits a feature to the ladder card, transition test admits it to the tilt; (c) declared transition horizon k = 5 (current) or 10 | open | | the external-leader calibration world (`docs/calibration/D_external`) drives transitions (gain 0.008 at k=5, 0.10 at k=21, lift 1.7) yet adds ~0 to the 5-day straddle outcome, so (a) rejects it; evidence for (b). Change only through `configs/gates.yaml` / `leading.yaml` with a line below |
| D21 | Forward outcome for the leading screen | forward 5d archetype P&L (default); realised-minus-implied over 21d | open | | WU-16; `configs/leading.yaml › leading.target` |

## Gate threshold changes
| Date | Key | Old | New | Reason |
|---|---|---|---|---|
| 2026-10-04 | `phase4_leading.n_perm`, `max_p_perm` | — | 100, 0.05 | selection control for a screen over many candidates: the outcome gain must beat a circular-shift null of the feature |
| 2026-10-04 | `phase4_leading.holdout_frac`, `max_to_holdout` | — | 0.2, 3 | the walk-forward runs on the first 80%; the top 3 retained candidates must not be contradicted on the final 20% (a 2-year window holds 3–4 episodes, so it guards rather than re-proves; the first draft required confirmation and rejected the true series in calibration world A) |
