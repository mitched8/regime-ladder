# Model guide

## What is estimated

For each option archetype, the distribution of cumulative net P&L from a fresh entry to horizon h,
conditional on the regime label on the entry date. See `SPEC.md` for the formal statement. The
design choice that matters: **condition on entry and measure at fixed horizons, rather than
estimate a daily P&L rate per regime and roll it forward.** Every observation then starts from the
same reference exposure and ages through the backtester's own repricing; the exposure is right by
construction and within-horizon regime transitions are already in the measured outcomes.

## Module map

| Module | Does | Does not |
|---|---|---|
| `schema.py` | defines the trade-day table and label frame | validate (that is `checks.py`) |
| `synth.py` | synthetic market + trades with analytic ground truth | anything realistic beyond what the tests need |
| `features.py` | plain feature functions with `asof`; the truncation test | any model fitting |
| `labels.py` | composite score, EWMA, level × direction labeller with a partition table, kink boundary estimate, `calibrate_states` (the state finder), transition counts, durations | HMMs (a challenger, out of scope here) |
| `profile.py` | the characteristics vector (own-pair, cross-pair, cross-asset), per-state profile, distinguishing ranking, today's placement | any model |
| `discover.py` | sub-state discovery inside a named state: k-means, BIC proxy, min-episodes and half-split stability conditions, auto-naming, assignment | deciding whether a sub-state changes the number (that is WU-11) |
| `combine.py` | vega-unit scaling of base legs; exact aggregation of legs into combinations; linear EV of any combination; a screen over candidate combinations | package-level hedging effects (the source hedges per leg, so there are none) |
| `ladder.py` | cumulative, ladder, HAC and block-bootstrap intervals, increments, shrinkage, persistence split, episodes | any forecasting of transitions |
| `evaluate.py` | walk-forward comparison against the unconditional benchmark | the continuous-feature benchmark (WU-11 adds it) |
| `gates.py` | reads `configs/gates.yaml`, evaluates pass/fail | set thresholds |
| `checks.py` | integrity checks on the table and label coverage | research judgements |
| `report.py` | tables, plots, the per-component card | a UI |

## States: what the synthetic world shows

The generator has five states (carry, rising, crisis, normalising, settling = level × direction) and
two hidden types of "rising" (dollar-led risk-off with strong cross-asset correlation and high
dollar-factor share; idiosyncratic with neither). Three results from it shape the design:

1. **Direction is a P&L fact, not a vol fact.** Scored on forward realised vol, the finder prefers
   three states (0.14 vs 0.13 out-of-sample R²). Scored on the straddle's forward 5-day earn, it
   prefers five (0.17–0.20 vs 0.09). The finder must therefore be run against forward outcomes,
   and Gate 2 is defined on them.
2. **The finder needs a switch budget.** Without one it picks zero hysteresis and labels that flip
   every few days; with `max_switches_per_year` it lands near the true switch rate.
3. **Discovery finds what is there and nothing else.** Inside "rising" it returns two sub-states at
   ~78% agreement with the hidden type, auto-named by what distinguishes them; inside carry and
   crisis it returns one. Stability across halves, not BIC, is the criterion that does this.
   Inside *estimated* carry it also surfaces a cluster that is mislabelled settling — discovery
   doubles as a label-impurity detector.

## What the synthetic validation shows

`python -m regime_ladder validate` simulates a market with a known 5-state regime path, trades
whose daily earn depends on the regime and on remaining tenor, a shared daily shock across all
live trades (overlap dependence) and AR(1) idiosyncratic noise, then runs the ladder with the
*true* labels and compares against the analytic expectation. Across seeds, nominal 90% intervals
cover the truth about 87–89% of the time and the median |error|/se is about 0.7. That is the
evidence that the statistics are implemented correctly; it says nothing about real data.

The demo (`python -m regime_ladder demo`) runs the same pipeline with *estimated* labels. On the
synthetic data the labeller agrees with the true regime roughly 60% of the time — the transition
regime lasts about nine days on average against a three-day smoothing half-life — and the Phase 2
and 3 gates fail where they pass with true labels. The labeller, not the ladder, is the bottleneck,
which is exactly what to expect on real data too and why Gate 2 comes before Gate 3.

## Reading the outputs

- `ladder.csv`: one row per (pair, archetype, tenor) × regime × h. `regime == "ALL"` is the unconditional benchmark. `n_eff` and `episodes` are the honesty columns.
- `increments.csv`: expected earn per day between consecutive horizons; the aged-position rate.
- `persistence_split.csv`: `ev_stayed`, `ev_broke`, `q_broke`; the identity holds exactly.
- `walk_forward.csv`: out-of-sample improvement over the unconditional mean, rank IC, calibration slope, per horizon.
- `card.md`: the per-component view a trader would read. Cash per unit notional; interval; episodes.

## Extending

Phase 4 (regime-conditional surface-move bootstrap through a repricer, time-varying transitions,
leading features) is deliberately not in this scaffold. When it is built, its first test is that,
run from historical entry dates with the base transition matrix, it reproduces `ladder.csv`
within Monte Carlo error at every horizon.
