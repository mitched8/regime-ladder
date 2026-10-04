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
| `labels.py` | composite score, EWMA, hysteresis thresholds, kink boundary estimate, transition counts, durations | HMMs (a challenger, out of scope here) |
| `ladder.py` | cumulative, ladder, HAC and block-bootstrap intervals, increments, shrinkage, persistence split, episodes | any forecasting of transitions |
| `evaluate.py` | walk-forward comparison against the unconditional benchmark | the continuous-feature benchmark (WU-11 adds it) |
| `gates.py` | reads `configs/gates.yaml`, evaluates pass/fail | set thresholds |
| `checks.py` | integrity checks on the table and label coverage | research judgements |
| `report.py` | tables, plots, the per-component card | a UI |

## What the synthetic validation shows

`python -m regime_ladder validate` simulates a market with a known 3-state regime path, trades
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
