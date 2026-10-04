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
| `labels.py` | composite score, EWMA, level × direction labeller with a partition table and previous-state dependence, step-fit boundary estimate, extreme band and merge rule, `calibrate_states` (the state finder), transition counts, durations | HMMs (a challenger, out of scope here) |
| `tags.py` | corr-sign, event-window, pinned and intervention-risk tags with `asof`; the episode-gated split of a state's ladder into state·tag cells | define states or find sub-states |
| `profile.py` | the characteristics vector (own-pair, cross-pair, cross-asset), per-state profile, distinguishing ranking, today's placement | any model |
| `discover.py` | sub-state discovery inside a named state: k-means, BIC proxy, min-episodes and repeated episode-split stability conditions, auto-naming, assignment | deciding whether a sub-state changes the number (that is WU-11) |
| `combine.py` | vega-unit scaling of base legs; exact aggregation of legs into combinations; linear EV of any combination; a screen over candidate combinations | package-level hedging effects (the source hedges per leg, so there are none) |
| `ladder.py` | cumulative, ladder, HAC and block-bootstrap intervals, increments, shrinkage, persistence split, episodes | any forecasting of transitions |
| `evaluate.py` | walk-forward comparison against the unconditional benchmark (raw or shrunk state means) | the continuous-feature benchmark (WU-11 adds it) |
| `shock.py` | economic surprise, ARL-calibrated CUSUM on capped squared surprises, BOCD, ridge-HAR physical forecast, shock score, the blend heuristic, lead profile | define the state |
| `transitions.py` | constant matrix diagnostics; multinomial-logit tilt; k-step maximum-likelihood fit with a baseline covariate; out-of-sample gain; forecast fan under declared covariate paths | a hidden-Markov model |
| `leading.py` | stored-energy and feedback features with `asof`; event proximity; pressure index; one-at-a-time incremental-value screen and the retention rule | any feature the desk has not registered through the same contract |
| `gates.py` | reads `configs/gates.yaml`, evaluates pass/fail | set thresholds |
| `checks.py` | integrity checks on the table and label coverage | research judgements |
| `report.py` | tables, plots, the per-component card | a UI |

## States: what the synthetic world shows

The generator has six named states (carry, rising, agitated, stressed, normalising, settling =
level × direction) plus a rare extreme state (one or two episodes a decade), hidden types of
"rising" (dollar-led with strong cross-asset correlation; idiosyncratic) and hidden types of
"agitated" whose spot-vol correlation sign flips between episodes. Results from it that shape the
design:

1. **Level bounds must be step-fitted on a vol target.** A kink (piecewise-linear) fit on P&L put the
   first boundary at 15 on a 0–100 composite and labelled most of carry as agitated (18% agreement
   with the true state). A piecewise-constant fit on forward ATM places the bounds between the true
   bands (≈ 45 / 73 against band means of 22–36 | 55–70 | 78) and agreement rises to 51–59%. On P&L
   alone the high boundary is lost because stressed earns and normalising loses — the high band
   looks like the mid band. Two targets, two jobs.
2. **Direction is a P&L fact.** With the bounds fixed, the six-state partitions beat level-only on
   forward P&L out of sample (R² 0.29–0.34 vs 0.24–0.27 across seeds); on forward realised vol the
   level-only partition wins on two seeds of three. Gate 2 is defined on P&L.
3. **The finder needs a switch budget**, or it picks zero hysteresis and labels that flip every few
   days; with `max_switches_per_year` it lands near the true switch rate (15–22 a year).
4. **Extreme behaves as a flag until it has episodes.** With 2–4 episodes in ten years the merge
   rule folds it into stressed; on the one seed with six episodes it is kept. Nothing conditions on
   eleven days.
5. **Discovery finds what is there and nothing else.** Inside "rising" it returns two sub-states at
   ~85% agreement with the hidden type; inside "agitated" two, at ~73% agreement with the hidden
   correlation sign; inside carry and stressed one. Stability across repeated episode-level
   splits, not BIC, is the criterion that does this (a single time half-split sat at 0.67 on one
   seed, just under the bar, for a split that is real).
6. **Tags recover the correlation flip directly.** The corr-sign tag labels the hidden `corr_neg`
   agitated days "neg" and the `corr_pos` days "pos" by better than two to one, and the split
   ladder shows the risk reversal earning very differently in the two cells.

## Shocks, transitions and leading features: what the synthetic world shows

7. **A single-day rule is the wrong detector.** On a calm path with three isolated 3σ days followed
   by realised running 50% above implied with no single day above 3σ, the 3σ rule raises three
   false alarms and misses the change; the CUSUM (ARL₀ 1000, squared surprises capped at 3σ²)
   raises none and detects it in 14–20 days across seeds. BOCD resets within ten days on a clean
   variance change and is deliberately slow on a modest one.
8. **The k-step likelihood with a baseline covariate is what makes a lead measurable.** In the
   world where stored energy is made causal, the *true* pressure is retained against true labels at
   k = 1. Against the labels the market actually produces it comes out with the *wrong sign* at
   k = 1 — pressure is highest when vol is low, i.e. far below the first boundary, where the
   labeller is least likely to move next — and is recovered (β > 0, positive gain in 4/4 folds) once
   the continuous state score sits in the tilt as a baseline and the likelihood scores the state five
   days out. The same baseline is in the outcome test. This is the general lesson: anything scored
   against a smoothed label must be given the continuous state first.
9. **The retention rule is strict on purpose.** At realistic effect sizes even the true pressure
   fails it on ten years of data; the engineered stored-energy proxies clear it only with true
   labels and a strong effect, and the null world retains nothing. Expect real leading features to
   need the full history, to be judged mostly on the transition test, and to fail more often than
   not. That is the design working.

## What the synthetic validation shows

`python -m regime_ladder validate` simulates a market with a known regime path, trades whose
daily earn depends on the regime and on remaining tenor, a shared daily shock across all live
trades (overlap dependence) and AR(1) idiosyncratic noise, then runs the ladder with the *true*
labels and compares against the analytic expectation. Across seeds, nominal 90% intervals cover
the truth about 88% of the time and the median |error|/se is about 0.7. The interval is the wider
of a block bootstrap and an episode-cluster bootstrap: with the block bootstrap alone, coverage at
h = 5–10 in the states with 15–20 episodes sat near 80%, because entries in one episode share its
exit and that dependence outlasts the h-day overlap the blocks are sized for. Resampling episodes
is what the "episodes" column has been promising all along. That is the evidence that the
statistics are implemented correctly; it says nothing about real data.

The demo (`python -m regime_ladder demo`) runs the same pipeline with *estimated* labels. On the
synthetic data the labeller agrees with the true regime roughly half the time — rising starts
inside the low band and is seen late — and the Phase 3 gate fails where it passes with true labels.
The labeller, not the ladder, is the bottleneck, which is exactly what to expect on real data too
and why Gate 2 comes before Gate 3. `--energy-beta 3` runs the demo in the world where stored
energy is causal.

## Reading the outputs

- `ladder.csv`: one row per (pair, archetype, tenor) × regime × h. `regime == "ALL"` is the unconditional benchmark. `n_eff` and `episodes` are the honesty columns.
- `increments.csv`: expected earn per day between consecutive horizons; the aged-position rate.
- `persistence_split.csv`: `ev_stayed`, `ev_broke`, `q_broke`; the identity holds exactly.
- `walk_forward.csv`: out-of-sample improvement over the unconditional mean, rank IC, calibration slope, per horizon.
- `card.md`: the per-component view a trader would read. Cash per unit notional; interval; episodes; the state profile, tags, shock score and leading line.
- `leading_screen.csv`: one row per candidate — baseline R², R² gain and folds up, transition gain and folds up, mean β, retain.
- `shock.csv`: surprise, CUSUM pressure, BOCD P(short run), recent |surprise|, score, HAR forecast.
- `transition_matrix.csv`, `fan_constant.csv`, `fan_tilted.csv`: the matrix and the 21-day state distribution from today, with and without the pressure tilt.

## Extending

A new state feature, tag or leading feature is one function with the `asof` contract, registered
in `FEATURES`, `TAGS` or `LEADING`; the truncation test picks it up by itself and the leading screen
scores it against the same baseline as everything else. The desk's own series (flow, dealer gamma,
the barrier book) enter exactly this way, from the untracked adapter.

Phase 5 (regime-conditional surface-move bootstrap through a repricer — the path model for an
existing book) is deliberately not in this scaffold. When it is built, its first test is that, run
from historical entry dates with the constant transition matrix, it reproduces `ladder.csv` within
Monte Carlo error at every horizon; the tilted matrix and the shock blend enter only after that.
