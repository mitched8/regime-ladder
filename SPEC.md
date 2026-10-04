# SPEC — what this scaffold estimates

This is the contract. The agent builds from this file and the current work-unit card in
`PLAN.md`, not from any longer document. If a component does not help estimate or validate the
quantity below, it does not belong in the build.

## 1. Target quantity

For an archetype **A** (a fully specified option structure with its hedge rule), an entry date
**t** with entry state **s_t** (the point-in-time regime label, optionally a small vector of
continuous state features), and a horizon **h** in trading days:

```
F_A(h | s_t)   = distribution of net P&L from entry at t to the close of day t+h,
                 in cash per unit of standard notional, under the stated hedge rule,
                 held for h days with no exit rule.
EV_A(h | s_t)  = its mean — the headline.   h ∈ {1, 3, 5, 10, 20, T}, capped at tenor T.
```

Reported alongside the mean: a 90% interval reflecting estimation uncertainty with overlap
respected; quantiles, P(profit) and ES(5%) of the outcome distribution; effective sample size
and the number of independent regime episodes behind each cell.

## 2. Exposure assumption

None beyond the archetype definition. Every observation starts from a fresh archetype at entry and
ages exactly h days through the backtester's own repricing and hedging. **No daily rate is rolled
forward; nothing is scaled.** Within-horizon regime transitions are inside the measured outcomes;
**no transition or switch term is ever added.**

## 3. Interpretation for a market-making book

The archetypes are the smile-vega-neutral components a book decomposes into. The ladder is read as
the marginal expected earn of carrying one unit of a component for the next h days from today's
state. Short horizons (1–5d) are primary; the increments between horizons (`ladder.increments`) give
the expected earn of an aged unit, which is how the ladder applies to existing inventory.

## 3a. States

The entry state is a small set of **named states** from two axes of a composite stress score —
level (low / mid / high, with an optional **extreme** band above) and direction (down / flat / up) —
mapped through a partition table with hysteresis and previous-state dependence: **carry, rising,
agitated, stressed, normalising, settling**. *Agitated* is the elevated, choppy market that goes
nowhere for weeks — a state in its own right, not a transit lounge between carry and crisis; a
full crisis is rare in G10, so *stressed* is the high band and *extreme* is a tail band that is kept
as a state only when it has enough episodes (≥ 5) and is otherwise merged into stressed and
surfaced as a flag. A three-state level-only partition (carry / transition / crisis) is kept as the
benchmark.

Two targets, two jobs (`labels.calibrate_states`). **Level boundaries** are a surface fact: they are
step-fitted (piecewise-constant, every band ≥ 8% of days) on a forward vol level — ATM a week ahead
— over the whole price history; the extreme bound is a training-window tail quantile (97.5%).
**Partition, direction thresholds, smoothing and hysteresis** are a P&L fact: chosen by out-of-sample
separation of the archetype's forward 5-day P&L under a switch budget, with level-only as the
challenger. A kink (piecewise-linear) fit is not used for the bounds: it puts its breaks at the ends
of a ramp, not in the middle of it. On synthetic data the finder places the bounds between the
true bands, prefers the direction partitions on P&L, and merges or keeps extreme by the episode rule.

**Tags** (`tags.py`) qualify a state without changing it, for the kinds of agitated market the
labeller cannot tell apart: the sign of spot-vol correlation, a scheduled-event window, a pinned
market (realised far under implied in a narrow range), and a pair-specific intervention-risk flag
(fast one-way move, level, official comment). A tag splits a state's ladder into *state·tag* cells
only where the cell has ≥ 5 episodes; otherwise it collapses back to the parent. Neutral tag values
never split.

**Sub-states** may be discovered inside a named state (`discover.py`) by clustering days on the
characteristics vector (`profile.py`: own-pair spot-vol behaviour, realised vs implied, skew and
wings; dollar-factor share and cross-pair correlation; correlations of spot and vol with equities,
rates, commodities, credit and vol indices). A sub-state is descriptive unless it has ≥ 5
episodes, is stable across repeated episode-level splits of the history (≥ 0.7 agreement), and —
separately — conditioning on it beats the parent state's ladder out of sample. Tags are declared
by the desk; sub-states are found without being told what to look for. The **state profile**
(what distinguishes a state, and where today sits within it) is always shown.

## 3b. Base legs, combinations, tenors

The primitives are the **base legs** the backtester runs, per tenor: ATM straddle, 25d put, 25d
call, 10d put, 10d call, each hedged leg by leg. Everything else is a **combination** — a weight
vector over base legs entered on the same date. Because hedging is per leg and P&L is per leg per
day, a combination's daily P&L is exactly the weighted sum of its legs' (`combine.combine`); its
ladder is the ladder of that series, with no new backtest runs. Means are linear, so the
state-conditional EV of any combination is the weighted sum of the legs' (`combine.ev_linear`);
intervals, quantiles and ES are computed on the combined series for the combinations actually
examined. Standard combinations (RR, fly at 25d and 10d) are declared in `configs/archetypes.yaml`.

Any major tenor from 1W to 1Y is available; the ladder is per (pair, leg or combination, tenor)
and horizons are capped at tenor. At a fixed short horizon, EV across tenors is a **tenor curve**
per component (`report.plot_tenor_curve`).

## 4. Units

Cash per unit of **vega at inception** for each base leg (`combine.to_vega_units`), so that
combination weights are vega units and a zero-sum weight vector is smile-vega-neutral by
construction; this is the market-making decomposition (a vega position plus smile tilts) and makes
EV comparable across legs and tenors as earn per unit of vega risk. Where no per-day vega is
available, the source's standard notional per `configs/archetypes.yaml`. Never percentage of
premium. Alternative scalings (equal notional, premium-neutral, equal ES contribution) are a
config choice, not the default (DECISIONS D14).

## 5. Probability model (Phase 3)

Empirical and non-parametric, conditional on the entry-date label. Means shrunk toward the
unconditional mean **at the same horizon** with strength κ; tail statistics reported unshrunk.
Intervals (default) are the wider of two bootstraps centred on the mean: a circular block bootstrap
with blocks ≈ 2h, for the overlap dependence of consecutive entries, and a cluster bootstrap over
the regime episodes the entries fall in, for the shared-fate dependence inside an episode (entries
in one episode all live through the same exit). Newey–West with bandwidth h is the fast
alternative. Effective N ≈ n · min(1, entry spacing / h). Episodes = contiguous runs of the regime
in the label series — the honesty column, and the unit the second bootstrap resamples.

## 6. Regime dependence

The empirical split `EV(h | entry regime persisted)`, `EV(h | it broke)` and `q = P(broke)`,
reported side by side, signed, in cash. `(1−q)·EV_stay + q·EV_broke = EV` exactly. It is a
decomposition for explanation, not a forecast. Never multiplied together; never a ratio.

## 6a. Shock detection (`shock.py`)

The labeller is smoothed and hysteretic, so it is late by construction. Shock detectors do not
define the state; they move probability mass toward the next one (`blend_pi`, a heuristic until the
calibration test in §7 says otherwise) and feed the regime probability vector of the path model.
Two quantities are kept apart: the **economic surprise** (the day's return over the implied daily vol
marked before it — the move relative to what the market priced, which is what option P&L is
realised against) and a **physical forecast** of realised vol (a HAR regression on realised vols,
ridge-shrunk toward persistence, refitted on fixed dates). Detectors on squared surprises: a
one-sided **CUSUM** against a trailing empirical null (implied-standardised squared returns do not
have mean one), squared surprises capped at 3σ² so an isolated day cannot trip it, threshold set
on an average-run-length basis (default one false alarm in ~4 years); and **Bayesian online
changepoint detection** supplying P(run length < m). The 3σ rule is kept only as the thing to beat.

## 6b. Transition probabilities (`transitions.py`)

For a fresh unit no transition model is needed or used. The matrix earns its place for explaining
the ladder, the path model, and incorporating today's leading signals. Time variation is a
row-normalised multinomial logit — P_ij(ψ) ∝ P0_ij · exp(β ψ (rank_j − rank_i)) — the only valid
form. β is fitted by maximum likelihood of the **k-step** transitions (k = 5: where the state is a
week out, not tonight's move, because a smoothed labeller's one-step moves lag the market by days)
with the continuous state score as a baseline covariate already in the tilt, so β is the signal's
incremental effect. A covariate is retained only if it improves the out-of-sample k-step likelihood.
Forecasting from today needs a declared future path per covariate — frozen, decaying, calendar,
zero; a frozen path is a sustained-pressure scenario, not a forecast.

## 6c. Leading features and stored energy (`leading.py`)

State features say where we are; leading features say what might push us out. Two hypotheses are
made measurable: **stored energy** — spot's distance from its trailing anchor in implied-vol units
(gap) multiplied by how temporary the holder of that gap looks (complacency: realised far under
implied, vol-of-vol low, front end cheap) — and **feedback** — jump clustering, a shift in spot-vol
coupling, cross-pair coherence. The scheduled-event calendar is the one covariate known in advance.
All are trailing percentiles obeying the `asof` contract and the truncation test.

The retention rule is the whole point: option P&L is realised relative to implied, so a signal the
surface already prices adds nothing. A feature is retained only if, out of sample and in most folds,
it (1) improves the k-step transition likelihood through the tilt **and** (2) improves the forward
outcome fit beyond a baseline that already uses the named state and the continuous state score.
Features are tested one at a time; the retained ones are averaged into the **pressure index** that
enters the tilt. On synthetic data where stored energy is made causal, the true pressure is retained
with true labels and with estimated labels; in the null world nothing is. The first draft's closure
weight and Hawkes fragility gauge stay parked for the reasons given in the framework.

## 7. Falsification tests — each one is a stop

1. **Phase 1.** Integrity checks fail, or golden trades do not reproduce → fix data before any statistic.
2. **Phase 2.** Entry labels do not separate h ≤ 10 outcomes out of sample with stable sign → the labeller is not describing anything the P&L sees.
3. **Phase 3.** The ladder does not beat both the unconditional statistic and a direct regularised model on the continuous state features, out of sample at h ≤ 10 → the regime layer adds nothing; stop at the unconditional product.
4. **Phase 4 — leading.** No leading feature passes the retention rule, or the pressure index built from the retained ones does not improve the out-of-sample k-step transition likelihood → the surface already prices what the features say; the constant matrix stands.
5. **Phase 5 (future).** A path simulator that cannot reproduce the Phase 3 ladder from historical entries is wrong and nothing is built on it.

Thresholds live in `configs/gates.yaml`; the agent reads them and never edits them.

## 8. Out of scope for this scaffold

Transaction costs (not modelled, not estimated, not displayed). Exit rules. The path model and
repricer (Phase 5). Any production concern. Anything that names or depends on a specific data owner's
systems — that lives only in the local, untracked `adapters/` and `configs/local.yaml`.
