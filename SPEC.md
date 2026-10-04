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
level (low / mid / high) and direction (down / flat / up) — mapped through a partition table with
hysteresis: **carry, rising, crisis, normalising, settling**. A three-state level-only partition
is kept as the benchmark. The partition, boundaries, direction thresholds, smoothing and
hysteresis are chosen by `labels.calibrate_states` on a training window, scored by out-of-sample
separation of a *forward outcome*: the archetype's forward 5-day P&L where the backtester covers
the window, forward surface change otherwise. On synthetic data, direction earns its place against
P&L and not against forward vol; the finder is built to show that rather than assume it.

**Sub-states** may be discovered inside a named state (`discover.py`) by clustering days on the
characteristics vector (`profile.py`: own-pair spot-vol behaviour, realised vs implied, skew and
wings; dollar-factor share and cross-pair correlation; correlations of spot and vol with equities,
rates, commodities, credit and vol indices). A sub-state is descriptive unless it has ≥ 5
episodes, is stable across halves of the history, and — separately — conditioning on it beats the
parent state's ladder out of sample. The **state profile** (what distinguishes a state, and where
today sits within it) is always shown.

## 4. Units

Cash per unit of standard notional (defined per archetype in `configs/archetypes.yaml`). Never
percentage of premium.

## 5. Probability model (Phase 3)

Empirical and non-parametric, conditional on the entry-date label. Means shrunk toward the
unconditional mean **at the same horizon** with strength κ; tail statistics reported unshrunk.
Intervals by circular block bootstrap with blocks ≈ 2h (default) or Newey–West with bandwidth h.
Effective N ≈ n · min(1, entry spacing / h). Episodes = contiguous runs of the regime in the
label series.

## 6. Regime dependence

The empirical split `EV(h | entry regime persisted)`, `EV(h | it broke)` and `q = P(broke)`,
reported side by side, signed, in cash. `(1−q)·EV_stay + q·EV_broke = EV` exactly. It is a
decomposition for explanation, not a forecast. Never multiplied together; never a ratio.

## 7. Falsification tests — each one is a stop

1. **Phase 1.** Integrity checks fail, or golden trades do not reproduce → fix data before any statistic.
2. **Phase 2.** Entry labels do not separate h ≤ 10 outcomes out of sample with stable sign → the labeller is not describing anything the P&L sees.
3. **Phase 3.** The ladder does not beat both the unconditional statistic and a direct regularised model on the continuous state features, out of sample at h ≤ 10 → the regime layer adds nothing; stop at the unconditional product.
4. **Phase 4 (future).** A path simulator that cannot reproduce the Phase 3 ladder from historical entries is wrong and nothing is built on it.

Thresholds live in `configs/gates.yaml`; the agent reads them and never edits them.

## 8. Out of scope for this scaffold

Transaction costs (not modelled, not estimated, not displayed). Exit rules. The Phase 4 path model
and repricer. Any production concern. Anything that names or depends on a specific data owner's
systems — that lives only in the local, untracked `adapters/` and `configs/local.yaml`.
