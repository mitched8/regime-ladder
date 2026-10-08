# Components in isolation

Everything in this scaffold is a plain function on a daily market frame. None of it needs the
backtester except the ladder itself and the forward-P&L target for the finder and the leading
screen (both fall back to a forward surface quantity). So every component — the state features,
the composite and the labeller, the calibrated transition matrix, the tags, the characteristics
vector and profiles, sub-state discovery, the shock detectors, the leading features and stored
energy, the tilt, the pressure index — can be computed, plotted and interrogated on its own, in a
notebook, from a frame you already have in the wider workspace.

## Two ways in

**1. One command, every quantity as a file.**

```
python -m regime_ladder inspect --market market.parquet [--td trade_days.parquet] --pair EURUSD --out out/inspect
```

writes ~26 files and a `README.md` that names, for each file, the function that produced it (so
the notebook call is one line away). The calibrated transition matrix is `transition_matrix.csv`
(with the sticky prior) next to `transition_counts.csv` (raw); the finder's full candidate table is
`finder_report.csv`; the labeller's intermediate series — composite, EWMA, slope, level band,
direction band, state — are `composite.csv`; the leading features and the stored-energy product
are `leading_features.csv`; the per-fold tilt fits are `tilt_fits.csv`.

**2. The functions directly.** The market frame is the only input. Column names are the generic
ones the adapter maps onto (`atm_1m, atm_1y, rr25_1m, fly25_1m, spot, rv_1w, rv_1m, rv_3m`, the
G10 panel, the cross-asset series); see `docs/LOCALISATION.md`.

```python
import pandas as pd
from regime_ladder import features, labels, tags, profile, discover, shock, transitions, leading

m = pd.read_parquet("market.parquet")              # or the frame from your own plumbing

# state features and the labeller, step by step
F     = features.build(m)                           # every registered feature; features.FEATURES is the registry
comp  = labels.composite(F[["vol_pct", "term_slope_pct", "rr_stress_pct"]])
cal   = labels.calibrate_states(comp, target=m["rv_1m"].shift(-21), level_target=m["atm_1m"].shift(-5))
cal["report"]                                       # every candidate spec with its OOS R²
spec  = cal["specs"]["6"]                           # or "3", "6x"; cal["spec"] is the finder's own pick
lab   = labels.apply_spec(comp, spec)

# transitions
P     = labels.transition_matrix(lab, prior_strength=10)
transitions.implied_vs_observed_duration(P, lab)
transitions.fan(P, pd.Series({lab.iloc[-1]: 1.0}))  # P(state at h) from today, constant matrix

# tags
tags.corr_sign_tag(m); tags.pinned_tag(m); tags.intervention_risk_tag(m, direction=1, move_threshold=0.05)
reg, cells = tags.tag_split(lab, tags.corr_sign_tag(m))

# what a state looks like, and what is hidden inside it
ch    = profile.characteristics(m)
prof  = profile.state_profile(ch, lab); profile.describe(prof, "agitated")
discover.discover(ch, lab, "agitated")              # k, stability, auto-names, per-day sub-state

# shocks
z     = shock.surprise(m)                           # return over implied daily vol marked before it
shock.cusum_series(m); shock.bocd_series(m); shock.har_series(m)
S     = shock.shock_score(m); shock.lead_profile(S["score"], lab)

# leading features and stored energy
leading.gap_z(m); leading.gap_pct(m); leading.complacency(m); leading.stored_energy(m)
L     = leading.build(m)                            # all candidates, 0..100; leading.LEADING is the registry
res   = leading.incremental_value(L["stored_energy"], lab, comp, target=m["rv_1m"].shift(-21) - m["atm_1m"])
scr   = leading.screen(L, lab, comp, target, cfg={"min_delta_r2": 0.005, "min_folds_up": 3, "min_transition_gain": 0.002})
psi   = leading.pressure(L, {"gap_pct": 1.0})
fit   = transitions.fit_tilt(lab, psi, P, k=5, base=(comp - comp.mean()) / comp.std())
```

Every one of these functions takes `asof=` and uses trailing data only; `features.truncation_agrees(fn, m, cut)`
is the test that proves it for any function you add.

## Plugging in your own series

A feature of yours is one function `f(market, asof=None, **params) -> pd.Series` registered in
`features.FEATURES` (state), `tags.TAGS` (tag) or `leading.LEADING` (leading). Registration is all
it takes: the truncation test parametrises over the registry, the leading screen scores every
registered candidate against the same baseline, and `inspect` writes it out with the rest. The
desk's own series — flow, dealer gamma, the barrier book, positioning — enter exactly this way,
from the untracked adapter, and nothing in the tracked repo names them.

## What is and is not calibrated

| quantity | how it is set | where to see it |
|---|---|---|
| level boundaries, extreme bound | step fit on forward ATM over the training window; tail quantile | `finder_spec.json`, `finder_report.csv` |
| partition, half-life, hysteresis, direction thresholds | OOS separation of forward P&L under the switch budget | `finder_report.csv` (every candidate), `finder_r2.json` |
| transition matrix | counts on the label history + sticky prior (`transitions.prior_strength`) | `transition_matrix.csv`, `transition_counts.csv`, `durations.csv` |
| tilt β, γ | maximum likelihood of the k-step transitions, per fold and on the whole history | `tilt_fits.csv`, `tilt_pressure.json` |
| CUSUM threshold h | average-run-length calibration by resampling the first year of squared surprises | `configs/leading.yaml › shock`, printed by `shock.cusum_series` |
| HAR coefficients | ridge toward persistence, refitted every 63 days on a 756-day window | inside `shock.har_series`; expose with `shock.har_fit` |
| shrinkage κ, block length | config (`kappa`), 2h; D7 says how κ is to be chosen on real data | `configs/default.yaml` |
| which leading features enter the pressure index | the retention rule (both tests, most folds) | `leading_screen.csv` |
| tag thresholds, event calendar, intervention config | declared by the desk | `configs/default.yaml › tags`, `configs/local.yaml` |

## Handing a component to a reviewer without the code

`python -m regime_ladder pack --stage {data,states,ladder,leading} --src out/inspect [--market ...] [--td ...]`
turns an `inspect` folder into one markdown packet (definitions, dated runs, compact tables, gate
results, and for leading features an alignment check against future realised vol). Paste it into a
separate model session with `templates/CHALLENGER_PROMPT.md`. Leading features need only the market
frame and labels: run `labels`, `inspect --market ...` and `pack --stage leading`, no backtester.

## Working on the states in isolation

`python -m regime_ladder sweep --market m.parquet --td td.parquet --pair EURUSD [--sweep configs/sweep.yaml]`
holds the trade table, the horizons and Gate 2 fixed and varies only the labeller: feature set, partition,
finder grid, direction window, finder target. Each variant is calibrated on the first 70% of the history;
`gate2_fit` is Gate 2's separated-cell fraction on entries in that window and `gate2_confirm` on the entries
after it, which the finder never saw. The table also shows the chosen spec, switches per year, the thinnest
state, the duration-ratio range, the straddle's high-minus-low sign at h = 5, whether the spec sits on the
edge of its own grid (`on_grid_edge`: widen the grid) and whether the asked-for partition fell back to a
smaller one (`fell_back`: too few episodes). `labels_<name>.parquet` per variant feeds `ladder` directly; `view --sweep out/sweep` adds a Sweep tab (the
table with Gate 2 bars and one label strip per variant) and the Finder tab shows what the fit saw for the pair in
use: the targets by composite decile with the bounds drawn, the separation by named state and by each axis alone,
and the finder's whole grid with the chosen cell marked. `pack --stage sweep --src out/sweep` writes the packet
for the challenger (checklist W1–W6). Adopt one variant into `configs/default.yaml › labeller` with a DECISIONS line, then
stop sweeping: every run reads the same forward P&L.

## The explorer: one object, one operation

The Explore tab of `view` is the front door. The object is the per-entry table `inspect` writes to
`entries.csv`: for every fresh unit entered on a date, its cumulative P&L at each horizon, total and by
component, divided by the unit's own vega at inception (or notional). The operation is to condition on a
daily statistic from `statistics.csv` (about 70 of them: surface and spot, the state features, the state
score and named state, the leading candidates, the shock detectors, the profile characteristics, and the
strategy's own recent earn), with a threshold the trader sets: "entries made when the ATM z-score was
above 2". The named states, the tags and the finder are all instances of that operation.

Every condition carries its own honesty: n and the number of runs it was on; on-minus-off with an
overlap-adjusted interval; the same difference on the first 70% and the last 30% of the history; and a
reference band from 200 circular shifts of the condition's own mask (same on-fraction, same run lengths),
so "unusual" means unusual for a condition of that shape and not merely non-zero. Drivers are the
components, on against off.

Below the condition, **across the values of a statistic** cuts the same earn into equal-count buckets of any
statistic (terciles to deciles; the categories, for a categorical one): mean earn with its interval per bucket,
first 70% against last 30%, runs, the components, and the rank correlation of bucket order with earn. A
monotone pattern across buckets is harder to get by chance than one good bucket. Clicking a bucket makes it
the condition.

**Adding a statistic** takes one of three steps, in order of effort: a raw market column goes under
`explore: passthrough:` in the config with a one-line definition; a function of the market frame goes in
`explore.EXTRA` as `name: (fn, definition, group)`, trailing data only (the truncation test covers every
entry); site-specific series (flow, positioning) go in an untracked `adapters/statistics.py` exposing
`extra(market) -> (frame, definitions)`, which is imported if present and never committed. Rerun `inspect`
and the statistic is in every dropdown.

A condition worth keeping is saved as a few lines of YAML under `filters:` in the config. `inspect` turns
it into a tag (`explore.filters_to_tags`), so it splits the ladder's cells, appears in the packets and is
tested by the walk-forward like any other tag. Nothing reaches the card from the explorer directly.

## Looking at the results without a notebook

`python -m regime_ladder view --src out/inspect [--src out/inspect_usdjpy ...] --out out/view.html`
writes one HTML file from the `inspect` output (`--src` is an inspect folder or a parent holding one
per pair; run `inspect` once per pair). It embeds the data and draws with plain SVG, so it opens from
a file on a locked-down machine with no server, network or library. Nothing is recomputed: every
number on the page is in an `inspect` file.

| tab | what it shows | the question it answers |
|---|---|---|
| Explore | the per-entry earn series with an EWMA and components, any statistic with a threshold, on vs off by horizon and by component, the honesty panel, save-as-filter | what conditions at entry change the earn of this strategy, and is that more than a random condition of the same shape would show |
| Overview | today's state per pair, the chosen spec, gate results, retained leading features, the strongest clean cells at h=5 | where does the project stand |
| Ladder | pivot over the ladder: any of pair, strategy, tenor and entry state as lines or panels, horizon or tenor on the x axis, EV / shrunk EV / P(profit) / quantiles / ES / earn per day, optionally minus the unconditional row, with the interval band | how do strategies, tenors and pairs compare conditional on the entry state |
| Matrix | rows = pair × strategy × tenor, columns = entry states, value at a chosen horizon, thin cells marked, today's state outlined; click a cell for its ladder | what would I carry today |
| Out of sample | walk-forward improvement, rank IC and calibration slope by horizon, same pivot | where does the regime layer beat the unconditional mean |
| States | composite and smoothed composite over the state bands, direction score, durations, transition matrix, share of days, state profile | do the states look like their names |
| Transitions | the fan from today's state, constant and tilted; tilt fits by fold | what does the matrix say about the next month |
| Shock | score and its parts with CUSUM alarms over the state bands, surprise, HAR forecast, lead profile | does the fast path fire before the label moves |
| Leading | the screen, chosen features over the state bands, the pressure index | do the leading features rise before the move |

When a comparison axis changes, the other dimensions are pinned to one value (shown in the filter
chips) so a colour means one thing; shift-click a chip to solo it, click to add. The page is for the
owner; the challenger still reads the packet.

## Adding a feature

One function, one dict entry, one config line. The function takes the market frame and `asof`, uses
trailing data only, and returns a Series named after itself:

```python
def my_feature(market, asof=None, window=63):
    m = market if asof is None else market.loc[:asof]
    return rolling_percentile(<something trailing>, 756).rename("my_feature")

leading.LEADING["my_feature"] = my_feature          # or features.FEATURES for a state feature
# configs/leading.yaml › leading.features: my_feature: {window: 63}
# optional: feature_targets: {my_feature: rr_25d} if it is signed
```

From there `tests/test_pit_truncation.py` covers it, the screen reports it, `inspect` writes it and
`pack` shows it. A desk series goes the same way from the adapter (see `docs/LOCALISATION.md`), and
the leading packet's alignment check (section 8b) is the first thing to read for one.

The screen itself: `leading.screen(candidates, labels, composite, targets, gates["phase4_leading"])`,
where each argument may be a dict keyed by pair for the pooled test. `leading.lift(feature, labels)`
is the event-study view for a sparse feature.
