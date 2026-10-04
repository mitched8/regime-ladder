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
