# Feature proposal — what to add, where it goes, how it earns its place

Status: proposal for owner review. Items that are accepted become work-unit cards in `PLAN.md`; the
slot each feature goes into is part of the decision.

Update 2026-10-04: of the plumbing in §7, items 2 (fold-local standardisation in the tilt), 3 (selection
control: permutation null plus a hold-out guard) and the pooled-pairs screen are implemented in
`leading.py` / `transitions.py` (see `DECISIONS.md` D22 and the gate-threshold log); `drift_t` and a
signed gap are registered; the synthetic extension has the external-leader world (`docs/calibration/D_external`).
Items 1 (state-feature selection) and 4 (per-column availability lag) remain open. No feature from
§3–§6 is implemented yet.

## 1. The problem this addresses

The registry is thin where it matters most for the three archetypes. Six state features describe
the ATM surface and its term structure; six leading features describe spot's distance from an
anchor and a complacent surface. Almost nothing describes how spot is *behaving* (trending,
ranging, breaking out), nothing describes what the smile has been *paying* relative to what spot
has *delivered* beyond the ATM, and nothing measures damping — whether a shock would die out. The
first two are the drivers of RR and fly P&L, which are two of the three archetypes.

The organising rule stays the same: option P&L is realised relative to implied. Every feature
below is either a surface fact (a level the ladder can condition on), a realised-vs-implied gap
(the thing the P&L is), a spot-behaviour fact (what gamma and vanna P&L are made of), or a
propensity signal (what might move the state). The slot it goes into follows from which of those
it is.

## 2. Slots, and the test each slot imposes

| Slot | What it is for | What a new entry must pass |
|---|---|---|
| **Level composite** (`labels.composite`) | Defines the named state. Surface facts only. | §5.3 selection: tercile spread in forward 5d P&L above noise with episode-bootstrap SE; sign stable in two non-overlapping sub-periods; rank-corr < 0.7 with every retained feature. Set frozen before the finder runs. |
| **Direction score** | The second axis of the state. | Finder chooses among candidates on OOS separation, inside training folds. |
| **Tag** (`tags.py`) | Qualifies a state without changing it; splits the ladder where episodes allow. | ≥ 5 episodes per (state, tag) cell; split ladder beats parent OOS (same test as sub-states). |
| **Leading** (`leading.py`) | Propensity to leave the state. | `incremental_value`: OOS Δ-R² on the forward target **and** k-step transition gain, over a baseline of named state + composite. One at a time. |
| **Damping** (new, in `leading.py`) | Propensity for a shock to be absorbed. | Same screen, but enters the tilt with a constrained negative sign and is tested on *downward* transitions and on realised < implied. |
| **Profile** (`profile.py`) | Description only. | None beyond the `asof` contract. Add freely. |

A feature is proposed into one slot. If it fails there it is not moved to a weaker slot to rescue it;
the profile is the only place a failed feature may remain, as description.

## 3. Tier 1 — no new data; computable from the current market frame and trade-day table

### 3a. Realised-versus-implied by smile dimension

The repo has a VRP feature for the ATM. The RR and fly archetypes have no equivalent. These are the
closest analogues to "what the P&L is" for those legs.

| Name | Construction | Slot | Why |
|---|---|---|---|
| `skew_rp` | Implied 25d RR (in vol) minus a trailing realised-skew proxy: the 63-day skewness of daily returns scaled to vol units, or the realised 25d RR from a delta-hedged backtest of the two 25d legs if the backtester supplies it | Tag (`skew_rich` / `fair` / `skew_cheap`) and leading | The RR's earn is the skew premium. A rich skew in carry and a rich skew in agitated are different trades. |
| `wing_rp` | Implied 25d fly minus a realised kurtosis proxy (63-day excess kurtosis of returns, or realised jump share) | Tag and leading | Same for the fly. Tail-hedging demand vs realised tails is what the fly earns or loses. |
| `har_minus_implied` | `shock.har_series` forecast of 21d realised vol minus `atm_1m` | Leading | Already computed in `shock.py`; never registered. It is a direct physical-vs-risk-neutral gap and the most natural leading feature in the codebase. |
| `vrp_momentum` | 10d change of `atm_1m − rv_1m` in units of its trailing sd | Direction candidate | A VRP that is collapsing is a transition in progress; a VRP that is widening is a surface pricing a move spot has not made. |

### 3b. Spot behaviour

This is the gap you identified. None of these touch the surface; they describe the path gamma and
vanna P&L is realised on. They do **not** go into the level composite: level is a surface fact. They
go in as a tag (today's spot mode) and as leading features (a break that has just happened).

| Name | Construction | Slot | Why |
|---|---|---|---|
| `range_break` | Spot's distance beyond the prior N-day high or low (N = 21, 63), signed, in units of `atm_1m · sqrt(N/252)`; zero inside the range. Trailing percentile of the absolute value. | Leading | A breakout is where barriers trigger, stops run and dealer gamma flips sign. It is the spark-meets-amplifier moment the anatomy describes, measured from price alone. |
| `range_position` | Where spot sits in its N-day range, 0..1, folded to distance from the nearer edge | Tag component and profile | Near an edge with low realised = pinned until it breaks; mid-range with low realised = carry proper. |
| `efficiency_ratio` | Kaufman: \|N-day return\| / Σ\|daily returns\| over N = 21 | Tag (`trending` / `ranging`) | At equal realised vol, a trending path moves a struck straddle away from its strike and starves it of gamma; a ranging path keeps it at the money. The pinned tag is a special case of ranging. |
| `variance_ratio` | Var(5-day returns) / (5 · Var(1-day)) over 126 days | Damping (< 1) and profile | Below one is mean reversion: active negative feedback. Above one is trend persistence. |
| `return_autocorr` | Lag-1 autocorrelation of daily returns over 63 days | Damping | Negative autocorrelation is a market that sells rallies and buys dips: dealers long gamma or level-based flows. |
| `drift_t` | Already built in `leading.py`; not registered | Leading | One-way markets build positioning. Register it. |

Proposal for the tag: replace `pinned` with a single three-valued `spot_mode` tag — `pinned`
(existing rule), `ranging` (low efficiency ratio, not pinned), `trending` (high efficiency ratio) —
so that one tag carries the spot story and cells are not multiplied.

### 3c. Surface features currently missing from the state

| Name | Construction | Slot | Why |
|---|---|---|---|
| `fly_pct` | Trailing percentile of `fly25_1m` | Level composite candidate | Wing demand is a surface fact the composite ignores; it is the fly archetype's own level. |
| `rr_norm_pct` | Percentile of `rr25_1m / atm_1m` | Level composite candidate, replacing `rr_stress_pct` if redundant | Raw RR scales with vol level; normalised skew is closer to orthogonal to `vol_pct`. |
| `volofvol_pct` | Percentile of 21d sd of ATM changes | Level composite candidate | Already inside `complacency` and the profile; a surface fact in its own right. Often moves before level. |
| `rr_change_10d` | 10d change of RR in units of its trailing sd | Direction candidate | The PDF's "levels identify, changes signal"; the current direction axis sees only the composite's slope. |

### 3d. Features from the backtester itself

The trade-day table is point-in-time P&L history. It is a legitimate feature source as long as only
realised rows (date ≤ today) are used.

| Name | Construction | Slot | Why |
|---|---|---|---|
| `vol_seller_pnl_21d` | Trailing 21d cumulative P&L of the *short* ATM straddle (negative of the long leg's cumulative), as a percentile | Leading | Premium supply. When systematic vol sellers are losing they withdraw and the surface reprices; when they are winning they add. This is mechanism E of the anatomy from data already in hand. |
| `wing_seller_pnl_21d` | Same for the short fly | Leading | The fly version. |

### 3e. Technical analysis as positioning and path proxies

Technical indicators are not used here as directional forecasts. They earn a place for three
reasons that fit the anatomy: systematic trend-followers and retail trade them at scale, so they
are **where positioning sits and when it flips** (fuel); stops and barriers cluster at the levels
they define (amplifier); and compression / expansion indicators are the TA vocabulary for
**stored energy and its release**. Everything below is trailing and passes the `asof` contract.
All of it is spot behaviour, so nothing goes into the level composite.

| Name | Construction | Slot | Mechanism |
|---|---|---|---|
| `cta_position` | Sign-weighted sum of moving-average crossovers at the standard trend-following lookbacks (e.g. 10/50, 20/100, 50/200) plus 1m/3m/12m return signs, scaled to −1..+1. A public replication of the trend-follower position. | Leading (fuel) | Crowded trend positioning is what gets hurt on a reversal. Its **5-day change** is the flow; its **level** is the stock. |
| `cta_flip_distance` | Distance from spot to the price at which the dominant crossover would change sign, in `atm · sqrt(21/252)` units | Leading (energy, known level) | A known release level: when it is one implied sd away, a day's move can force a systematic unwind. The closest thing TA gives to the barrier book. |
| `bb_squeeze` | Bollinger bandwidth (20d sd / 20d mean) as a trailing percentile, inverted so high = compressed | Leading (energy) | The squeeze is stored energy in TA language: a tight range that has lasted. Pairs naturally with `complacency` (surface quiet) — a squeeze with cheap vol is the product the hypothesis describes. |
| `atr_vs_implied` | 14d average true range, annualised, divided by `atm_1m` | Tag component and leading | Range-based realised vs implied. Captures intraday travel that close-to-close `rv` misses. Needs high/low; FX is 24h so the London-cut range is well defined. |
| `overextension` | Spot minus 50d mean in units of 50d sd (z), and RSI(14) as its bounded cousin | Leading (fuel) | Measures how far one-way the move has been: crowded on one side. Use the z; RSI is the same information bounded. |
| `momentum_divergence` | New N-day high (or low) in spot while 14d RSI or 10d return momentum fails to make a new extreme | Damping (exhaustion) | Divergence is the TA reading of a move losing participants: the energy is spent. A damping signal, not an energy one. |
| `round_number_proximity` | Distance to the nearest big figure (e.g. 1.1000, 150.00), in implied-sd units, folded | Leading (amplifier) | Option barriers, corporate orders and stops cluster at round numbers; proximity raises the chance a spark meets an amplifier. Pair-specific granularity in config. |
| `swing_level_proximity` | Distance to the nearest prior 63d swing high / low, in implied-sd units | Leading (amplifier) | The support / resistance version of the same mechanism. |
| `acceleration` | Second difference of the 10d smoothed log price, in sd units | Leading | Parabolic moves precede reversals (release), not continuations. Low prior; cheap to test. |
| `momentum_agreement` | Sign agreement of 1m, 3m and 12m returns (0..3) | Tag (`trend_aligned` / `mixed`) | All horizons agreeing = established trend with aligned positioning; disagreement = a regime in transition on the spot side. |

Two of these are duplicates of §3b under other names: `bb_squeeze` and `range_position` /
`efficiency_ratio` measure compression from different angles. Keep both in the screen; the
redundancy check (rank-corr 0.7) decides.

**Hurst exponent / fractal dimension** is the TA persistence measure; `variance_ratio` in §3b is
the statistically cleaner version of the same thing. Not proposed separately.

### 3f. Shock components as leading covariates

`shock.py` computes `cusum_pressure`, `bocd_p_short` and `score` and shows them on the card. They
are never tested as leading features. Register them in `LEADING` so they are screened like
everything else. If they pass they enter the pressure index; if not, the card line is just
description. The PDF's covariate vector included CUSUM for this reason.

## 4. Tier 2 — columns the adapter layer already anticipates

`configs/profile.yaml` names `vix`, `move`, `dxy`, `credit`, rates and six G10 spots. Adding the
corresponding **vols** and a rate differential unlocks the global and carry dimensions.

| Name | Needs | Slot | Why |
|---|---|---|---|
| `g10_vol_pct` | ATM 1m for the six other pairs | Tag (`global` / `local` stress) and leading | Crises are global, skew regimes local. A pair's stressed state that is also a G10 stressed state is a different animal from an idiosyncratic one. A tag, not a composite feature, so the pair's level remains its own. |
| `g10_vol_dispersion` | Same | Leading (energy) | Compressed dispersion across G10 vols = one factor pricing everything; a release will spread. |
| `cross_asset_vol_gap` | `vix`, `move` | Leading (energy) | FX vol percentile minus the max of VIX and MOVE percentiles. FX complacent while rates or equity vol is stressed is stored energy from outside the pair. |
| `carry_to_vol` | 2y rate differential | Tag or leading | Defines whether "carry" means anything for this pair; high carry-to-vol with low vol is the crowded carry state. |
| `spot_vs_rates_anchor` | 2y differential | Leading (energy), replacing or alongside `gap_z` | Spot residual against a rolling regression on the rate differential. The current `gap_z` anchors spot to its own mean, which says nothing about why it should go back. |
| `term_kink` | `atm_1w` | Leading (spark) and tag (`event_priced`) | 1W / 1M ATM ratio percentile. Short-dated richness is the surface telling you it sees an event; a rich kink that then passes without a move is the "priced" case the central constraint warns about. |
| `skew_term` | `rr25_3m` or `rr25_1y` | Profile; leading candidate | Front-end RR diverging from the back is a transition the surface has only half absorbed. |

## 4a. Destabilising indicators from other markets

FX vol events are rarely born in FX. They arrive through a small number of transmission channels,
and the useful signal is almost never a level in the other market: it is **speed** (a move large
relative to what that market's own implied priced), **structure breaking** (a term structure
inverting, a basis widening, a correlation flipping sign) or **forced flow** (systematic
deleveraging). Levels are priced; breaks and speed are where the FX surface has not yet caught up.

Three rules for this family:

- **Express as surprise, not level.** Each market move is standardised by *its own* implied vol
  where one exists (equity by VIX, rates by MOVE, oil by OVX, gold by GVZ). That gives a
  cross-market economic-surprise vector on the same footing as `shock.surprise`.
- **Test as leading, score on realised-minus-implied.** A cross-market shock the FX surface has
  already repriced is the "priced" case. `shock.lead_profile` separates a coincident indicator
  from a leading one; both are reported.
- **Mind the clock.** US equity, credit and Treasury closes land after a London FX cut. For a
  London-marked surface they are next-day information and must be lagged one day. Asia closes
  (JGBs, CNH fix, Nikkei) are same-day. Point-in-time means the cut, not the calendar date.

### Channels and candidates

| Channel | Name | Construction | Slot | Pairs most exposed |
|---|---|---|---|---|
| **Dollar funding** | `xccy_basis` | 3m cross-currency basis for the pair, level and 5d change in sd units | Leading (energy / amplifier) | EURUSD, USDJPY, all vs USD at quarter- and year-end |
| | `fra_ois` | FRA–OIS or SOFR–OIS spread, 5d change | Leading | All USD pairs |
| **Rates vol** | `move_surprise` | 5d change in 10y yield / MOVE-implied 5d sd | Leading (spark) | USDJPY, EURUSD, GBPUSD |
| | `move_vs_fxvol` | MOVE percentile minus `vol_pct` | Leading (energy) | All; the clearest cross-market gap |
| | `curve_speed` | 10d change in 2s10s slope, in sd units | Leading | USD pairs; recession / policy-pivot repricing |
| | `front_end_repricing` | 5d change in 2y yield / its trailing sd (policy-path shock) | Leading (spark) | All; the PDF's "policy-path repricing", here as a surprise |
| **Equity risk appetite** | `vix_term_inversion` | VIX / VIX3M ratio; > 1 is inverted | Tag (`eq_stressed`) and leading | AUD, NZD, CAD, NOK, MXN side; JPY, CHF haven side |
| | `vix_surprise` | 5d equity index return / VIX-implied 5d sd | Leading (spark) | Same |
| | `vvix_pct` | Vol-of-vol in equities, percentile | Leading (amplifier) | Same |
| | `eq_drawdown_speed` | Drawdown from 63d high divided by days since the high | Leading | Same |
| **Credit** | `hy_spread_change` | 10d change in HY CDS index in sd units; HY/IG ratio | Leading | Risk pairs; CHF via European credit |
| | `bank_cds` | Change in a bank CDS basket | Leading | EUR, CHF, GBP |
| **Commodities / terms of trade** | `oil_surprise` | 5d oil return / OVX-implied sd | Leading (spark) | CAD, NOK, RUB; inverse for JPY, INR |
| | `gold_surprise` | 5d gold return / GVZ-implied sd | Leading | CHF, AUD; a haven-flow signal when it moves with JPY |
| | `metals_momentum` | 21d copper / iron-ore return in sd units | Leading | AUD, CLP, ZAR |
| **China / EM** | `cnh_fix_deviation` | USDCNY fix minus model (prior close + basket move), in pips sd | Leading (spark) | AUD, NZD, KRW, TWD, and USD broadly |
| | `cnh_cny_spread` | Offshore minus onshore, percentile | Leading (energy) | Same |
| | `em_fx_vol_pct` | EM FX vol index percentile minus own `vol_pct` | Leading (energy) | High-beta G10 |
| **Japan-specific** | `jgb_cap_distance` | 10y JGB yield distance to any announced cap or band, in bp | Leading (energy, policy suppression) | USDJPY; the EURCHF-2015 / YCC-2022 mechanism |
| **Correlation structure** | `bond_equity_corr_flip` | Sign change in 63d stock–bond return correlation | Tag and leading | All; a flip is the hallmark of a macro regime change |
| | `cross_asset_corr_pct` | Mean absolute pairwise correlation across equities, rates, credit, oil, gold, DXY over 21d, percentile | Leading (amplifier) | All; everything moving together is the one-factor state a shock spreads through |
| **Systematic deleveraging** | `vol_target_leverage` | Inverse of trailing 21d cross-asset realised vol, percentile | Leading (fuel) | All; high implied leverage is the stock that gets cut |
| | `risk_parity_drawdown` | 21d return of a 60/40-with-leverage proxy, in sd units | Leading (fuel) | All |
| **Calendar / turn** | `quarter_end_proximity` | Days to quarter / fiscal-year end, kernel like `event_proximity` | Leading (calendar) | USDJPY (Mar), USD funding generally (Dec) |

### Composite destabiliser index

Screen each one at a time. Those retained are averaged into a `cross_market_pressure` index that
sits beside the pair's own `pressure` in the tilt, with its own β. Keeping it separate answers the
desk question directly — is the push coming from inside this pair or from outside — and lets the
card say which.

### Pair exposure map

A pair does not need every channel. `configs/local.yaml` carries a per-pair list of channels and
tickers, e.g.:

```yaml
exposure:
  EURUSD: [dollar_funding, rates_vol, credit, correlation]
  USDJPY: [rates_vol, japan, equity, dollar_funding]
  AUDUSD: [china_em, commodities, equity, correlation]
```

Only the listed channels are screened for that pair. That keeps the candidate count — and the
false-discovery problem — per pair rather than global.

## 5. Tier 3 — external data; fuel and amplifier

These are where the PDF says the edge is and where the repo has nothing. Only the first is public.

| Name | Needs | Slot | Notes |
|---|---|---|---|
| `imm_position_z`, `imm_change_4w` | CFTC weekly, lagged to publication (Friday for Tuesday data) | Leading (fuel) | The one public positioning series. Point-in-time means a 3-day lag; `lag_check` already exists for this. |
| `event_premium` | 1W ATM and a typed event calendar | Leading (spark) | Implied over the event vs mean realised on past events of the same type. Distinguishes a priced spark from an unpriced one. |
| `implied_corr_gap` | A third leg's ATM (e.g. EURJPY for EURUSD × USDJPY) | Leading (energy) | Implied minus realised cross correlation. Needs three vols; cheap once the G10 vol panel exists. |
| Dealer gamma, barrier map, client flow, structured book | Desk | Leading | D20. Registered through the same contract; nothing special in code. |

## 6. Damping as a first-class quantity

The repo has energy (`stored_energy`, `gap_pct`, `complacency`) and amplifiers (`jump_cluster_pct`,
`coupling_shift_pct`, `coherence_pct`). It has no dampers, so every signal pushes one way. Propose a
`damping` index alongside `pressure`, built from the retained damping features:

- `variance_ratio` below one, `return_autocorr` negative (price-based feedback);
- VRP richness in the upper tercile after a stress episode (vol supply returning);
- term structure normalising from inversion (`term_slope_pct` falling from a high);
- `volofvol_pct` falling while `vol_pct` is still high (exhaustion).

Damping enters `transitions.tilt` as a second covariate with its coefficient constrained ≤ 0, so
the effective pressure is `β_E · energy − β_D · damping + γ · composite`. It is screened one at a
time like energy, but its transition test is scored on moves *down* the rank and its outcome test
on realised *below* implied. The energy × damping quadrant (unstable / pinned-until-break / choppy /
carry) becomes a label on the card, not a number.

## 7. Plumbing that has to come first

Three changes make the additions above safe. Without them, a longer registry is a longer list of
things to overfit.

1. **State feature selection** (`labels.select_features`). Implement the PDF §5.3 procedure:
   tercile spread in forward 5d P&L with episode-bootstrap standard errors, sign stability across
   two non-overlapping halves, redundancy check at rank-corr 0.7, the set frozen in `DECISIONS.md`
   before `calibrate_states` runs. Today the composite's members are a config list nobody tests.
2. **Fold-local standardisation in `incremental_value`.** The feature and the state score are
   standardised with full-sample mean and sd before the walk-forward regression. Move the
   standardisation inside each training fold. Small effect, but the docstring's claim that nothing
   is fitted on the future should be true.
3. **A screen-wide false-discovery control.** With six candidates, "at least one retained" is
   lenient; with thirty it is a guarantee. Two options: hold out the final two years and require any
   retained feature to retain again there; or rank candidates on the training folds and allow only
   the top three into the hold-out test. Record the choice as a gate threshold (D16).

4. **A cut-aware `asof`.** The current contract cuts by calendar date. Cross-market series
   (§4a) need a per-column availability lag so a US close is not visible to a London-marked
   surface on the same date. One `lag_days` entry per column in `configs/local.yaml`, applied in
   the adapter before the frame reaches `features.build`, keeps the feature code unchanged.

Two smaller items: register `drift_t`; and extend `synth.py` with a trend/breakout mechanism, a
damping mechanism (a mean-reverting sub-type with negative return autocorrelation) and one
external shock series that leads the state with a lag, each with a null world, so the Tier 1 spot
features and the cross-market screen can be shown to recover something before real data.

## 8. Data requirements summary

| Column(s) | Unlocks |
|---|---|
| None | Everything in §3 and §6 except `atr_vs_implied` |
| Daily high / low at the cut | `atr_vs_implied` |
| `atm_1w` | `term_kink`, `event_premium` |
| VIX, VIX3M, VVIX, equity index | Equity channel in §4a |
| MOVE, 2y and 10y yields | Rates-vol channel |
| 3m cross-currency basis, FRA–OIS | Dollar-funding channel |
| HY / IG CDS indices, bank CDS basket | Credit channel |
| Oil, OVX, gold, GVZ, copper | Commodity channel |
| USDCNY fix, CNH, EM FX vol index | China / EM channel |
| 10y JGB and announced cap | Japan channel |
| `atm_3m`, `rr25_3m` | `skew_term`, VRP by tenor |
| Six G10 ATM 1m vols | `g10_vol_pct`, `g10_vol_dispersion`, `implied_corr_gap` |
| `vix`, `move` (already in profile config) | `cross_asset_vol_gap` |
| 2y rate differential | `carry_to_vol`, `spot_vs_rates_anchor` |
| CFTC IMM | `imm_position_z`, `imm_change_4w` |
| Typed event calendar | `event_premium`, typed `event_proximity` |

## 9. Proposed order

1. Plumbing (§7) — one work unit. Nothing else is trustworthy without it.
2. Tier 1 spot behaviour and realised-vs-implied (§3a, §3b) with the synthetic extension — the
   features that speak to the RR and fly archetypes and to the breakout question.
3. Tier 1 surface additions, TA proxies and backtester-derived features (§3c–3f) — cheap, low
   risk. `cta_position`, `bb_squeeze` and `cta_flip_distance` first among the TA set: they are the
   ones with a mechanism (positioning, energy, known level) rather than a pattern.
4. Damping (§6) — needs §3b's `variance_ratio`, `return_autocorr` and §3e's
   `momentum_divergence` to exist.
5. Tier 2 as the adapter grows (§4). `g10_vol_pct` as a tag first; it is the single most useful
   addition that needs new data.
6. Cross-market destabilisers (§4a), one channel at a time per pair, starting with the two that
   are cheapest and most general: rates vol (`move_vs_fxvol`, `move_surprise`) and equity
   (`vix_term_inversion`, `vix_surprise`). Dollar funding next for USD pairs.
7. Tier 3 when the data access questions (D20) are answered.

## 10. What is deliberately not proposed

- Adding spot-behaviour or cross-asset features to the **level composite**. Level stays a surface
  fact; the finder's step fit on forward vol depends on that.
- A fourth or fifth state. The episode budget does not support it; tags and sub-states exist for
  this.
- Hawkes branching ratio and the closure-weight kernel. Parked for the reasons in `leading.py`;
  `variance_ratio` and `return_autocorr` are the cheap stand-ins for feedback.
- Any feature whose point-in-time reconstruction is doubtful (sentiment indices without archived
  snapshots, estimated CTA positioning).
