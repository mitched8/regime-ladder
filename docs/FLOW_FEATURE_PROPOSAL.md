# Flow feature proposal — what the bank's own options and spot flow can add

Status: proposal for owner review. Companion to `FEATURE_PROPOSAL.md`, which covers features from
market data. This document covers features from two proprietary sources: the FX options franchise
(every price made, every trade done, and the resulting position with clients and client segments)
and the FX spot franchise (what the bank sees of client spot flow). Nothing here is implemented.

All flow is aggregated to five **counterparty segments** — hedge funds, real money, corporates,
banks as clients, interbank — and nothing finer. The segment is the unit of every feature; no
feature sees an individual counterparty. That is the whole data-handling design, and it is also
what makes the segments informative: each one trades for a different reason, on a different clock.

| Segment | Trades for | Clock | What its flow tends to mean |
|---|---|---|---|
| Hedge funds | Return; often informed, often leveraged | Days to weeks | Spark; fast fuel that gets cut quickly |
| Real money | Asset allocation and hedging of portfolios | Weeks to months | Slow fuel; persistent programmes; month-end rebalancing |
| Corporates | Hedging commercial exposure | Months; seasonal | Slowest fuel; level-driven; often fades moves (damping) |
| Banks as clients | Recycling their own client flow; regional franchises | Days | A second-hand read on flow the bank does not see directly |
| Interbank | Dealers hedging inventory | Hours to days | Street positioning: dealers buying vol from us are short |

## 1. What this data is, and what it is not

The anatomy in `leading.py` says a spot move becomes a vol event only when it hits **fuel**
(positioning that gets hurt) and an **amplifier** (dealer short gamma, thin liquidity), and that
**premium supply and demand** sets what the surface does afterwards. Market data can only infer
those things from their shadows — a quiet surface, a stretched spot. Flow data observes them. The
PDF marked flow, dealer gamma and the barrier book as the places where edge is most likely; this is
the plan for turning them into features that pass the same tests as everything else.

Three properties of the data shape every feature below.

**It is a sample of the street, not the street.** The bank sees its own share, and that share
differs by pair, segment and product. A feature built from raw notional confounds the street's
behaviour with the bank's market share. Every flow quantity is therefore (a) normalised by its own
trailing distribution — the same percentile contract as every other feature — and (b) where a
street-level quantity is wanted (dealer gamma, strike map) scaled by an estimated share per
segment and pair, with the estimate recorded in `configs/local.yaml` and revisited quarterly.

**Prices made are richer than trades done.** A request for a price is a demand signal whether or
not it trades. A quote that lost tells you a competitor was more aggressive on that side — which
is information about where the street is axed. Trade-only data has survivorship bias toward what
the bank was competitive on. Features are built from both, and the ratio between them is itself a
feature (§3.5).

**The client position is the stock; the flow is the change.** Positioning that gets hurt is a
stock quantity. The flow tells you whether it is building (new risk) or being cut (unwinds), and
the two have opposite meanings for the transition model. Every flow feature below is tagged
opening or closing against the running client position, and the two are kept as separate series.

## 2. Rules

- **Point-in-time at the cut.** A trade enters the frame on the business date of the surface mark
  it precedes. Trades after the London cut are next-day information for a London-marked surface.
  Warehouse latency (T+1 loads) is handled by `lag_days` per source, and `lag_check` in the leading
  screen reports value with an extra day of delay.
- **Segments only.** Features are computed per segment and in aggregate across the four client
  segments. Interbank is kept separate from the client aggregate throughout: it is the street's
  hedging, not end-user demand, and mixing the two cancels the signal in both.
- **Vega-weighted for options, notional-weighted for spot.** Option flow is aggregated in vega
  (and gamma, vanna, volga where the risk system supplies them) at the time of trade, not in
  notional or premium. Spot flow is in base-currency notional.
- **Opening vs closing.** Determined against the running position per client and strike /
  expiry. Where the position system cannot resolve it, the trade is "unknown" and excluded from
  the opening / closing splits but kept in gross.
- **Same slots, same tests.** Flow features go into the leading, damping, tag or profile slots
  defined in `FEATURE_PROPOSAL.md` §2 and pass the same `incremental_value` screen over a baseline
  that already knows the named state and the composite. The level composite is closed to them.

## 3. FX options flow features

### 3.1 Premium supply and demand (mechanism E)

| Name | Construction | Slot | What it measures |
|---|---|---|---|
| `vega_demand_imbalance` | Client net vega bought minus sold over 5d and 21d, divided by gross vega traded; percentile | Leading | Whether the client base is a net buyer or seller of vol. Net buying ahead of a move is hedging demand; net selling is supply that caps implied. |
| `rfq_vega_imbalance` | Same, computed on vega *requested* (all prices made), not traded | Leading | Demand before dealer selectivity filters it. Divergence from the traded version shows what the street declined to sell. |
| `wing_demand_share` | Vega requested in strikes beyond 25d as a share of all vega requested; percentile | Leading | Tail-hedging demand building before the fly reprices. The fly archetype's own flow. |
| `skew_demand_direction` | Signed vega in puts minus calls (one-sided RR demand), 21d, per segment | Leading and tag (`skew_bid_puts` / `balanced` / `skew_bid_calls`) | Who is paying for which side of the smile. The RR archetype's flow. One-sided real-money demand is a slow build; one-sided leveraged demand is a spark. |
| `tenor_demand_shift` | Share of vega requested in ≤ 1M tenors vs > 3M, 21d, change vs 126d | Leading | Demand migrating to the front end is the client base seeing an event the surface may not fully price. |
| `yield_enhancement_supply` | Vega sold via structured products (dual currency deposits, reverse convertibles) — mostly arriving through the banks-as-clients segment — 21d; percentile | Damping | The classic vol supply that rises with vol and caps it. Its withdrawal is the premium mechanism turning. |
| `vol_seller_pain` | Mark-to-market P&L of the net-short-vega segments' open option positions, as a fraction of their gross vega; 21d change | Leading (fuel) | Vol sellers under water withdraw supply and sometimes buy back. The fuel for a vol squeeze, measured directly instead of via the backtester proxy in `FEATURE_PROPOSAL.md` §3d. |
| `interbank_vega_imbalance` | Net vega the interbank segment buys from us minus sells to us, 5d and 21d, divided by gross; percentile | Leading | Dealers buying vol are covering short gamma or vega; dealers selling are long and recycling. The most direct read on street positioning available, and the counterpart to `hit_ratio_asymmetry`. |
| `interbank_share_of_gross` | Interbank vega as a share of all vega traded, 21d; percentile | Leading (amplifier) | When the street is passing risk among itself rather than to end users, inventory is stressed and liquidity for the next shock is thinner. |

### 3.2 Positioning that gets hurt (mechanism B)

| Name | Construction | Slot | What it measures |
|---|---|---|---|
| `client_vega_position` | Net client vega held with the bank by segment, scaled by segment share; percentile and z vs 252d | Leading (fuel) | The stock of client vol exposure. Extreme short vega across segments is the position that gets hurt on a vol event. |
| `client_position_pnl` | Unrealised P&L on client option positions by segment, as a fraction of premium paid; 5d change | Leading (fuel) | Positions moving into loss are the ones that get cut. The pain threshold the PDF's "accumulated positions" source describes. |
| `unwind_share` | Closing trades as a share of gross vega traded, 5d; percentile | Leading | A rise in unwinds is positioning being cut — fuel being consumed. Opposite sign to `vega_demand_imbalance` when both are high. |
| `position_concentration` | Share of the total client vega position held by the single largest segment | Leading (fuel) | Positioning held by one type of counterparty is fragile: they tend to act on the same trigger at the same time. Positioning spread across segments with different clocks is not. |
| `delta_position_client` | Net client delta-equivalent from options by segment, scaled | Leading | Directional exposure embedded in options. Adds to the spot positioning in §4. |

### 3.3 Dealer gamma and the amplifier (mechanism C)

The bank's own book is the measured part of the street's book. Scaled by share, it is the best
available estimate of street gamma. Where DTCC SDR prints are available they extend the strike
map; where they are not, the bank's book is the map.

| Name | Construction | Slot | What it measures |
|---|---|---|---|
| `dealer_gamma_near_spot` | Bank's net gamma within ± 1 implied daily sd of spot, sign and size, scaled to street; percentile of size, sign as a tag | Leading (amplifier) and tag (`street_short_gamma` / `street_long_gamma`) | Short gamma near spot means dealers hedge *with* the move; long gamma means they hedge against it. This is the amplifier / damper switch stated outright. |
| `gamma_flip_level` | Spot level at which net dealer gamma changes sign, distance from spot in implied-sd units | Leading (known level) | Through this level the market's hedging flow reverses direction. A known release level, like a barrier. |
| `gamma_profile_asymmetry` | Net gamma above spot minus below, within ± 2 sd | Leading | Which direction the amplification is in. A short-gamma book above spot amplifies rallies, not sell-offs. |
| `vanna_position` | Street vanna estimate (bank's book, scaled) | Leading | How the street's delta hedge changes when vol moves: the spot–vol feedback loop's gain. |
| `volga_position` | Street volga estimate | Leading | How the street's vega hedge changes when vol moves: the vol-of-vol feedback gain. |
| `dealer_gamma_change_5d` | 5d change in `dealer_gamma_near_spot` | Leading | Gamma being sold to the street (clients buying options) or bought. The flow version of the amplifier. |

### 3.4 Known levels and release dates (stored energy)

| Name | Construction | Slot | What it measures |
|---|---|---|---|
| `strike_cluster_proximity` | Distance from spot to the nearest strike with notional above the 90th percentile of the book, in implied-sd units | Leading (amplifier) | Large strikes pin spot into expiry and release it after. |
| `expiry_roll_off` | Vega and gamma expiring in the next 1, 5 and 10 days as a share of the book | Leading (calendar) | A known date on which the gamma profile changes. Enters with the `calendar` future path in `transitions.covariate_path`. |
| `barrier_proximity` | Nearest knock-out / knock-in cluster above and below spot, distance in implied-sd units, signed by what the trigger does to dealer gamma | Leading (amplifier or damper) | A KO that removes dealer long gamma flips a damper into an amplifier at a known level. The PDF's structural-suppression release mechanism. |
| `barrier_notional_in_range` | Barrier notional within ± 1.5 implied sd of spot, scaled | Leading | How much of the structured book is live at current levels. |
| `structured_vega_at_risk` | Vega in structured products (TARFs, PRDCs, accumulators) that knocks out within ± 2 sd | Leading (amplifier) | Forced vega and gamma at known levels. Pair-specific: CNH and KRW TARFs, USDJPY PRDCs. |

### 3.5 Information in the pricing process

These use the price-made data, which no market-data feature can replicate.

| Name | Construction | Slot | What it measures |
|---|---|---|---|
| `hit_ratio_asymmetry` | Hit ratio when the client is buying vol minus hit ratio when selling, 21d | Leading | If the bank wins more when it sells vol, competitors are more aggressive buyers — the street is short vol and bidding to cover. An inference about street positioning from losing quotes. |
| `rfq_intensity` | Number of prices requested per day, vega-weighted, as a percentile | Leading (spark) | Client attention. Rises before scheduled events (priced) and before unscheduled ones (not priced); `event_proximity` separates the two. |
| `rfq_spread_dispersion` | Where multi-dealer cover prices are visible: dispersion of dealer quotes on the same request, in vol | Leading (amplifier — liquidity) | Dealers disagreeing on price is thin liquidity in the surface. |
| `quote_width` | The bank's own bid–offer in vol by tenor and delta bucket, percentile | Leading (amplifier) and profile | The desk's own risk appetite is a liquidity measure; it widens when the desk is uncertain. |
| `flow_toxicity` | Post-trade mark-out: average change in mid vol 1d and 5d after client trades, signed by client direction, by segment | Leading (spark) | Segments whose trades are followed by moves in their favour are informed. Their current direction is a signal; the others' is noise. |
| `informed_segment_direction` | Net vega direction of the segment(s) with the highest trailing `flow_toxicity` | Leading (spark) | The actionable version of the above. |

## 4. FX spot flow features

### 4.1 Positioning and its change (fuel)

| Name | Construction | Slot | What it measures |
|---|---|---|---|
| `spot_flow_net` | Client net base-currency bought minus sold, 5d and 21d, divided by gross; by segment and aggregate; percentile | Leading (fuel) | Directional positioning building. Combined with `drift_t`: a one-way market *with* one-way client flow is crowded; without it, the move is being absorbed. |
| `spot_flow_persistence` | Autocorrelation of daily net flow over 63d | Leading | Persistent flow is a programme (hedging, rebalancing, trend-following); it continues and it ends. Non-persistent flow is noise. |
| `spot_position_proxy` | Cumulative net client flow over 63d by segment, scaled by share | Leading (fuel) | The stock of spot positioning, where the position system does not track it directly. |
| `segment_disagreement` | Dispersion of net-flow sign across segments, 5d | Tag (`segments_aligned` / `two_way`) | Everyone on one side is a crowded trade; two-way flow is a market that clears. |
| `hedge_fund_vs_real_money` | Hedge-fund net flow minus real-money net flow, 21d | Leading | The fast-money vs slow-money split. Fast money leading slow money is the start of a move; slow money following is its middle; fast money fading slow money is its end. |
| `corporate_hedge_ratio_change` | Corporate segment flow relative to its seasonal norm (same month, prior years) | Leading | Corporates changing hedge ratios is a slow, large, persistent flow — fuel that builds over months. |
| `corporate_fade` | Sign disagreement between corporate net flow and 5d spot return, rolling share | Damping | Corporates sell strength and buy weakness by construction of their hedging programmes. When they stop fading, a damper has gone. |
| `interbank_spot_imbalance` | Interbank net flow divided by gross, 5d | Leading (amplifier) | Dealers all hedging the same way is the street's inventory moving together — the spot-side read on `interbank_vega_imbalance`. |

### 4.2 Absorption and damping

The spot flow data's most distinctive contribution: who is on the other side.

| Name | Construction | Slot | What it measures |
|---|---|---|---|
| `flow_price_elasticity` | Rolling 21d regression of daily spot return on client net flow; coefficient as a percentile | Leading (amplifier) when high; damping when low | Price impact per unit of flow. High impact is thin liquidity: the amplifier. Low impact is deep liquidity: the damper. |
| `flow_price_divergence` | Sign disagreement between 5d client net flow and 5d spot return, as a rolling share | Damping | Clients selling into a rally means someone else is buying — the move is being absorbed, and the client base is fading it. The direct observation of level-based flows. |
| `absorption_ratio` | Gross client flow / realised range over 5d, percentile | Damping | Large flow for little movement is a market with depth. |
| `fixing_flow_size` | Net flow at the WMR 4pm fix and other benchmark fixes, as a share of daily gross, and its month-end seasonal | Leading (calendar) | Rebalancing flow is predictable in timing and often in direction; it is damping during the month and a spark at month-end. |

### 4.3 Orders and triggers

Where the bank runs a resting-order book for clients.

| Name | Construction | Slot | What it measures |
|---|---|---|---|
| `stop_density_near_spot` | Stop-loss order notional within ± 1 implied daily sd of spot, above and below, scaled | Leading (amplifier) | Stops are the fuel-meets-amplifier mechanism in its purest form: a move through the level produces flow in the same direction. |
| `limit_density_near_spot` | Take-profit and limit order notional in the same band | Damping | Limits are the opposite: flow against the move. |
| `order_imbalance` | Stops minus limits, signed by side | Leading | Net amplification or damping at current levels. |
| `barrier_trigger_flow` | Expected spot hedging flow if the nearest barrier cluster triggers, from the options book | Leading (amplifier) | The spot-side consequence of §3.4. |

### 4.4 Information

| Name | Construction | Slot | What it measures |
|---|---|---|---|
| `spot_flow_toxicity` | Post-trade mark-out of spot trades by segment, 1h and 1d | Leading | Which segments are informed in spot, as in options. Expect hedge funds to lead; the interest is in when the ranking changes. |
| `informed_spot_direction` | Net flow direction of the most-informed segment | Leading (spark) | The actionable version. |

## 5. Cross features: options × spot

The combination is where the anatomy's multiplicative structure lives.

| Name | Construction | Slot | What it measures |
|---|---|---|---|
| `hedged_vs_naked_positioning` | Sign agreement between client spot position and client option delta position, by segment | Tag (`conviction` / `hedged`) | Long spot and long calls is conviction. Long spot and long puts is a hedged position that will be *held* through a move — damping. Long spot and short puts is a position that gets hurt twice. |
| `fuel_x_amplifier` | `spot_flow_net` (crowding) × `dealer_gamma_near_spot` (short-gamma sign) — the anatomy's product term | Leading | The PDF's interaction term, measured rather than assumed. A crowded spot position in a short-gamma street is the unstable quadrant. |
| `fuel_x_level` | `spot_position_proxy` × proximity of the nearest stop / barrier / gamma-flip level | Leading | Crowded positioning with a known release level nearby. |
| `premium_vs_positioning` | `vega_demand_imbalance` vs `spot_flow_net`: clients buying vol while cutting spot, or selling vol while adding spot | Leading | The second is the complacent configuration the stored-energy hypothesis describes, seen in flow rather than inferred from the surface. |
| `energy_feedback_quadrant` | Energy from `FEATURE_PROPOSAL.md` (gap, squeeze, cross-market gaps) × feedback from this document (`order_imbalance`, `dealer_gamma_near_spot`, `flow_price_elasticity`) | Card label | The four-quadrant map (unstable / pinned until it breaks / choppy / carry), now with feedback measured from the book rather than from price behaviour. |

## 6. Tags and profile

Tags from this document, each three-valued or binary and episode-gated like the existing ones:

- `street_gamma` — short / neutral / long near spot.
- `skew_demand` — puts bid / balanced / calls bid.
- `segments_aligned` — aligned / two-way.
- `conviction` — naked / hedged.
- `street_covering` — interbank net buying vol / neutral / net selling.

Profile additions (description only, no test): vega position by segment, dealer gamma profile
summary, top strikes and barriers, order-book summary, net flow by segment over 21d. The card's
state-profile block gains a "book" paragraph, with the segment table from §1 as its vocabulary.

## 7. Testing, and what to expect

Everything goes through the leading screen one at a time, over the baseline of named state +
composite, with the two tests (forward-outcome R² gain; k-step transition gain), the lag check and
the false-discovery control proposed in `FEATURE_PROPOSAL.md` §7. Three additions specific to flow:

- **Share-scaling sensitivity.** Each street-scaled feature is screened with the share estimate
  halved and doubled. A feature that retains only under one scaling is not retained.
- **Segment stability.** A feature retained in aggregate is checked by segment; if the aggregate
  effect comes entirely from one segment, the segment version replaces it. Five segments means a
  feature with a segment split has five chances to retain; that is counted in the false-discovery
  control, not ignored.
- **Null test by time permutation.** Flow cannot be synthesised credibly. Instead each retained
  feature is re-screened with its series circularly shifted by 63, 126 and 252 days; retention
  under a shift is a false positive.

Expectations, stated before the data is seen: the amplifier features (§3.3, §4.3) and the
information features (§3.5, §4.4) are the most likely to retain, because they measure mechanisms
the surface cannot see. The demand features (§3.1) are the most likely to be priced — the surface
moves as the flow arrives — and should be judged mainly on the transition test. Positioning
features (§3.2, §4.1) are slow and will show up at k = 5–10 rather than k = 1. Most of the list
will fail; the design assumes that.

## 8. Data requirements

| Source | Fields needed | Enables |
|---|---|---|
| FXO RFQ log | Timestamp, pair, tenor, strike / delta, structure, side, vega requested, our price, traded flag, cover price where visible, counterparty segment | §3.1, §3.5 |
| FXO trade blotter | As above plus traded vega, gamma, vanna, volga at trade; opening / closing flag | §3.1–3.3 |
| Position by segment | Daily net Greeks per segment, by strike and expiry; unrealised P&L per segment | §3.2, §3.4 |
| Bank option book | Daily gamma, vanna, volga profile by strike; barrier inventory with trigger levels and notional; expiry calendar | §3.3, §3.4 |
| Spot trade blotter | Timestamp, pair, side, notional, counterparty segment, venue, fix flag | §4.1, §4.2, §4.4 |
| Client order book | Resting stop / limit orders by level and notional | §4.3 |
| Share estimates | Per pair and segment, quarterly | Scaling throughout |
| Post-trade marks | Mid at +1h, +1d, +5d for mark-outs | Toxicity |

## 9. Order

1. The segment mapping (which counterparty codes fall into which of the five) and the share
   estimates per pair and segment, in `configs/local.yaml`.
2. The bank's own book first — no counterparty data needed: `dealer_gamma_near_spot`,
   `gamma_flip_level`, `expiry_roll_off`, `strike_cluster_proximity`, `barrier_proximity`,
   `quote_width`. These are the amplifier and known-level features and the ones most likely to
   retain.
3. Interbank next, because it is one segment with one meaning: `interbank_vega_imbalance`,
   `interbank_share_of_gross`, `interbank_spot_imbalance`, and `hit_ratio_asymmetry` from the RFQ
   log, which measures the same thing from the other side.
4. Options demand and positioning across the four client segments: `vega_demand_imbalance`,
   `wing_demand_share`, `skew_demand_direction`, `client_vega_position`, `unwind_share`,
   `vol_seller_pain`.
5. The rest of the RFQ log: `rfq_vega_imbalance`, `rfq_intensity`, `flow_toxicity`.
6. Spot flow by segment: `spot_flow_net`, `hedge_fund_vs_real_money`, `corporate_fade`,
   `flow_price_elasticity`, `flow_price_divergence`; then orders if the book is available.
7. Cross features (§5) once both sides exist.

## 10. Not proposed

- Anything finer than the five segments. The segment is the unit because each one has a reason to
  trade; a finer cut adds noise, not mechanism.
- The desk's own trading P&L or inventory as a feature. The book's *Greeks* are a measurement of the
  street; its P&L is not, and using it would be circular with the ladder's target.
- Using flow to relabel the state. Flow is fuel, amplifier and premium; the state is the surface.
