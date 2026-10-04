# Review packet · stage `leading` · calibration A_energy (synthetic)

| key | value |
|---|---|
| pair | EURUSD |
| data | 2015-01-01 to 2024-08-28 (2520 days) |
| finder / leading target | expected forward 5d straddle P&L given the true state path (synthetic, noiseless); diagnostic outcome at 21d |
| git commit | 556ee10 |
| config hash | ba38712377 |
| generated | 2026-10-04 17:40 |

## Definitions
- Units: P&L is cash per unit of standard notional (per unit of vega at inception once D1 is closed). Horizons h are trading days from entry, capped at the option tenor.
- Point-in-time: every feature, label and statistic stamped on date t uses only data available at the close of t. Labels are assigned from trailing data; forward quantities appear only as targets.
- Costs are excluded everywhere by design.
- States: carry, rising, agitated, stressed, normalising, settling (+ extreme if it has >= 5 episodes, else merged into stressed). Stress rank low to high (used for 'moves up' and by the transition tilt): carry < settling < rising < agitated < normalising < stressed < extreme.
- Leading features (0..100; all trailing percentiles except stored_energy, which is a product of two percentiles divided by 100, so its typical level is ~20-25, not 50):
  - gap_pct: how unusual today's distance of spot from its 120-day anchor is, in implied-vol units.
  - complacency: realised under implied, low vol-of-vol, cheap front end, combined.
  - stored_energy = gap_pct x complacency / 100 (a stretched price held by a complacent market; the hypothesis is the product).
  - jump_cluster_pct: clustering of large returns. coupling_shift_pct: change in short vs long spot-vol coupling. coherence_pct: cross-pair co-movement.
  - gap_z_signed: the signed stretch itself (a z, not a percentile); drift_t: t-statistic of the 60-day drift (signed). Signed features are judged on the RR outcome (`target` column), unsigned ones on the straddle.
  - Any other name is a desk series supplied through the adapter; it is claimed trailing-only.
- Screen, one feature at a time, against a baseline that already knows the named state AND the continuous composite, 4 walk-forward folds after a 40% initial training block (`n_pairs` > 1 means the test is pooled across pairs with pair dummies, folds cut at common dates, one tilt coefficient across pairs):
  - (1) outcome test: `delta_r2` = pooled out-of-sample R² gain on the forward outcome named in `target`; `folds_r2_up` = folds in which the gain is > 0; `n` = days with feature, label, composite and outcome; `r2_base` = the baseline's own OOS R² on that feature's sample (samples differ by feature because warm-ups differ). `delta_r2_low` is the gain when the feature is allowed to act only in the calm states (carry, settling) as well as overall: the conditional form of the stored-energy hypothesis. `delta_r2_lagged` repeats the test with the feature delayed one day; differences under 0.005 are noise.
  - (2) transition test: `transition_gain` = out-of-sample k-step transition log-likelihood gain per transition (nats) through a tilt, with the composite already in the tilt, both z-scored per fold on the training part; the declared k is 5. `transition_gain_k10` and `_k21` are the same at longer horizons, diagnostics only: a slow build-up that leads by a week can show at 21 before it shows at 5. `folds_transition_up` = folds with gain > 0 at the declared k; `n_transitions` = days with feature and label (no outcome needed, so it can exceed `n`). `beta_mean` = tilt coefficient per standard deviation of the feature, averaged over folds.
  - `fail_reason` names the first rule a non-retained feature failed, on unrounded values (the table rounds). `target_note` says when a feature's configured outcome was unavailable and the default was used.
  - `p_perm`: share of 100 circular shifts of the feature (>= 63 days) whose outcome gain matches or beats the observed one; the selection control for a screen over many candidates. A real feature has p_perm near 0.01 (the floor); a noise feature anywhere.
  - Retention, two stages. Walk-forward on the first 80% of the history: delta_r2 >= 0.005 AND folds_r2_up >= 3 AND transition_gain >= 0.002 AND folds_transition_up >= 3 AND p_perm <= 0.05. Then only the top 3 candidates by delta_r2 that pass are tested once on the final 20% (`holdout_tested` = True; trained on everything before it) and keep the flag unless the hold-out CONTRADICTS them: `delta_r2_holdout` must be > -0.005 and `transition_gain_holdout` > -0.002. The hold-out holds three or four episodes, so it is asked not to reverse the finding, not to re-prove it; read a hold-out value near zero as neutral, a clearly negative one as the warning it is. A candidate that passed the walk-forward but was not in the top 3 shows `holdout_tested` = False and is not retained.
- Lift (event study): on days when the feature is at or above its trailing 90th percentile, the probability that the state enters the high band within 20 days, divided by the same probability on all eligible days (days not already in the high band). `n_signal_runs` counts separate signal episodes — the effective sample. 90% block-bootstrap interval. Not a gate; it is the more powerful test for a sparse feature and the number a trader can read directly.
- Shock score (0..100) = mean of CUSUM pressure on squared surprises (an alarm is a day on which the pressure reaches 0.999, i.e. the statistic crosses its calibrated threshold), BOCD probability of a short run, and recent |surprise|; surprise = return / implied daily vol marked the day before.
- Gate 4: at least one feature retained AND the pressure index of the retained features passes test (2) on its own: gain >= 0.002 per transition, > 0 in >= 3/4 folds.
- Pressure index = mean of the RETAINED features after each is z-scored against its own trailing 3-year window (0 = normal for that feature; percentiles, products and signed series on one footing). If nothing is retained, no pressure index exists and only the constant fan is meaningful.
- High band = stressed and extreme. Transition tilt: P_ij(psi) proportional to P0_ij x exp(beta x psi x (rank_j - rank_i)), with the composite as a second covariate (gamma); likelihood scored k = 5 days ahead. The fan horizon (21 days) is the transition forecast's horizon, not a ladder horizon.

## 1. Gate 4
```
{
 "retained": [
  "desk_flow"
 ],
 "pressure_transition_gain": 0.004645719142973367,
 "pressure_folds_positive": 4,
 "pass": true
}
```
## 2. Screen (one feature at a time vs state + composite baseline)
Target substitutions: gap_z_signed: rr_25d unavailable, judged on straddle_atm; drift_t: rr_25d unavailable, judged on straddle_atm

`delta_r2_diag` is the outcome test on straddle_atm_h21 (the same archetype at the diagnostic horizon); it does not decide retention.

| feature | target | n | n_pairs | r2_base | delta_r2 | p_perm | delta_r2_low | delta_r2_diag | delta_r2_lagged | folds_r2_up | transition_gain | folds_transition_up | holdout_tested | delta_r2_holdout | transition_gain_holdout | retain | fail_reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| desk_flow | straddle_atm | 1895 | 1 | 0.2515 | 0.0219 | 0.0099 | 0.0090 | 0.0186 | 0.0183 | 4 | 0.0064 | 4 | True | -0.0007 | 0.0041 | True |  |
| coupling_shift_pct | straddle_atm | 1702 | 1 | 0.3355 | 0.0007 | 0.2079 | 0.0022 | -0.0162 | -0.0010 | 2 | -0.0002 | 3 | False |  |  | False | delta_r2;folds_r2_up;transition_gain;p_perm |
| jump_cluster_pct | straddle_atm | 1808 | 1 | 0.3201 | -0.0015 | 0.2277 | -0.0100 | -0.0128 | -0.0095 | 1 | 0.0023 | 3 | False |  |  | False | delta_r2;folds_r2_up;p_perm |
| coherence_pct | straddle_atm | 1807 | 1 | 0.3196 | -0.0038 | 0.4158 | -0.0116 | 0.0024 | -0.0065 | 2 | 0.0010 | 2 | False |  |  | False | delta_r2;folds_r2_up;transition_gain;folds_transition_up;p_perm |
| complacency | straddle_atm | 1807 | 1 | 0.3196 | -0.0087 | 0.4257 | 0.0061 | -0.0773 | -0.0157 | 1 | -0.0161 | 2 | False |  |  | False | delta_r2;folds_r2_up;transition_gain;folds_transition_up;p_perm |
| stored_energy | straddle_atm | 1768 | 1 | 0.3215 | -0.0097 | 0.4356 | -0.0420 | -0.0032 | -0.0035 | 1 | -0.0315 | 2 | False |  |  | False | delta_r2;folds_r2_up;transition_gain;folds_transition_up;p_perm |
| gap_pct | straddle_atm | 1768 | 1 | 0.3215 | -0.0135 | 0.5149 | -0.0111 | -0.0020 | 0.0004 | 2 | -0.0071 | 2 | False |  |  | False | delta_r2;folds_r2_up;transition_gain;folds_transition_up;p_perm |
| gap_z_signed | straddle_atm | 1895 | 1 | 0.2515 | -0.0298 | 0.8416 | -0.0313 | -0.0727 | 0.0022 | 1 | 0.0018 | 3 | False |  |  | False | delta_r2;folds_r2_up;transition_gain;p_perm |
| drift_t | straddle_atm | 1895 | 1 | 0.2515 | -0.0348 | 0.9109 | -0.0210 | -0.0609 | 0.0050 | 1 | 0.0007 | 3 | False |  |  | False | delta_r2;folds_r2_up;transition_gain;p_perm |

Transition test by horizon (k = 5 decides; 10 and 21 are diagnostics):
| feature | n_transitions | transition_gain | transition_gain_k10 | transition_gain_k21 | beta_mean |
|---|---|---|---|---|---|
| desk_flow | 1955 | 0.0064 | 0.0043 | -0.0000 | 0.1385 |
| coupling_shift_pct | 1701 | -0.0002 | -0.0012 | -0.0012 | 0.0500 |
| jump_cluster_pct | 1807 | 0.0023 | -0.0045 | -0.0199 | -0.0774 |
| coherence_pct | 1806 | 0.0010 | -0.0018 | -0.0187 | -0.1007 |
| complacency | 1806 | -0.0161 | -0.0359 | -0.0926 | 0.1097 |
| stored_energy | 1767 | -0.0315 | -0.0434 | -0.0520 | 0.0954 |
| gap_pct | 1767 | -0.0071 | -0.0115 | -0.0103 | 0.0099 |
| gap_z_signed | 1955 | 0.0018 | -0.0056 | -0.0139 | -0.0618 |
| drift_t | 1955 | 0.0007 | -0.0016 | -0.0037 | -0.0308 |

## 2b. Lift: P(high band within 20 days | feature in its top decile) vs base rate
| feature | p_event_signal | p_event_all | lift | lift_lo | lift_hi | n_signal | n_signal_runs | n_events_after_signal |
|---|---|---|---|---|---|---|---|---|
| stored_energy | 0.144 | 0.162 | 0.890 | 0.288 | 1.540 | 270 | 42 | 39 |
| gap_pct | 0.213 | 0.162 | 1.313 | 0.734 | 1.984 | 277 | 39 | 59 |
| complacency | 0.094 | 0.169 | 0.557 | 0.211 | 0.979 | 308 | 49 | 29 |
| jump_cluster_pct | 0.102 | 0.169 | 0.603 | 0.122 | 1.175 | 265 | 28 | 27 |
| coupling_shift_pct | 0.155 | 0.151 | 1.024 | 0.508 | 1.642 | 297 | 50 | 46 |
| coherence_pct | 0.045 | 0.169 | 0.265 | 0.000 | 0.542 | 201 | 38 | 9 |
| gap_z_signed | 0.232 | 0.157 | 1.472 | 0.701 | 2.371 | 233 | 16 | 54 |
| drift_t | 0.242 | 0.157 | 1.535 | 0.808 | 2.365 | 207 | 38 | 50 |
| desk_flow | 0.163 | 0.157 | 1.037 | 0.795 | 1.271 | 1311 | 24 | 214 |

## 3. Feature distributions
| index | count | mean | std | min | 5% | 50% | 95% | max | first_valid | last_valid | share_at_mode |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stored_energy | 2272.0 | 21.2 | 15.6 | 0.0 | 1.4 | 18.6 | 49.5 | 79.1 | 2015-12-15 | 2024-08-28 | 0.0 |
| gap_pct | 2272.0 | 44.3 | 29.8 | 0.0 | 3.0 | 41.0 | 96.0 | 100.0 | 2015-12-15 | 2024-08-28 | 0.0 |
| complacency | 2311.0 | 50.1 | 17.3 | 2.1 | 24.0 | 49.7 | 79.4 | 97.2 | 2015-10-21 | 2024-08-28 | 0.0 |
| jump_cluster_pct | 2312.0 | 30.9 | 30.2 | 0.0 | 0.0 | 25.0 | 90.9 | 100.0 | 2015-10-20 | 2024-08-28 | 0.3 |
| coupling_shift_pct | 2206.0 | 46.8 | 27.9 | 0.0 | 4.9 | 44.9 | 94.1 | 100.0 | 2016-03-16 | 2024-08-28 | 0.0 |
| coherence_pct | 2311.0 | 48.5 | 29.0 | 0.0 | 4.1 | 46.8 | 94.3 | 100.0 | 2015-10-21 | 2024-08-28 | 0.0 |
| gap_z_signed | 2460.0 | -0.0 | 1.0 | -2.8 | -1.4 | -0.1 | 1.6 | 3.3 | 2015-03-26 | 2024-08-28 | 0.0 |
| drift_t | 2460.0 | -0.1 | 1.1 | -3.2 | -1.7 | -0.1 | 1.8 | 2.9 | 2015-03-26 | 2024-08-28 | 0.0 |
| desk_flow | 2460.0 | 50.8 | 3.8 | 50.0 | 50.0 | 50.0 | 54.3 | 100.0 | 2015-03-26 | 2024-08-28 | 0.9 |

## 4. Mean feature value by state (0..100)
| index | agitated | carry | normalising | rising | settling | stressed |
|---|---|---|---|---|---|---|
| stored_energy | 21.0 | 25.2 | 15.7 | 20.5 | 22.7 | 14.9 |
| gap_pct | 43.7 | 39.0 | 65.7 | 46.6 | 55.8 | 40.8 |
| complacency | 48.6 | 67.5 | 28.4 | 45.6 | 41.0 | 38.4 |
| jump_cluster_pct | 30.1 | 33.6 | 33.5 | 30.8 | 33.8 | 26.4 |
| coupling_shift_pct | 53.0 | 44.4 | 21.0 | 44.0 | 43.1 | 38.3 |
| coherence_pct | 46.9 | 48.9 | 48.1 | 53.3 | 55.7 | 42.0 |
| gap_z_signed | -0.1 | -0.2 | 0.7 | 0.0 | 0.2 | 0.0 |
| drift_t | -0.0 | -0.3 | 0.6 | -0.0 | 0.1 | 0.0 |
| desk_flow | 50.1 | 51.6 | 50.0 | 50.1 | 53.0 | 50.0 |

## 5. Correlations (features and the composite)
| index | stored_energy | gap_pct | complacency | jump_cluster_pct | coupling_shift_pct | coherence_pct | gap_z_signed | drift_t | desk_flow | composite |
|---|---|---|---|---|---|---|---|---|---|---|
| stored_energy | 1.00 | 0.83 | 0.28 | -0.01 | 0.01 | 0.07 | -0.17 | -0.17 | 0.24 | -0.17 |
| gap_pct | 0.83 | 1.00 | -0.20 | 0.13 | 0.01 | 0.22 | 0.04 | -0.01 | 0.26 | 0.08 |
| complacency | 0.28 | -0.20 | 1.00 | -0.30 | -0.00 | -0.24 | -0.21 | -0.19 | -0.01 | -0.57 |
| jump_cluster_pct | -0.01 | 0.13 | -0.30 | 1.00 | 0.17 | 0.29 | 0.14 | 0.07 | 0.03 | -0.05 |
| coupling_shift_pct | 0.01 | 0.01 | -0.00 | 0.17 | 1.00 | 0.01 | 0.06 | 0.08 | 0.01 | -0.03 |
| coherence_pct | 0.07 | 0.22 | -0.24 | 0.29 | 0.01 | 1.00 | 0.06 | 0.01 | 0.11 | -0.06 |
| gap_z_signed | -0.17 | 0.04 | -0.21 | 0.14 | 0.06 | 0.06 | 1.00 | 0.89 | -0.06 | 0.15 |
| drift_t | -0.17 | -0.01 | -0.19 | 0.07 | 0.08 | 0.01 | 0.89 | 1.00 | -0.10 | 0.19 |
| desk_flow | 0.24 | 0.26 | -0.01 | 0.03 | 0.01 | 0.11 | -0.06 | -0.10 | 1.00 | -0.18 |
| composite | -0.17 | 0.08 | -0.57 | -0.05 | -0.03 | -0.06 | 0.15 | 0.19 | -0.18 | 1.00 |

## 6. Lead profile: mean value in the days before ANY move up the state ranking (e.g. carry to rising), before moves down, and on all days. Moves skip ranks (a rising-to-stressed move counts once), so up and down counts need not match
| feature | window_days | all_days | before_escalation | before_de_escalation | n_escalations | n_de_escalations |
|---|---|---|---|---|---|---|
| stored_energy | 5 | 21.2 | 24.1 | 22.7 | 114.0 | 135.0 |
| stored_energy | 20 | 21.2 | 23.6 | 22.8 | 114.0 | 135.0 |
| gap_pct | 5 | 44.3 | 51.2 | 50.8 | 114.0 | 135.0 |
| gap_pct | 20 | 44.3 | 50.4 | 51.7 | 114.0 | 135.0 |
| complacency | 5 | 50.1 | 47.8 | 45.0 | 114.0 | 135.0 |
| complacency | 20 | 50.1 | 49.3 | 45.3 | 114.0 | 135.0 |
| jump_cluster_pct | 5 | 30.9 | 30.9 | 32.8 | 114.0 | 135.0 |
| jump_cluster_pct | 20 | 30.9 | 30.2 | 30.3 | 114.0 | 135.0 |
| coupling_shift_pct | 5 | 46.8 | 43.3 | 42.6 | 114.0 | 135.0 |
| coupling_shift_pct | 20 | 46.8 | 42.6 | 41.5 | 114.0 | 135.0 |
| coherence_pct | 5 | 48.5 | 54.0 | 51.7 | 114.0 | 135.0 |
| coherence_pct | 20 | 48.5 | 51.7 | 50.1 | 114.0 | 135.0 |
| gap_z_signed | 5 | -0.0 | -0.1 | -0.0 | 114.0 | 135.0 |
| gap_z_signed | 20 | -0.0 | -0.1 | -0.1 | 114.0 | 135.0 |
| drift_t | 5 | -0.1 | -0.2 | -0.1 | 114.0 | 135.0 |
| drift_t | 20 | -0.1 | -0.2 | -0.1 | 114.0 | 135.0 |
| desk_flow | 5 | 50.8 | 52.0 | 51.5 | 114.0 | 135.0 |
| desk_flow | 20 | 50.8 | 51.3 | 51.1 | 114.0 | 135.0 |

## 7. Every move into the high band (19): values before it
| date | from | to | stored_energy_t-20 | stored_energy_t-5 | stored_energy_t-1 | gap_z_signed_t-5 | gap_z_signed_t-1 | drift_t_t-5 | drift_t_t-1 | desk_flow_t-5 | desk_flow_t-1 | pressure_t-1 | shock_max_20d | days_since_cusum_alarm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2015-12-15 | rising | stressed |  |  |  | -1.3 | -0.7 | -2.0 | -1.1 | 50.0 | 50.0 |  | 56.4 |  |
| 2016-01-27 | rising | stressed | 1.9 | 4.8 | 4.4 | -0.5 | -0.4 | -0.3 | -0.3 | 50.0 | 50.0 | -0.4 | 35.3 |  |
| 2016-03-30 | rising | stressed | 48.3 | 18.9 | 21.5 | -0.9 | -1.0 | -1.7 | -1.5 | 50.0 | 50.0 | -0.4 | 36.4 |  |
| 2017-03-28 | rising | stressed | 0.3 | 6.6 | 0.0 | 0.2 | -0.0 | 0.3 | 0.1 | 50.0 | 50.0 | -0.3 | 46.4 |  |
| 2017-11-30 | rising | stressed | 1.3 | 4.5 | 11.1 | 0.2 | 0.6 | -0.1 | 0.2 | 50.0 | 50.0 | -0.3 | 81.9 |  |
| 2018-02-16 | rising | stressed | 2.2 | 15.3 | 1.0 | -0.5 | 0.0 | 0.0 | 0.1 | 50.0 | 50.0 | -0.2 | 44.9 |  |
| 2018-05-17 | rising | stressed | 11.4 | 7.9 | 7.3 | 0.4 | 0.2 | 0.5 | 0.3 | 50.0 | 50.0 | -0.2 | 29.7 |  |
| 2018-10-09 | rising | stressed | 34.0 | 45.3 | 43.1 | -1.5 | -1.4 | -2.7 | -2.3 | 50.0 | 50.0 | -0.2 | 33.8 |  |
| 2019-09-20 | rising | stressed | 48.4 | 30.8 | 23.9 | -1.0 | -0.8 | -1.4 | -1.3 | 50.0 | 50.0 | -0.2 | 36.5 |  |
| 2019-12-04 | rising | stressed | 23.2 | 49.5 | 40.6 | 1.1 | 1.0 | 1.5 | 1.9 | 50.0 | 50.0 | -0.2 | 50.4 |  |
| 2020-04-16 | agitated | stressed | 24.7 | 32.5 | 57.8 | 2.8 | 2.4 | 2.9 | 2.8 | 50.0 | 50.0 | -0.2 | 51.8 |  |
| 2021-02-24 | agitated | stressed | 18.6 | 16.7 | 15.3 | 0.4 | 0.3 | 0.6 | 0.4 | 50.0 | 50.0 | -0.2 | 51.9 |  |
| 2022-08-18 | rising | stressed | 24.6 | 16.5 | 20.3 | -0.9 | -0.9 | -0.9 | -1.2 | 50.0 | 50.0 | -0.2 | 53.8 |  |
| 2022-09-19 | agitated | stressed | 15.7 | 36.3 | 25.2 | -1.5 | -1.1 | -1.1 | -0.9 | 50.0 | 50.0 | -0.2 | 58.6 |  |
| 2023-04-17 | rising | stressed | 3.5 | 14.5 | 7.4 | -0.3 | -0.2 | -1.3 | -1.1 | 50.0 | 50.0 | -0.2 | 30.6 |  |
| 2023-05-11 | agitated | stressed | 3.1 | 4.7 | 1.4 | 0.3 | 0.1 | 0.0 | 0.3 | 50.0 | 50.0 | -0.2 | 48.5 |  |
| 2023-09-15 | rising | stressed | 48.8 | 34.4 | 20.4 | 0.8 | 0.3 | 1.6 | 0.8 | 50.0 | 50.0 | -0.2 | 46.9 |  |
| 2024-06-18 | rising | stressed | 1.0 | 18.8 | 2.6 | 0.4 | 0.0 | 1.5 | 1.2 | 50.0 | 50.0 | -0.2 | 31.0 |  |
| 2024-08-21 | rising | stressed | 22.2 | 6.4 | 0.5 | -0.1 | 0.0 | -0.6 | -0.3 | 50.0 | 50.0 | -0.2 | 47.9 |  |

## 8. Shock detector against the labels
CUSUM alarms: 0; followed by a move into the high band within 30 calendar days: 0; moves into the high band preceded by an alarm within 30 days: 0 of 19.

Surprise: mean -0.011, sd 1.009 (should be ~1 if implied matches realised on average), 1%/99% -2.39/+2.32; max CUSUM pressure 0.9945 on 2021-04-20 (alarm when it reaches 0.999).

Shock score lead profile (5 days before):
| all | before_up | before_up_n | before_down | before_down_n |
|---|---|---|---|---|
| 30.07 | 30.54 | 114 | 30.71 | 135 |

Alarm dates (first 60): 

## 8b. Alignment check: Spearman correlation of each feature on day t with trailing 1-week realised vol on day t+k (k < 0 past, k > 0 future). A stamping error is a sharp peak at one future k; a genuine slow lead rises gently and plateaus
| index | k=-21 | k=-10 | k=-5 | k=-1 | k=+0 | k=+1 | k=+2 | k=+5 | k=+10 | k=+15 | k=+21 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stored_energy | -0.01 | -0.12 | -0.10 | -0.11 | -0.11 | -0.09 | -0.07 | -0.03 | -0.02 | 0.02 | -0.03 |
| gap_pct | 0.09 | 0.10 | 0.12 | 0.10 | 0.10 | 0.10 | 0.11 | 0.10 | 0.10 | 0.12 | 0.05 |
| complacency | -0.22 | -0.45 | -0.47 | -0.49 | -0.49 | -0.45 | -0.41 | -0.28 | -0.24 | -0.17 | -0.12 |
| jump_cluster_pct | -0.01 | 0.19 | 0.25 | 0.26 | 0.26 | 0.22 | 0.18 | 0.05 | 0.05 | 0.04 | 0.04 |
| coupling_shift_pct | -0.02 | 0.05 | 0.07 | 0.04 | 0.03 | 0.03 | 0.02 | 0.02 | -0.03 | -0.03 | 0.02 |
| coherence_pct | -0.00 | 0.14 | 0.13 | 0.14 | 0.14 | 0.11 | 0.08 | -0.01 | -0.01 | 0.02 | 0.04 |
| gap_z_signed | 0.08 | 0.10 | 0.10 | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 | 0.10 | 0.10 | 0.10 |
| drift_t | 0.07 | 0.11 | 0.11 | 0.10 | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 | 0.08 | 0.08 |
| desk_flow | -0.04 | -0.04 | -0.04 | -0.05 | -0.06 | -0.08 | -0.09 | -0.08 | -0.05 | -0.04 | -0.02 |

## 9. Fan from today's state (stressed, 2024-08-28); pressure from ['desk_flow']; pressure today -0.22 (0 = normal; the tilt moves the fan only when pressure is away from 0); whole-history tilt fit on the raw pressure scale (β per unit of pressure, not per sd as in section 2): {"beta": 0.1201, "gamma": 0.3365, "loglik": -2478.7673, "loglik0": -2489.1842, "gain_per_transition": 0.0041, "n": 2515, "k": 5}
| h | P(high band) constant | P(high band) tilted |
|---|---|---|
| 5 | 0.726 | 0.720 |
| 10 | 0.536 | 0.528 |
| 21 | 0.302 | 0.295 |

