# Review packet · stage `leading` · calibration B_null (synthetic)

| key | value |
|---|---|
| pair | EURUSD |
| data | 2015-01-01 to 2024-08-28 (2520 days) |
| finder / leading target | expected forward 5d straddle P&L given the true state path (synthetic, noiseless); diagnostic outcome at 21d |
| git commit | 556ee10 |
| config hash | ba38712377 |
| generated | 2026-10-04 17:42 |

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
 "retained": [],
 "pressure_transition_gain": NaN,
 "pressure_folds_positive": 0,
 "pass": false
}
```
## 2. Screen (one feature at a time vs state + composite baseline)
Target substitutions: drift_t: rr_25d unavailable, judged on straddle_atm; gap_z_signed: rr_25d unavailable, judged on straddle_atm

`delta_r2_diag` is the outcome test on straddle_atm_h21 (the same archetype at the diagnostic horizon); it does not decide retention.

| feature | target | n | n_pairs | r2_base | delta_r2 | p_perm | delta_r2_low | delta_r2_diag | delta_r2_lagged | folds_r2_up | transition_gain | folds_transition_up | holdout_tested | delta_r2_holdout | transition_gain_holdout | retain | fail_reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| drift_t | straddle_atm | 1895 | 1 | 0.3530 | 0.0247 | 0.0495 | 0.0191 | 0.0550 | -0.0287 | 3 | 0.0001 | 2 | False |  |  | False | transition_gain;folds_transition_up |
| gap_z_signed | straddle_atm | 1895 | 1 | 0.3530 | 0.0194 | 0.0693 | -0.0086 | 0.0290 | -0.0187 | 3 | 0.0015 | 3 | False |  |  | False | transition_gain;p_perm |
| jump_cluster_pct | straddle_atm | 1808 | 1 | 0.3909 | -0.0009 | 0.3366 | -0.0091 | -0.0087 | -0.0015 | 1 | 0.0002 | 2 | False |  |  | False | delta_r2;folds_r2_up;transition_gain;folds_transition_up;p_perm |
| coupling_shift_pct | straddle_atm | 1702 | 1 | 0.3162 | -0.0025 | 0.5050 | 0.0013 | -0.0064 | -0.0022 | 1 | 0.0015 | 3 | False |  |  | False | delta_r2;folds_r2_up;transition_gain;p_perm |
| coherence_pct | straddle_atm | 1807 | 1 | 0.3909 | -0.0037 | 0.4752 | -0.0001 | -0.0475 | 0.0000 | 1 | -0.0032 | 1 | False |  |  | False | delta_r2;folds_r2_up;transition_gain;folds_transition_up;p_perm |
| complacency | straddle_atm | 1807 | 1 | 0.3909 | -0.0090 | 0.7723 | -0.0315 | 0.0020 | -0.0063 | 1 | 0.0005 | 2 | False |  |  | False | delta_r2;folds_r2_up;transition_gain;folds_transition_up;p_perm |
| desk_flow | straddle_atm | 1895 | 1 | 0.3530 | -0.0128 | 0.8119 | -0.0159 | 0.0011 | -0.0078 | 1 | -0.0001 | 1 | False |  |  | False | delta_r2;folds_r2_up;transition_gain;folds_transition_up;p_perm |
| stored_energy | straddle_atm | 1768 | 1 | 0.2369 | -0.0489 | 0.8812 | -0.1120 | -0.0876 | -0.0027 | 2 | -0.0040 | 2 | False |  |  | False | delta_r2;folds_r2_up;transition_gain;folds_transition_up;p_perm |
| gap_pct | straddle_atm | 1768 | 1 | 0.2369 | -0.0558 | 0.9109 | -0.1073 | -0.1153 | -0.0021 | 1 | -0.0041 | 2 | False |  |  | False | delta_r2;folds_r2_up;transition_gain;folds_transition_up;p_perm |

Transition test by horizon (k = 5 decides; 10 and 21 are diagnostics):
| feature | n_transitions | transition_gain | transition_gain_k10 | transition_gain_k21 | beta_mean |
|---|---|---|---|---|---|
| drift_t | 1955 | 0.0001 | -0.0000 | -0.0044 | -0.0844 |
| gap_z_signed | 1955 | 0.0015 | 0.0013 | -0.0010 | -0.0718 |
| jump_cluster_pct | 1807 | 0.0002 | 0.0002 | 0.0013 | 0.0783 |
| coupling_shift_pct | 1701 | 0.0015 | 0.0015 | -0.0075 | 0.0301 |
| coherence_pct | 1806 | -0.0032 | -0.0063 | -0.0134 | 0.0488 |
| complacency | 1806 | 0.0005 | 0.0052 | 0.0092 | 0.0202 |
| desk_flow | 1955 | -0.0001 | -0.0004 | 0.0012 | -0.0040 |
| stored_energy | 1767 | -0.0040 | -0.0139 | -0.0336 | -0.0267 |
| gap_pct | 1767 | -0.0041 | -0.0116 | -0.0244 | -0.0761 |

## 2b. Lift: P(high band within 20 days | feature in its top decile) vs base rate
| feature | p_event_signal | p_event_all | lift | lift_lo | lift_hi | n_signal | n_signal_runs | n_events_after_signal |
|---|---|---|---|---|---|---|---|---|
| stored_energy | 0.027 | 0.159 | 0.169 | 0.053 | 0.331 | 297 | 33 | 8 |
| gap_pct | 0.041 | 0.159 | 0.258 | 0.000 | 0.563 | 292 | 38 | 12 |
| complacency | 0.014 | 0.164 | 0.087 | 0.000 | 0.243 | 279 | 35 | 4 |
| jump_cluster_pct | 0.134 | 0.164 | 0.813 | 0.349 | 1.480 | 217 | 22 | 29 |
| coupling_shift_pct | 0.089 | 0.165 | 0.537 | 0.135 | 1.017 | 237 | 33 | 21 |
| coherence_pct | 0.220 | 0.164 | 1.340 | 0.698 | 2.170 | 264 | 41 | 58 |
| gap_z_signed | 0.079 | 0.156 | 0.507 | 0.171 | 0.940 | 303 | 43 | 24 |
| drift_t | 0.143 | 0.156 | 0.919 | 0.406 | 1.668 | 244 | 45 | 35 |
| desk_flow | 0.109 | 0.156 | 0.697 | 0.249 | 1.188 | 377 | 40 | 41 |

## 3. Feature distributions
| index | count | mean | std | min | 5% | 50% | 95% | max | first_valid | last_valid | share_at_mode |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stored_energy | 2272.0 | 26.1 | 20.0 | 0.0 | 1.9 | 22.0 | 68.1 | 88.0 | 2015-12-15 | 2024-08-28 | 0.0 |
| gap_pct | 2272.0 | 48.7 | 29.2 | 0.0 | 4.9 | 47.8 | 97.4 | 100.0 | 2015-12-15 | 2024-08-28 | 0.0 |
| complacency | 2311.0 | 51.5 | 17.7 | 8.4 | 23.1 | 50.8 | 83.4 | 93.8 | 2015-10-21 | 2024-08-28 | 0.0 |
| jump_cluster_pct | 2312.0 | 34.5 | 31.0 | 0.0 | 0.0 | 29.8 | 90.5 | 100.0 | 2015-10-20 | 2024-08-28 | 0.3 |
| coupling_shift_pct | 2206.0 | 45.3 | 27.4 | 0.0 | 4.1 | 44.5 | 91.7 | 100.0 | 2016-03-16 | 2024-08-28 | 0.0 |
| coherence_pct | 2311.0 | 49.8 | 29.3 | 0.0 | 4.0 | 49.7 | 94.1 | 100.0 | 2015-10-21 | 2024-08-28 | 0.0 |
| gap_z_signed | 2460.0 | 0.1 | 0.9 | -2.4 | -1.3 | 0.1 | 1.4 | 2.9 | 2015-03-26 | 2024-08-28 | 0.0 |
| drift_t | 2460.0 | 0.1 | 1.0 | -2.9 | -1.6 | 0.2 | 1.7 | 3.1 | 2015-03-26 | 2024-08-28 | 0.0 |
| desk_flow | 2460.0 | 52.3 | 6.4 | 50.0 | 50.0 | 50.0 | 66.1 | 96.6 | 2015-03-26 | 2024-08-28 | 0.8 |

## 4. Mean feature value by state (0..100)
| index | agitated | carry | normalising | rising | settling | stressed |
|---|---|---|---|---|---|---|
| stored_energy | 16.5 | 38.9 | 17.7 | 23.3 | 27.9 | 18.4 |
| gap_pct | 37.5 | 57.0 | 64.5 | 43.2 | 61.1 | 46.2 |
| complacency | 43.3 | 67.4 | 29.2 | 52.4 | 46.8 | 37.4 |
| jump_cluster_pct | 39.8 | 27.3 | 21.5 | 45.5 | 35.9 | 30.2 |
| coupling_shift_pct | 50.8 | 41.8 | 30.5 | 41.8 | 43.7 | 47.2 |
| coherence_pct | 56.3 | 47.4 | 48.1 | 53.5 | 40.3 | 46.6 |
| gap_z_signed | 0.1 | 0.1 | 0.5 | 0.2 | 0.1 | -0.0 |
| drift_t | 0.2 | 0.1 | 0.8 | 0.2 | -0.1 | -0.0 |
| desk_flow | 50.1 | 54.3 | 53.0 | 50.2 | 56.1 | 50.0 |

## 5. Correlations (features and the composite)
| index | stored_energy | gap_pct | complacency | jump_cluster_pct | coupling_shift_pct | coherence_pct | gap_z_signed | drift_t | desk_flow | composite |
|---|---|---|---|---|---|---|---|---|---|---|
| stored_energy | 1.00 | 0.86 | 0.59 | -0.24 | -0.20 | -0.15 | 0.12 | 0.06 | 0.59 | -0.46 |
| gap_pct | 0.86 | 1.00 | 0.19 | -0.12 | -0.22 | -0.14 | 0.16 | 0.10 | 0.56 | -0.22 |
| complacency | 0.59 | 0.19 | 1.00 | -0.44 | -0.12 | -0.12 | 0.01 | -0.04 | 0.22 | -0.67 |
| jump_cluster_pct | -0.24 | -0.12 | -0.44 | 1.00 | 0.08 | 0.28 | -0.03 | -0.05 | -0.05 | 0.08 |
| coupling_shift_pct | -0.20 | -0.22 | -0.12 | 0.08 | 1.00 | 0.09 | -0.01 | -0.01 | -0.11 | 0.06 |
| coherence_pct | -0.15 | -0.14 | -0.12 | 0.28 | 0.09 | 1.00 | 0.05 | 0.04 | -0.05 | 0.05 |
| gap_z_signed | 0.12 | 0.16 | 0.01 | -0.03 | -0.01 | 0.05 | 1.00 | 0.89 | 0.08 | 0.00 |
| drift_t | 0.06 | 0.10 | -0.04 | -0.05 | -0.01 | 0.04 | 0.89 | 1.00 | 0.02 | 0.03 |
| desk_flow | 0.59 | 0.56 | 0.22 | -0.05 | -0.11 | -0.05 | 0.08 | 0.02 | 1.00 | -0.33 |
| composite | -0.46 | -0.22 | -0.67 | 0.08 | 0.06 | 0.05 | 0.00 | 0.03 | -0.33 | 1.00 |

## 6. Lead profile: mean value in the days before ANY move up the state ranking (e.g. carry to rising), before moves down, and on all days. Moves skip ranks (a rising-to-stressed move counts once), so up and down counts need not match
| feature | window_days | all_days | before_escalation | before_de_escalation | n_escalations | n_de_escalations |
|---|---|---|---|---|---|---|
| stored_energy | 5 | 26.1 | 26.3 | 21.7 | 70.0 | 81.0 |
| stored_energy | 20 | 26.1 | 29.7 | 21.6 | 70.0 | 81.0 |
| gap_pct | 5 | 48.7 | 47.5 | 50.4 | 70.0 | 81.0 |
| gap_pct | 20 | 48.7 | 51.7 | 49.6 | 70.0 | 81.0 |
| complacency | 5 | 51.5 | 53.1 | 43.7 | 70.0 | 81.0 |
| complacency | 20 | 51.5 | 56.5 | 43.2 | 70.0 | 81.0 |
| jump_cluster_pct | 5 | 34.5 | 40.6 | 32.7 | 70.0 | 81.0 |
| jump_cluster_pct | 20 | 34.5 | 35.5 | 33.9 | 70.0 | 81.0 |
| coupling_shift_pct | 5 | 45.3 | 43.7 | 41.8 | 70.0 | 81.0 |
| coupling_shift_pct | 20 | 45.3 | 40.9 | 42.1 | 70.0 | 81.0 |
| coherence_pct | 5 | 49.8 | 52.9 | 44.1 | 70.0 | 81.0 |
| coherence_pct | 20 | 49.8 | 51.8 | 49.1 | 70.0 | 81.0 |
| gap_z_signed | 5 | 0.1 | 0.2 | 0.2 | 70.0 | 81.0 |
| gap_z_signed | 20 | 0.1 | 0.3 | 0.2 | 70.0 | 81.0 |
| drift_t | 5 | 0.1 | 0.2 | 0.2 | 70.0 | 81.0 |
| drift_t | 20 | 0.1 | 0.3 | 0.2 | 70.0 | 81.0 |
| desk_flow | 5 | 52.3 | 50.9 | 52.3 | 70.0 | 81.0 |
| desk_flow | 20 | 52.3 | 51.6 | 51.3 | 70.0 | 81.0 |

## 7. Every move into the high band (19): values before it
| date | from | to | stored_energy_t-20 | stored_energy_t-5 | stored_energy_t-1 | gap_z_signed_t-5 | gap_z_signed_t-1 | drift_t_t-5 | drift_t_t-1 | desk_flow_t-5 | desk_flow_t-1 | shock_max_20d | days_since_cusum_alarm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2015-11-06 | rising | stressed |  |  |  | -0.9 | -0.9 | -1.5 | -1.2 | 50.0 | 50.0 | 68.9 | 172 |
| 2016-06-07 | rising | stressed | 15.9 | 7.7 | 6.4 | -0.6 | -0.4 | -0.4 | -0.5 | 50.0 | 50.0 | 58.0 | 386 |
| 2018-09-27 | rising | stressed | 66.0 | 45.7 | 39.5 | 1.0 | 0.8 | 1.4 | 0.7 | 50.0 | 50.0 | 33.0 | 1228 |
| 2018-11-28 | agitated | stressed | 24.9 | 46.7 | 42.9 | -1.1 | -1.0 | -1.4 | -1.1 | 50.0 | 50.0 | 60.0 | 1290 |
| 2019-09-06 | agitated | stressed | 0.5 | 22.8 | 26.5 | 0.7 | 0.6 | 1.3 | 1.2 | 50.0 | 50.0 | 35.0 | 1572 |
| 2020-03-16 | agitated | stressed | 31.4 | 28.6 | 31.4 | 0.8 | 0.8 | 0.8 | 0.8 | 50.0 | 50.0 | 35.6 | 1764 |
| 2020-03-27 | agitated | stressed | 36.0 | 48.5 | 44.4 | 1.4 | 1.4 | 1.4 | 2.1 | 50.0 | 50.0 | 44.8 | 1775 |
| 2020-06-15 | agitated | stressed | 34.4 | 5.5 | 7.6 | 0.5 | 0.4 | 0.5 | -0.0 | 50.0 | 50.0 | 55.1 | 1855 |
| 2020-10-29 | agitated | stressed | 26.8 | 21.5 | 18.2 | -0.9 | -1.0 | -0.5 | -0.6 | 50.0 | 50.0 | 66.2 | 1991 |
| 2020-12-17 | rising | stressed | 21.9 | 11.7 | 6.2 | -0.4 | -0.2 | -1.2 | -0.8 | 50.0 | 50.0 | 49.7 | 2040 |
| 2021-01-28 | rising | stressed | 2.2 | 2.9 | 9.1 | -0.1 | 0.3 | 0.5 | 0.9 | 50.0 | 50.0 | 30.9 | 2082 |
| 2021-05-31 | rising | stressed | 16.5 | 35.1 | 29.2 | 0.9 | 0.6 | 1.4 | 1.0 | 50.0 | 50.0 | 33.5 | 2205 |
| 2022-04-28 | agitated | stressed | 9.8 | 28.5 | 27.7 | 0.9 | 0.9 | 1.6 | 1.3 | 50.0 | 50.0 | 37.9 | 2537 |
| 2022-05-24 | agitated | stressed | 41.1 | 17.3 | 1.4 | 0.7 | 0.1 | 0.8 | 0.3 | 50.0 | 50.0 | 52.6 | 2563 |
| 2022-05-26 | agitated | stressed | 22.5 | 16.7 | 0.2 | 0.7 | 0.0 | 0.8 | 0.1 | 50.0 | 50.0 | 53.2 | 2565 |
| 2022-08-19 | agitated | stressed | 34.0 | 13.7 | 13.4 | 0.6 | 0.5 | 0.7 | 1.3 | 50.0 | 50.0 | 35.8 | 2650 |
| 2023-04-05 | rising | stressed | 39.4 | 25.0 | 31.6 | 0.6 | 0.8 | 0.5 | 1.0 | 50.0 | 50.0 | 52.6 | 2879 |
| 2023-06-15 | agitated | stressed | 36.3 | 46.6 | 43.7 | 0.9 | 0.9 | 0.8 | 0.8 | 50.0 | 50.0 | 29.3 | 2950 |
| 2024-05-21 | rising | stressed | 18.8 | 7.9 | 5.1 | -0.2 | 0.1 | 0.5 | 1.5 | 50.0 | 50.0 | 46.2 | 3291 |

## 8. Shock detector against the labels
CUSUM alarms: 1; followed by a move into the high band within 30 calendar days: 0; moves into the high band preceded by an alarm within 30 days: 0 of 19.

Surprise: mean +0.013, sd 1.008 (should be ~1 if implied matches realised on average), 1%/99% -2.41/+2.27; max CUSUM pressure 1.0000 on 2015-05-18 (alarm when it reaches 0.999).

Shock score lead profile (5 days before):
| all | before_up | before_up_n | before_down | before_down_n |
|---|---|---|---|---|
| 29.52 | 32.35 | 70 | 27.54 | 81 |

Alarm dates (first 60): 2015-05-18

## 8b. Alignment check: Spearman correlation of each feature on day t with trailing 1-week realised vol on day t+k (k < 0 past, k > 0 future). A stamping error is a sharp peak at one future k; a genuine slow lead rises gently and plateaus
| index | k=-21 | k=-10 | k=-5 | k=-1 | k=+0 | k=+1 | k=+2 | k=+5 | k=+10 | k=+15 | k=+21 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stored_energy | -0.08 | -0.27 | -0.31 | -0.32 | -0.32 | -0.29 | -0.27 | -0.21 | -0.14 | -0.12 | -0.08 |
| gap_pct | -0.03 | -0.13 | -0.17 | -0.17 | -0.17 | -0.16 | -0.15 | -0.15 | -0.12 | -0.12 | -0.10 |
| complacency | -0.17 | -0.45 | -0.45 | -0.46 | -0.46 | -0.41 | -0.37 | -0.23 | -0.16 | -0.12 | -0.04 |
| jump_cluster_pct | -0.04 | 0.21 | 0.25 | 0.29 | 0.30 | 0.26 | 0.21 | 0.08 | 0.01 | -0.00 | -0.03 |
| coupling_shift_pct | 0.06 | 0.03 | 0.04 | 0.08 | 0.08 | 0.07 | 0.06 | 0.05 | 0.06 | 0.04 | 0.04 |
| coherence_pct | 0.00 | 0.11 | 0.10 | 0.13 | 0.13 | 0.11 | 0.09 | 0.02 | 0.00 | 0.03 | -0.00 |
| gap_z_signed | -0.04 | -0.03 | -0.02 | -0.01 | -0.01 | -0.00 | 0.00 | 0.02 | 0.03 | 0.02 | 0.01 |
| drift_t | -0.06 | -0.03 | -0.02 | -0.01 | -0.01 | -0.00 | 0.00 | 0.02 | 0.04 | 0.04 | 0.04 |
| desk_flow | -0.05 | -0.14 | -0.22 | -0.26 | -0.28 | -0.28 | -0.27 | -0.25 | -0.20 | -0.21 | -0.14 |

## 9. Fan from today's state (agitated, 2024-08-28): constant matrix only (nothing retained, so no pressure index)
| h | P(high band) constant |
|---|---|
| 5 | 0.068 |
| 10 | 0.106 |
| 21 | 0.135 |

