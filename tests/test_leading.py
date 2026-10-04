import numpy as np
import pandas as pd
import pytest

from regime_ladder import features, labels, leading, synth, transitions

CFG = {"min_delta_r2": 0.005, "min_folds_up": 3, "min_transition_gain": 0.002}


def _fwd_pnl(m, h=5):
    s = m["state_true"].values
    return pd.Series([sum(synth.earn("straddle_atm", s[i + a], 21 - a) for a in range(1, h + 1)) if i + h < len(m) else np.nan
                      for i in range(len(m))], index=m.index)


def _world(energy_beta):
    m = synth.simulate_market(2520, seed=0, energy_beta=energy_beta)
    comp = labels.composite(features.build(m, names=["vol_pct", "term_slope_pct", "rr_stress_pct"]))
    return m, comp, _fwd_pnl(m)


@pytest.fixture(scope="module")
def null_world():
    return _world(0.0)


@pytest.fixture(scope="module")
def energy_world():
    # a strong effect on purpose: this is a test of the machinery, not of the effect size
    return _world(3.0)


@pytest.mark.parametrize("name", list(leading.LEADING))
def test_leading_features_are_point_in_time(market, name):
    assert features.truncation_agrees(leading.LEADING[name], market, "2018-03-15"), f"{name} uses future data"


def test_stored_energy_tracks_the_world_it_is_named_for(energy_world):
    m, _, _ = energy_world
    L = leading.build(m)
    assert L["gap_pct"].corr(m["pressure_true"]) > 0.2
    assert L["stored_energy"].corr(m["pressure_true"]) > 0.2
    # the product is the hypothesis: energy is higher than the gap alone where the holder looks weak
    weak = m["atm_1m"] < 9
    assert L.loc[weak, "stored_energy"].mean() / L.loc[weak, "gap_pct"].mean() > L.loc[~weak, "stored_energy"].mean() / L.loc[~weak, "gap_pct"].mean()
    assert L["stored_energy"].dropna().between(0, 100).all()


def test_screen_retains_nothing_in_the_null_world(null_world):
    m, comp, y = null_world
    sc = leading.screen(leading.build(m), m["state_true"], comp, y, CFG)
    assert not sc["retain"].any(), sc
    assert sc["delta_r2"].abs().max() < 0.01


def test_screen_retains_the_true_pressure_and_an_energy_feature_where_energy_matters(energy_world):
    m, comp, y = energy_world
    L = leading.build(m, names=["stored_energy", "gap_pct", "complacency", "jump_cluster_pct"])
    L["pressure_true"] = m["pressure_true"]
    sc = leading.screen(L, m["state_true"], comp, y, CFG).set_index("feature")
    assert sc.loc["pressure_true", "retain"], sc
    assert sc.loc["pressure_true", "transition_gain"] > 0.005 and sc.loc["pressure_true", "delta_r2"] > 0.01
    assert sc.loc[["stored_energy", "gap_pct"], "retain"].any(), sc


def test_true_pressure_survives_estimated_labels(energy_world):
    """The transition test is run against labels the market would actually produce, not the truth. With the
    continuous state score as a baseline covariate in the tilt and the k-step likelihood, the true pressure is
    still retained; without the baseline it would be confounded by distance to the boundary (pressure is highest
    when vol is low, i.e. far below the first boundary) and come out with the wrong sign."""
    import yaml
    from regime_ladder.__main__ import make_labels
    m, comp, y = energy_world
    cfg = yaml.safe_load(open("configs/default.yaml"))
    _, lab_est, _ = make_labels(m, cfg, "EURUSD", y)
    assert (lab_est.values == m["state_true"].values).mean() < 0.8  # genuinely estimated, not the truth
    res = leading.incremental_value(m["pressure_true"], lab_est, comp, y)
    assert leading.retain(res, CFG), res
    assert res["beta_mean"] > 0
    naive = transitions.oos_gain(lab_est, (m["pressure_true"] - m["pressure_true"].mean()) / m["pressure_true"].std(), folds=4, k=1)
    assert naive.attrs["total_gain_per_transition"] < res["transition_gain"]


def test_a_lagged_feature_loses_value(energy_world):
    m, comp, y = energy_world
    now = leading.incremental_value(m["pressure_true"], m["state_true"], comp, y)
    late = leading.incremental_value(m["pressure_true"], m["state_true"], comp, y, lag=10)
    assert late["transition_gain"] < now["transition_gain"] and late["delta_r2"] < now["delta_r2"]


def test_pressure_index_feeds_the_tilt(energy_world):
    m, _, _ = energy_world
    L = leading.build(m, names=["gap_pct", "stored_energy"])
    psi = leading.pressure(L)
    assert psi.dropna().between(-2.5, 2.5).all()
    g = transitions.oos_gain(m["state_true"], psi, folds=4)
    assert g.attrs["total_gain_per_transition"] > 0 and g["beta"].mean() > 0


def test_event_proximity_is_a_ramp_into_the_event():
    idx = pd.bdate_range("2024-01-01", periods=30)
    e = leading.event_proximity(idx, ["2024-01-22"], horizon=5)
    assert e.loc["2024-01-22"] == 1.0 and abs(e.loc["2024-01-19"] - 0.8) < 1e-9 and e.loc["2024-01-12"] == 0.0 and e.loc["2024-01-23"] == 0.0
    L = leading.build(synth.simulate_market(300, seed=0), names=["gap_pct"], events=["2015-06-15"])
    assert "event_proximity" in L.columns and L["event_proximity"].max() == 100


# ----------------------------------------------------------------------------- plumbing: pooling, hold-out, horizons, lift, external world
@pytest.fixture(scope="module")
def external_world():
    """An exogenous AR(1) tilts transitions five days after it is observed; nothing in the surface or the path knows it."""
    m = synth.simulate_market(2520, seed=0, external_beta=1.0, external_lead=5)
    comp = labels.composite(features.build(m, names=["vol_pct", "term_slope_pct", "rr_stress_pct"]))
    return m, comp, _fwd_pnl(m)


def test_external_series_has_its_own_stream_and_leads_only_where_it_is_causal(null_world, external_world):
    m0, _, _ = null_world
    m1 = synth.simulate_market(2520, seed=0, external_beta=0.0)
    assert (m1["state_true"].values == m0["state_true"].values).all()           # adding the column changed nothing else
    m, comp, y = external_world
    sc = leading.screen(pd.DataFrame({"x": m["external_true"], "x_late": m["external_true"].shift(-10)}), m["state_true"], comp, y, CFG).set_index("feature")
    assert sc.loc["x", "retain"] and sc.loc["x", "transition_gain_k21"] > sc.loc["x", "transition_gain"] > 0
    l1 = leading.lift(m["external_true"], m["state_true"]); l0 = leading.lift(m0["external_true"], m0["state_true"])
    assert l1["lift"] > 1.3 and l1["lift_lo"] > 1 and l1["n_signal_runs"] > 20
    assert l0["lift_lo"] < 1 < l0["lift_hi"] or l0["lift"] < 1.2


def test_pooled_screen_stacks_pairs_and_cuts_folds_at_common_dates(external_world):
    m, comp, y = external_world
    m2 = synth.simulate_market(2520, seed=1, external_beta=1.0, external_lead=5)
    comp2 = labels.composite(features.build(m2, names=["vol_pct", "term_slope_pct", "rr_stress_pct"])); y2 = _fwd_pnl(m2)
    L = {"A": pd.DataFrame({"x": m["external_true"]}), "B": pd.DataFrame({"x": m2["external_true"]})}
    sc = leading.screen(L, {"A": m["state_true"], "B": m2["state_true"]}, {"A": comp, "B": comp2}, {"A": y, "B": y2}, CFG).iloc[0]
    assert sc["n_pairs"] == 2 and sc["n"] > 4000 and sc["retain"]
    r = leading.incremental_value({"A": m["external_true"], "B": m2["external_true"]}, {"A": m["state_true"], "B": m2["state_true"]},
                                  {"A": comp, "B": comp2}, {"A": y, "B": y2}, ks=())
    assert r["n_pairs"] == 2 and r["delta_r2"] > 0 and r["transition_gain"] > 0


def test_holdout_confirms_or_removes_the_flag(external_world):
    m, comp, y = external_world
    L = pd.DataFrame({"x": m["external_true"], "noise": pd.Series(np.random.default_rng(3).normal(size=len(m)), index=m.index)})
    cfg = {**CFG, "holdout_frac": 0.2, "max_to_holdout": 1}
    sc = leading.screen(L, m["state_true"], comp, y, cfg).set_index("feature")
    assert sc.loc["x", "holdout_tested"] and sc.loc["x", "delta_r2_holdout"] > -cfg["min_delta_r2"] and sc.loc["x", "retain"]
    assert not sc.loc["noise", "holdout_tested"] and not sc.loc["noise", "retain"]
    # a feature that passes the walk-forward keeps the flag unless the hold-out CONTRADICTS it (beyond minus the threshold)
    r = {"n": 1000, "delta_r2": 0.02, "folds_r2_up": 4, "transition_gain": 0.01, "folds_transition_up": 4, "holdout_tested": True,
         "delta_r2_holdout": -0.001, "transition_gain_holdout": 0.01}
    assert leading.retain(r, cfg) and not leading.retain({**r, "delta_r2_holdout": -0.01}, cfg) and not leading.retain({**r, "transition_gain_holdout": -0.01}, cfg)
    # the permutation p-value gates retention when present
    assert not leading.retain({**r, "p_perm": 0.2}, {**cfg, "max_p_perm": 0.05}) and leading.retain({**r, "p_perm": 0.01}, {**cfg, "max_p_perm": 0.05})


def test_feature_targets_route_signed_features_to_the_signed_outcome(external_world):
    m, comp, y = external_world
    L = leading.build(m, names=["gap_pct", "gap_z_signed"])
    sc = leading.screen(L, m["state_true"], comp, {"straddle_atm": y, "rr_25d": -y}, CFG, feature_targets={"gap_z_signed": "rr_25d"}).set_index("feature")
    assert sc.loc["gap_pct", "target"] == "straddle_atm" and sc.loc["gap_z_signed", "target"] == "rr_25d"


def test_permutation_p_value_separates_signal_from_noise(energy_world):
    m, comp, y = energy_world
    L = pd.DataFrame({"pressure_true": m["pressure_true"], "noise": pd.Series(np.random.default_rng(1).normal(size=len(m)), index=m.index).rolling(10).mean()})
    sc = leading.screen(L, m["state_true"], comp, y, {**CFG, "n_perm": 50, "max_p_perm": 0.05}, diag_targets={"default": "default"}).set_index("feature")
    assert sc.loc["pressure_true", "p_perm"] < 0.05 < sc.loc["noise", "p_perm"]
    assert abs(sc.loc["pressure_true", "delta_r2_diag"] - sc.loc["pressure_true", "delta_r2"]) < 1e-12   # same target -> same gain


def test_pressure_zscore_window_puts_features_on_one_footing(energy_world):
    m, _, _ = energy_world
    L = leading.build(m, names=["stored_energy", "gap_pct", "drift_t"])
    psi = leading.pressure(L, zscore_window=504)
    assert psi.dropna().abs().median() < 1.0 and psi.first_valid_index() > L.first_valid_index()
    raw = leading.pressure(L[["stored_energy"]])
    assert raw.mean() < -0.5                                                     # the centring-at-50 bias the z-score removes
