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
