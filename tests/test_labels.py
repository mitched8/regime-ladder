import numpy as np
import pandas as pd
import pytest

from regime_ladder import features, labels, synth


@pytest.fixture(scope="module")
def long_market():
    return synth.simulate_market(2520, seed=0)


@pytest.fixture(scope="module")
def comp(long_market):
    return labels.composite(features.build(long_market, names=["vol_pct", "term_slope_pct", "rr_stress_pct"]))


def _fwd_pnl(m, h=5):
    st = m["state_true"].values
    return pd.Series([sum(synth.earn("straddle_atm", st[i + a], 21 - a) for a in range(1, h + 1)) if i + h < len(m) else np.nan
                      for i in range(len(m))], index=m.index)


def test_finder_prefers_five_states_on_pnl_and_three_on_vol(long_market, comp):
    on_pnl = labels.calibrate_states(comp, _fwd_pnl(long_market))
    on_vol = labels.calibrate_states(comp, long_market["rv_1m"].shift(-21))
    assert on_pnl["spec"]["partition"] == "5" and on_pnl["best5_r2"] > on_pnl["best3_r2"] + 0.03
    assert on_vol["best3_r2"] >= on_vol["best5_r2"] - 0.01


def test_calibrated_five_state_labels_are_usable(long_market, comp):
    cal = labels.calibrate_states(comp, _fwd_pnl(long_market))
    lab = labels.apply_spec(comp, cal["spec5"])
    assert (lab.values == long_market["state_true"].values).mean() >= 0.45
    assert cal["spec5"]["switches_per_year"] <= 30
    assert min(labels.episodes(lab).values()) >= 5
    assert set(lab) <= set(labels.STATES5)


def test_kink_estimator_finds_boundaries_in_range(long_market, comp):
    b1, b2 = labels.estimate_breaks(labels.ewma(comp, 3), long_market["rv_1m"].shift(-21))
    assert 20 <= b1 <= 50 and 60 <= b2 <= 90, (b1, b2)


def test_partition_rules():
    idx = pd.RangeIndex(6)
    level = pd.Series(["low", "low", "low", "mid", "high", "high"], index=idx)
    direction = pd.Series(["flat", "down", "up", "up", "flat", "down"], index=idx)
    out = labels.two_axis_labels(level, direction)
    # carry drifting down stays carry; low/up -> rising; high -> crisis; high/down -> normalising
    assert list(out) == ["carry", "carry", "rising", "rising", "crisis", "normalising"]
    level2 = pd.Series(["high", "low"], index=[0, 1]); dir2 = pd.Series(["down", "down"], index=[0, 1])
    assert list(labels.two_axis_labels(level2, dir2)) == ["normalising", "settling"]  # coming from above -> settling


def test_hysteresis_reduces_switching(long_market, comp):
    sc = labels.ewma(comp, 3)
    assert labels.switches(labels.level_labels(sc, 40, 72, 3.0)) < labels.switches(labels.level_labels(sc, 40, 72, 0.0))


def test_transition_matrix_five_states(long_market):
    M = labels.transition_matrix(long_market["state_true"], prior_strength=10)
    assert list(M.index) == list(labels.STATES5)
    assert np.allclose(M.sum(axis=1), 1.0) and (np.diag(M.values) > 0.8).all()


def test_durations_partition_the_series(long_market):
    d = labels.durations(long_market["state_true"])
    assert sum(sum(v) for v in d.values()) == len(long_market)
