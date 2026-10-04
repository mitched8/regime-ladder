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


@pytest.fixture(scope="module")
def cal(long_market, comp):
    return labels.calibrate_states(comp, _fwd_pnl(long_market), level_target=long_market["atm_1m"].shift(-5))


def test_finder_prefers_direction_states_on_pnl(cal):
    # the level x direction partitions beat level-only on forward P&L: direction is a P&L fact
    assert cal["spec"]["partition"] in ("6", "6x")
    assert max(cal["r2"]["6"], cal["r2"]["6x"]) > cal["r2"]["3"] + 0.03


def test_level_bounds_land_between_the_true_bands(cal, long_market, comp):
    # composite means by true state: carry ~22, settling ~36 | rising ~55, agitated ~62, normalising ~70 | stressed ~78
    b1, b2 = cal["specs"]["6"]["bounds"]
    assert 35 <= b1 <= 55 and 62 <= b2 <= 85, (b1, b2)
    lab = labels.apply_spec(comp, cal["specs"]["6"])
    ok = comp.notna()
    assert (lab[ok].values == long_market.loc[ok, "state_true"].values).mean() >= 0.45
    assert cal["specs"]["6"]["switches_per_year"] <= 30
    assert min(labels.episodes(lab).values()) >= 5
    assert set(lab) <= set(labels.STATES6)


def test_step_fit_beats_kink_on_a_ramp():
    # a step between two bands with a ramp in between: the step fit puts the boundary mid-ramp
    rng = np.random.default_rng(0)
    s = pd.Series(rng.uniform(0, 100, 4000))
    y = pd.Series(np.where(s < 30, -5, np.where(s < 70, 0, 5)) + rng.normal(0, 1, 4000))
    b1, b2 = labels.estimate_breaks(s, y, 2)
    assert 25 <= b1 <= 35 and 65 <= b2 <= 75, (b1, b2)


def test_extreme_band_is_a_tail_and_merges_when_rare(cal, comp):
    s6x = cal["specs"]["6x"]
    assert len(s6x["bounds"]) == 3 and s6x["bounds"][2] > s6x["bounds"][1]
    assert s6x["bounds"][2] >= labels.ewma(comp, s6x["halflife"]).quantile(0.95)
    lab = labels.apply_spec(comp, s6x)
    if s6x["extreme_merged"]:
        assert "extreme" not in set(lab) and s6x["extreme_episodes"] < 5
    else:
        assert labels.episodes(lab)["extreme"] >= 5


def test_merge_rare_rule():
    lab = pd.Series(["carry"] * 10 + ["extreme"] * 3 + ["stressed"] * 5 + ["extreme"] * 2 + ["carry"] * 5)
    merged, info = labels.merge_rare(lab, min_episodes=5)
    assert info == {"merged": True, "rare_episodes": 2, "rare_days": 5} and "extreme" not in set(merged)
    kept, info2 = labels.merge_rare(lab, min_episodes=2)
    assert not info2["merged"] and (kept == lab).all()


def test_partition_rules():
    idx = pd.RangeIndex(7)
    level = pd.Series(["low", "low", "low", "mid", "mid", "high", "high"], index=idx)
    direction = pd.Series(["flat", "down", "up", "up", "flat", "flat", "down"], index=idx)
    out = labels.two_axis_labels(level, direction)
    # carry drifting down stays carry; low/up -> rising; mid/flat -> agitated; high -> stressed; high/down -> normalising
    assert list(out) == ["carry", "carry", "rising", "rising", "agitated", "stressed", "normalising"]
    level2 = pd.Series(["high", "mid", "low"], index=[0, 1, 2]); dir2 = pd.Series(["down", "down", "down"], index=[0, 1, 2])
    assert list(labels.two_axis_labels(level2, dir2)) == ["normalising", "normalising", "settling"]  # coming down from above
    level3 = pd.Series(["extreme", "extreme", "high"], index=[0, 1, 2]); dir3 = pd.Series(["up", "down", "down"], index=[0, 1, 2])
    assert list(labels.two_axis_labels(level3, dir3)) == ["extreme", "extreme", "normalising"]


def test_hysteresis_reduces_switching(long_market, comp):
    sc = labels.ewma(comp, 3)
    assert labels.switches(labels.level_labels(sc, (45, 73), 3.0)) < labels.switches(labels.level_labels(sc, (45, 73), 0.0))


def test_transition_matrix_orders_named_states(long_market):
    M = labels.transition_matrix(long_market["state_true"], prior_strength=10)
    assert list(M.index) == list(labels.ALL_STATES)
    assert np.allclose(M.sum(axis=1), 1.0) and (np.diag(M.values) > 0.75).all()


def test_durations_partition_the_series(long_market):
    d = labels.durations(long_market["state_true"])
    assert sum(sum(v) for v in d.values()) == len(long_market)
