import numpy as np
import pandas as pd
import pytest

from regime_ladder import labels, synth, transitions


@pytest.fixture(scope="module")
def P0():
    return pd.DataFrame(synth.DEFAULT_P, index=synth.STATES, columns=synth.STATES)


def test_tilt_is_a_valid_matrix_and_monotone_in_pressure(P0):
    assert np.allclose(transitions.tilt(P0, 0.0, 1.0).values, P0.values)
    for psi in (0.5, 1.0, 2.0, 5.0):
        T = transitions.tilt(P0, psi, 1.0)
        assert np.allclose(T.sum(axis=1), 1.0) and (T.values >= 0).all() and (T.values <= 1).all()
    up = [transitions.tilt(P0, psi, 1.0).loc["carry", ["rising", "agitated", "stressed"]].sum() for psi in (0, 1, 2)]
    assert up[0] < up[1] < up[2]
    down = [transitions.tilt(P0, psi, 1.0).loc["stressed", "normalising"] for psi in (0, 1, 2)]
    assert down[0] > down[1] > down[2]
    assert transitions.tilt(P0, np.nan, 1.0).equals(P0)


def test_fit_recovers_a_known_beta_and_rejects_a_null(P0):
    rng = np.random.default_rng(0)
    n = 6000
    psi = np.zeros(n)
    for t in range(1, n):
        psi[t] = 0.9 * psi[t - 1] + rng.normal(0, np.sqrt(1 - 0.81))
    idx = pd.bdate_range("2000-01-03", periods=n)
    for beta_true in (1.0, 0.0):
        s = transitions.simulate_tilted_chain(P0, psi, beta_true, seed=1)
        lab = pd.Series(np.array(synth.STATES)[s], index=idx)
        fit = transitions.fit_tilt(lab, pd.Series(psi, index=idx), P0)
        assert abs(fit["beta"] - beta_true) < 0.25, fit
        g = transitions.oos_gain(lab, pd.Series(psi, index=idx), folds=4)
        if beta_true > 0:
            assert g.attrs["total_gain_per_transition"] > 0.01 and g.attrs["folds_positive"] >= 3
        else:
            assert abs(g.attrs["total_gain_per_transition"]) < 0.005


def test_fan_and_paths(P0):
    pi0 = pd.Series({"carry": 1.0})
    f0 = transitions.fan(P0, pi0, psi_path=np.zeros(21), beta=1.0)
    f1 = transitions.fan(P0, pi0, psi_path=transitions.covariate_path(1.5, 21, "frozen"), beta=1.0)
    f2 = transitions.fan(P0, pi0, psi_path=transitions.covariate_path(1.5, 21, "decay", halflife=3), beta=1.0)
    assert np.allclose(f0.sum(axis=1), 1.0) and np.allclose(f1.sum(axis=1), 1.0)
    p0, p1, p2 = (transitions.p_state_at(f, ["stressed", "extreme"], hs=(21,))[21] for f in (f0, f1, f2))
    assert p0 < p2 < p1  # sustained pressure > decaying pressure > none
    ed = transitions.expected_days(f1)
    assert abs(ed.sum() - 21) < 1e-9 and ed["carry"] < transitions.expected_days(f0)["carry"]
    cal = transitions.covariate_path(0.0, 5, "calendar", path=[0, 0, 2.0])
    assert list(cal) == [0, 0, 2.0, 0, 0]


def test_duration_check_matches_a_true_markov_chain(P0):
    m = synth.simulate_market(5040, seed=0)
    d = transitions.implied_vs_observed_duration(P0, m["state_true"])
    named = d.drop(index="extreme")
    assert ((named["observed_mean_days"] / named["implied_days"]).between(0.6, 1.5)).all(), d
