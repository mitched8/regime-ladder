import numpy as np
import pandas as pd
import pytest

from regime_ladder import features, shock


@pytest.fixture(scope="module")
def fig7():
    m = shock.figure7_path(seed=0)
    return m, shock.surprise(m)


def test_three_sigma_rule_alarms_on_isolated_days_and_misses_the_shift(fig7):
    _, z = fig7
    a = shock.three_sigma(z)
    assert a.iloc[:250].sum() == 3 and a.iloc[250:].sum() == 0


def test_cusum_ignores_isolated_days_and_detects_the_sustained_shift(fig7):
    _, z = fig7
    h = shock.calibrate_cusum((z ** 2).iloc[1:200].values, k=0.3, arl0=1000)
    c = shock.cusum(z ** 2, 0.3, h, null_window=120)
    assert c["alarm"].iloc[:250].sum() == 0
    assert c["alarm"].iloc[250:].any() and int(np.argmax(c["alarm"].values[250:])) <= 25


def test_cusum_threshold_scales_with_the_run_length_target(fig7):
    _, z = fig7
    x2 = (z ** 2).iloc[1:200].values
    assert shock.calibrate_cusum(x2, 0.3, arl0=250) < shock.calibrate_cusum(x2, 0.3, arl0=1000) < shock.calibrate_cusum(x2, 0.3, arl0=4000)


def test_bocd_resets_on_a_clean_variance_change():
    rng = np.random.default_rng(1)
    x = pd.Series(np.concatenate([rng.normal(0, 1, 300), rng.normal(0, 3, 60)]))
    b = shock.bocd(x, hazard=1 / 100, m=10)
    assert b["p_run_short"].iloc[50:300].mean() < 0.15
    assert b["p_run_short"].iloc[300:310].max() > 0.8
    assert b["map_run"].iloc[320] < 40


def test_har_forecast_is_no_worse_than_persistence_out_of_sample(market):
    fwd = market["rv_1m"].shift(-21)
    f = shock.har_series(market)
    both = pd.concat([f, fwd.rename("fwd"), market["rv_1m"].rename("naive")], axis=1).iloc[504:].dropna()
    assert both["har_vol_forecast"].corr(both["fwd"]) >= both["naive"].corr(both["fwd"]) - 0.02
    assert 4 < both["har_vol_forecast"].mean() < 20
    unshrunk = shock.har_series(market, ridge=0.0)
    u = pd.concat([unshrunk, fwd.rename("fwd")], axis=1).iloc[504:].dropna()
    assert u.corr().iloc[0, 1] < both["har_vol_forecast"].corr(both["fwd"])  # the shrinkage earns its place here


@pytest.mark.parametrize("fn", [shock.surprise, shock.cusum_series, shock.bocd_series, shock.har_series])
def test_detectors_are_point_in_time(market, fn):
    assert features.truncation_agrees(fn, market, "2018-03-15")


def test_shock_score_bounds_and_lead(market):
    sc = shock.shock_score(market)
    s = sc["score"].dropna()
    assert s.between(0, 100).all() and len(s) > 1000
    lp = shock.lead_profile(sc["score"], market["state_true"])
    assert lp["before_up_n"].iloc[0] > 20
    assert lp["before_up"].iloc[0] > lp["all"].iloc[0]  # the score carries some lead into escalations


def test_blend_pi_moves_mass_up_and_keeps_a_distribution():
    pi = pd.Series({"carry": 0.7, "rising": 0.2, "stressed": 0.1})
    out = shock.blend_pi(pi, score=50, w_max=0.5)
    assert abs(out.sum() - 1) < 1e-12 and out["carry"] < pi["carry"] and out["stressed"] > pi["stressed"]
    assert np.allclose(shock.blend_pi(pi, score=0), pi)
