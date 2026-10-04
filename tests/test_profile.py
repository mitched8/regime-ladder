import numpy as np
import pandas as pd
import pytest

from regime_ladder import features, profile, synth


@pytest.fixture(scope="module")
def world():
    m = synth.simulate_market(2520, seed=0)
    return m, profile.characteristics(m), m["state_true"].rename("regime")


def test_characteristics_are_point_in_time(world):
    m, ch, _ = world
    cut = "2019-06-28"
    full = ch.loc[:cut]; trunc = profile.characteristics(m, asof=cut)
    a, b = full.align(trunc, join="outer")
    both = a.notna() & b.notna()
    assert np.allclose(a[both].values.astype(float), b[both].values.astype(float), equal_nan=True)


def test_state_means_average_to_overall_mean(world):
    _, ch, lab = world
    ch = ch.dropna()  # the identity holds exactly only on complete rows
    prof = profile.state_profile(ch, lab)
    df = ch.join(lab, how="inner")
    w = df["regime"].value_counts(normalize=True)
    recon = sum(prof[f"{s}_mean"] * w[s] for s in w.index)
    assert np.allclose(recon.dropna(), prof.loc[recon.dropna().index, "all_mean"], atol=1e-9)


def test_profile_separates_stressed_on_skew_and_wings(world):
    _, ch, lab = world
    prof = profile.state_profile(ch, lab)
    top = set(profile.distinguishing(prof, "stressed", 3).index)
    assert top & {"rr_level", "fly_level", "term_slope"}
    assert profile.describe(prof, "stressed").startswith("stressed ·")


def test_today_placement_percentiles(world):
    _, ch, lab = world
    p = profile.today_placement(ch, lab)
    assert p["pct_within_state"].dropna().between(0, 1).all() and p["state"].iloc[0] == lab.iloc[-1]
