import numpy as np
import pandas as pd
import pytest

from regime_ladder import discover, profile, synth


@pytest.fixture(scope="module")
def world():
    m = synth.simulate_market(2520, seed=0)
    return m, profile.characteristics(m), m["state_true"].rename("regime")


def test_discovery_recovers_hidden_rising_subtypes(world):
    m, ch, lab = world
    res = discover.discover(ch, lab, "rising")
    assert res["k"] == 2, res["bic"]
    ct = pd.crosstab(m.loc[res["labels"].index, "subtype_true"], res["labels"]).values
    agree = max(np.trace(ct), np.trace(ct[:, ::-1])) / ct.sum()
    assert agree >= 0.7
    assert all("·" in n for n in res["names"].values())


def test_discovery_finds_nothing_where_nothing_is_hidden(world):
    _, ch, lab = world
    for s in ("carry", "stressed"):
        assert discover.discover(ch, lab, s)["k"] == 1


def test_assign_is_consistent_with_fit(world):
    _, ch, lab = world
    res = discover.discover(ch, lab, "rising")
    day = res["labels"].index[10]
    assert discover.assign(ch.loc[day], res) == res["labels"].loc[day]
