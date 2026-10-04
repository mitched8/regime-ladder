"""Every registered feature must pass the point-in-time truncation test."""
import pandas as pd
import pytest

from regime_ladder import features


@pytest.mark.parametrize("name", list(features.FEATURES))
@pytest.mark.parametrize("cut", ["2017-06-30", "2019-03-15"])
def test_feature_is_point_in_time(market, name, cut):
    assert features.truncation_agrees(features.FEATURES[name], market, cut), f"{name} uses future data"


def test_a_leaky_feature_fails_the_test(market):
    def leaky(m, asof=None):
        m = m if asof is None else m.loc[:pd.Timestamp(asof)]
        return (m["atm_1m"] - m["atm_1m"].mean()).rename("leaky")  # full-sample mean = look-ahead
    assert not features.truncation_agrees(leaky, market, "2018-01-10")


def test_build_returns_all_features(market):
    f = features.build(market)
    assert list(f.columns) == list(features.FEATURES)
    assert f.index.equals(market.index)
