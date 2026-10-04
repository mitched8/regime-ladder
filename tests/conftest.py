import pytest

from regime_ladder import ladder, schema, synth


@pytest.fixture(scope="session")
def market():
    return synth.simulate_market(1260, seed=0)


@pytest.fixture(scope="session")
def trade_days(market):
    return schema.coerce(synth.simulate_trades(market, seed=100))


@pytest.fixture(scope="session")
def true_labels(market):
    return schema.labels_frame("EURUSD", market.index, market["regime_true"].values)


@pytest.fixture(scope="session")
def cum_true(trade_days, true_labels):
    return ladder.attach_entry_labels(ladder.cumulative(trade_days), true_labels)
