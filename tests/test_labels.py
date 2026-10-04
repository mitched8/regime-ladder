import numpy as np

from regime_ladder import features, labels


def _score(market):
    f = features.build(market, names=["vol_pct", "term_slope_pct", "rr_stress_pct"])
    return labels.ewma(labels.composite(f), 3)


def test_kink_estimator_recovers_boundaries(market):
    b1, b2 = labels.estimate_breaks(_score(market), market["rv_1m"].shift(-21))
    assert 32 <= b1 <= 48 and 64 <= b2 <= 80, (b1, b2)


def test_labels_track_true_regime_and_hysteresis_reduces_switching(market):
    s = _score(market)
    raw = labels.threshold_labels(s, 40, 72, 0.0)
    hyst = labels.threshold_labels(s, 40, 72, 3.0)
    truth = market["regime_true"]
    assert (hyst.values == truth.values).mean() >= 0.55
    assert labels.switches(hyst) < labels.switches(raw)


def test_transition_matrix_is_row_stochastic_and_sticky(market):
    M = labels.transition_matrix(market["regime_true"], prior_strength=10)
    assert np.allclose(M.sum(axis=1), 1.0)
    assert (np.diag(M.values) > 0.8).all()


def test_durations_partition_the_series(market):
    d = labels.durations(market["regime_true"])
    assert sum(sum(v) for v in d.values()) == len(market)
