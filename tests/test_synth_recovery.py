"""The statistics must recover the analytic truth from synthetic data with true labels.

This is the test that the ladder, its intervals and its effective-N logic are right,
independent of any real data. If it fails, nothing downstream can be trusted.
"""
import numpy as np

from regime_ladder import ladder, schema, synth


def _coverage(seed, ci):
    m = synth.simulate_market(1260, seed=seed)
    td = synth.simulate_trades(m, seed=seed + 100)
    lab = schema.labels_frame("EURUSD", m.index, m["state_true"].values)
    lad = ladder.ladder(ladder.attach_entry_labels(ladder.cumulative(td), lab), lab, ci=ci, n_boot=600)
    x = lad[lad["regime"] != "ALL"].merge(synth.true_ladder(), on=["archetype", "regime", "h"])
    x = x[x["n_eff"] >= 20]
    cov = ((x["ci_lo"] <= x["true_ev"]) & (x["true_ev"] <= x["ci_hi"])).mean()
    z = ((x["mean"] - x["true_ev"]).abs() / x["se"]).median()
    return cov, z


def test_block_bootstrap_intervals_cover_truth():
    covs, zs = zip(*[_coverage(s, "boot") for s in range(6)])
    assert np.mean(covs) >= 0.80, f"mean 90% coverage {np.mean(covs):.2f} too low"
    assert np.median(zs) < 1.2


def test_hac_intervals_cover_truth():
    covs, _ = zip(*[_coverage(s, "hac") for s in range(6)])
    assert np.mean(covs) >= 0.75


def test_true_ladder_matches_brute_force_expectation():
    """Analytic truth equals the mean of the deterministic earn path over many simulated regime paths."""
    rng = np.random.default_rng(3)
    P, T, a = synth.DEFAULT_P, 21, "rr_25d"
    truth = synth.true_ladder(archetypes=(a,), tenor_days=T).set_index(["regime", "h"])["true_ev"]
    for k, kname in enumerate(synth.STATES):
        tot = np.zeros(T)
        n = 4000
        for _ in range(n):
            s = k
            cum = 0.0
            for d in range(1, T + 1):
                s = rng.choice(len(P), p=P[s])
                cum += synth.earn(a, synth.STATES[s], T - d)
                tot[d - 1] += cum
        mc = tot / n
        tol = 0.35 if kname != "extreme" else 0.6  # extreme paths have the widest spread of outcomes
        assert abs(mc[20] - truth[(kname, 21)]) < tol, (kname, mc[20], truth[(kname, 21)])
