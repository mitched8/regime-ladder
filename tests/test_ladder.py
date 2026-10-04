"""Identities the ladder must satisfy exactly, plus sanity on the interval machinery."""
import numpy as np
import pandas as pd

from regime_ladder import ladder, schema, synth


def test_unconditional_row_equals_plain_mean(cum_true, true_labels):
    lad = ladder.ladder(cum_true, true_labels, ci="hac")
    for (pair, arch, tenor, h), g in cum_true.groupby(["pair", "archetype", "tenor_days", "h"]):
        row = lad[(lad.archetype == arch) & (lad.h == h) & (lad.regime == "ALL")].iloc[0]
        assert abs(row["mean"] - g["cum_pnl"].mean()) < 1e-9


def test_increments_telescope_to_cumulative(cum_true, true_labels):
    lad = ladder.ladder(cum_true, true_labels, ci="hac")
    inc = ladder.increments(lad)
    for key, g in inc.groupby(["archetype", "regime"]):
        g = g.sort_values("h_from")
        total = (g["earn_per_day"] * (g["h_to"] - g["h_from"] + 1)).sum()
        last = lad[(lad.archetype == key[0]) & (lad.regime == key[1])].sort_values("h")["mean"].iloc[-1]
        assert abs(total - last) < 1e-9


def test_persistence_split_identity(trade_days, true_labels):
    sp = ladder.persistence_split(trade_days, true_labels)
    recon = (1 - sp["q_broke"]) * sp["ev_stayed"].fillna(0) + sp["q_broke"] * sp["ev_broke"].fillna(0)
    assert np.allclose(recon, sp["ev"], atol=1e-9)
    assert sp["q_broke"].between(0, 1).all()


def test_shrinkage_moves_toward_unconditional(cum_true, true_labels):
    lad = ladder.shrink(ladder.ladder(cum_true, true_labels, ci="hac"), kappa=50)
    allm = lad[lad.regime == "ALL"].set_index(["archetype", "h"])["mean"]
    g = lad[lad.regime != "ALL"]
    m_all = np.array([allm[(a, h)] for a, h in zip(g.archetype, g.h)])
    between = ((g["mean_shrunk"] - g["mean"]) * (m_all - g["mean"]) >= -1e-12)
    assert between.all()
    assert lad.loc[lad.regime == "ALL", "w_shrink"].eq(1.0).all()


def test_effective_n_scales_with_horizon(cum_true, true_labels):
    lad = ladder.ladder(cum_true, true_labels, ci="hac")
    g = lad[(lad.regime == "ALL") & (lad.archetype == "rr_25d")].set_index("h")
    assert abs(g.loc[1, "n_eff"] - g.loc[1, "n"]) < 1e-6
    assert abs(g.loc[10, "n_eff"] - g.loc[10, "n"] / 10) < 0.5


def test_hac_se_exceeds_naive_under_positive_autocorrelation():
    rng = np.random.default_rng(0)
    e = rng.normal(size=5000)
    x = np.convolve(e, np.ones(5), mode="valid")  # 5-day overlapping sums: strong positive autocorrelation
    naive = x.std(ddof=1) / np.sqrt(len(x))
    assert ladder.hac_se(x, bandwidth=5) > 1.8 * naive
    iid = rng.normal(size=5000)
    assert abs(ladder.hac_se(iid, bandwidth=5) / (iid.std(ddof=1) / np.sqrt(len(iid))) - 1) < 0.1


def test_block_bootstrap_interval_brackets_mean():
    rng = np.random.default_rng(1)
    x = rng.normal(0.3, 1, 400)
    lo, hi = ladder.block_bootstrap_ci(x, block_len=6, n_boot=800, rng=rng)
    assert lo < x.mean() < hi and 0 < hi - lo < 0.5


def test_unlabelled_entries_are_counted_not_silently_dropped(cum_true, true_labels):
    lab = true_labels.iloc[30:]
    out = ladder.attach_entry_labels(cum_true.drop(columns="regime"), lab)
    assert out.attrs["unlabelled_entries"] > 0


def test_episode_count(true_labels):
    eps = ladder.count_episodes(true_labels)["EURUSD"]
    named = set(synth.STATES) - {"extreme"}
    assert named <= set(eps) and all(eps[s] > 3 for s in named)  # extreme is rare by design: present or not
