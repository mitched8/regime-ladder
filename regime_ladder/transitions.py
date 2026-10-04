"""Transition probabilities: the constant matrix, and the time-varying one that lets today's leading
signals change today's propensity to move.

For a FRESH unit the horizon ladder needs no transition model: transitions inside the window are in
the outcomes. The matrix earns its place for three other things — explaining the ladder (how long
does carry usually last from here?), the path model for an existing book, and incorporating a
leading signal where today's propensity to move differs from the state's historical average.

Time variation is a row-normalised multinomial logit:

    P_ij(psi) = P0_ij · exp(beta · psi · (rank_j − rank_i)) / sum_k P0_ik · exp(beta · psi · (rank_k − rank_i))

so a positive pressure psi (with beta > 0) pushes mass toward higher-ranked (more stressed) states
and away from lower ones, rows always sum to one and every entry stays inside (0, 1). This is the
only valid form; scaling the off-diagonals and setting the diagonal as the remainder breaks for large
tilts. One parameter beta is fitted by maximum likelihood on the observed transitions, and it is
retained only if it improves the out-of-sample transition log-likelihood (`oos_gain`) — the first of
the two tests a leading feature must pass.

Forecasting from today needs a FUTURE PATH for the covariate. Each covariate declares one: frozen
at today's value, decaying to zero with a half-life, known in advance (a calendar), or zero. Holding
pressure fixed is a sustained-pressure scenario, not a forecast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .labels import ALL_STATES, STATES3, durations, transition_matrix

RANK = {"carry": 0.0, "settling": 0.5, "rising": 1.0, "agitated": 1.5, "normalising": 2.0, "stressed": 3.0, "extreme": 4.0,
        "transition": 1.0, "crisis": 2.0}


def ranks(names) -> np.ndarray:
    return np.array([RANK.get(n, float(i)) for i, n in enumerate(names)])


# ----------------------------------------------------------------------------- the tilt
def tilt(P0: pd.DataFrame, psi: float, beta: float, rank: np.ndarray | None = None) -> pd.DataFrame:
    """Row-normalised multinomial-logit tilt of P0 by pressure psi with sensitivity beta."""
    r = ranks(P0.index) if rank is None else rank
    if not np.isfinite(psi):
        return P0
    W = P0.values * np.exp(beta * psi * (r[None, :] - r[:, None]))
    return pd.DataFrame(W / W.sum(axis=1, keepdims=True), index=P0.index, columns=P0.columns)


def _tilt_array(P0: np.ndarray, r: np.ndarray, psi: np.ndarray, beta: float) -> np.ndarray:
    """Stack of tilted matrices for a vector of psi values: shape (len(psi), k, k). NaN psi -> P0."""
    psi = np.where(np.isfinite(psi), psi, 0.0)
    W = P0[None] * np.exp(beta * psi[:, None, None] * (r[None, None, :] - r[None, :, None]))
    return W / W.sum(axis=2, keepdims=True)


def _pressure_values(psi: pd.Series, beta: float, base: pd.Series | None, gamma: float, index) -> np.ndarray:
    """Effective pressure gamma * base + beta * psi on `index`; a missing value in either term counts as zero."""
    p = beta * np.nan_to_num(psi.reindex(index).values.astype(float))
    if base is not None and gamma != 0.0:
        p = p + gamma * np.nan_to_num(base.reindex(index).values.astype(float))
    return p


def loglik(labels: pd.Series, psi: pd.Series, P0: pd.DataFrame, beta: float, k: int = 1,
           base: pd.Series | None = None, gamma: float = 0.0) -> float:
    """Log-likelihood of the observed k-step transitions s_t -> s_{t+k} under the k-th power of the tilt by
    the effective pressure gamma * base_t + beta * psi_t, aligned to the FROM date and held frozen over
    the k days.

    k = 1 scores tonight's move. k > 1 scores where the state is k days out, which is the question a
    leading feature is meant to answer and the horizon the fan displays — and it is the only fair test
    against a smoothed, hysteretic labeller, whose one-step moves lag the market by days. Overlapping
    k-step observations are fine for comparing models on the same folds; they are not independent, so
    nothing here is a p-value.

    `base` with `gamma` is a baseline covariate already in the tilt (the continuous state score, which
    carries the distance to the next boundary): beta then measures what psi adds beyond it."""
    names = list(P0.index)
    idx = {n: i for i, n in enumerate(names)}
    s = labels.map(idx).values
    p = _pressure_values(psi, beta, base, gamma, labels.index)
    ok = ~np.isnan(s[:-k]) & ~np.isnan(s[k:])
    a, b, pp = s[:-k][ok].astype(int), s[k:][ok].astype(int), p[:-k][ok]
    T = _tilt_array(P0.values, ranks(names), pp, 1.0)
    Tk = T
    for _ in range(k - 1):
        Tk = Tk @ T
    return float(np.log(np.clip(Tk[np.arange(len(a)), a, b], 1e-300, None)).sum())


def _maximise(f, grid):
    ll = np.array([f(b) for b in grid])
    i = int(np.argmax(ll))
    lo, hi = grid[max(0, i - 1)], grid[min(len(grid) - 1, i + 1)]
    phi = (np.sqrt(5) - 1) / 2
    c, d = hi - phi * (hi - lo), lo + phi * (hi - lo)
    fc, fd = f(c), f(d)
    for _ in range(20):
        if fc > fd:
            hi, d, fd = d, c, fc
            c = hi - phi * (hi - lo); fc = f(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + phi * (hi - lo); fd = f(d)
    return float(0.5 * (lo + hi))


def fit_tilt(labels: pd.Series, psi: pd.Series, P0: pd.DataFrame, grid=None, k: int = 1,
             base: pd.Series | None = None) -> dict:
    """Maximum-likelihood beta for psi (k-step likelihood). With `base`, its coefficient gamma is fitted
    first on its own and held fixed while beta is fitted, so beta is psi's INCREMENTAL effect. Returns
    beta, gamma, log-likelihoods with and without psi, and the gain per observation (nats)."""
    grid = np.linspace(-3, 3, 31) if grid is None else np.asarray(grid)
    gamma = _maximise(lambda g: loglik(labels, base, P0, g, k), grid) if base is not None else 0.0
    beta = _maximise(lambda b: loglik(labels, psi, P0, b, k, base, gamma), grid)
    ll1, ll0 = loglik(labels, psi, P0, beta, k, base, gamma), loglik(labels, psi, P0, 0.0, k, base, gamma)
    n = int(labels.notna().sum() - k)
    return {"beta": beta, "gamma": gamma, "loglik": ll1, "loglik0": ll0, "gain_per_transition": (ll1 - ll0) / max(n, 1), "n": n, "k": k}


def oos_gain(labels: pd.Series, psi: pd.Series, P0: pd.DataFrame | None = None, folds: int = 4,
             min_train_frac: float = 0.4, prior_strength: float = 10.0, k: int = 1, base: pd.Series | None = None) -> pd.DataFrame:
    """Time-ordered folds: estimate P0 (sticky prior), gamma (for `base`) and beta on the training part, score
    the test part with the k-step likelihood with and without psi. One row per fold with the out-of-sample
    gain per observation and the fitted beta. psi earns its place if the gain is positive in most folds and
    in total."""
    n = len(labels)
    cuts = np.linspace(int(min_train_frac * n), n, folds + 1).astype(int)
    rows = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        tr, te = labels.iloc[:a], labels.iloc[a - k: b]  # overlap k days so the first test observation is scored
        names = [s for s in list(ALL_STATES) + list(STATES3) if s in set(labels)]
        P_tr = transition_matrix(tr, names=names, prior_strength=prior_strength) if P0 is None else P0
        fit = fit_tilt(tr, psi, P_tr, k=k, base=base)
        ll1 = loglik(te, psi, P_tr, fit["beta"], k, base, fit["gamma"])
        ll0 = loglik(te, psi, P_tr, 0.0, k, base, fit["gamma"])
        m = len(te) - k
        rows.append({"fold": len(rows) + 1, "train_end": tr.index[-1], "beta": fit["beta"], "gamma": fit["gamma"], "n_test": m,
                     "oos_gain_per_transition": (ll1 - ll0) / max(m, 1)})
    out = pd.DataFrame(rows)
    out.attrs["total_gain_per_transition"] = float((out["oos_gain_per_transition"] * out["n_test"]).sum() / max(out["n_test"].sum(), 1))
    out.attrs["folds_positive"] = int((out["oos_gain_per_transition"] > 0).sum())
    out.attrs["k"] = k
    return out


# ----------------------------------------------------------------------------- forecasting from today
def covariate_path(psi_now: float, horizon: int, policy: str = "frozen", halflife: float = 10.0, path=None) -> np.ndarray:
    """Declared future path of a covariate over the next `horizon` days."""
    if policy == "frozen":
        return np.full(horizon, psi_now)
    if policy == "decay":
        return psi_now * 0.5 ** (np.arange(1, horizon + 1) / halflife)
    if policy == "zero":
        return np.zeros(horizon)
    if policy == "calendar":
        p = np.asarray(path, dtype=float)
        return p[:horizon] if len(p) >= horizon else np.concatenate([p, np.zeros(horizon - len(p))])
    raise ValueError(policy)


def fan(P0: pd.DataFrame, pi0: pd.Series, psi_path=None, beta: float = 0.0) -> pd.DataFrame:
    """State distribution 1..H days ahead from a start vector pi0 (a point mass or today's probability vector),
    under the constant matrix (psi_path None) or the tilted sequence. Row h = pi after h steps."""
    pi = pi0.reindex(P0.index).fillna(0.0).values.astype(float)
    pi = pi / pi.sum()
    H = 21 if psi_path is None else len(psi_path)
    rows = []
    for h in range(H):
        M = P0.values if psi_path is None else tilt(P0, float(psi_path[h]), beta).values
        pi = pi @ M
        rows.append(pi)
    return pd.DataFrame(rows, index=pd.RangeIndex(1, H + 1, name="h"), columns=P0.index)


def expected_days(f: pd.DataFrame) -> pd.Series:
    return f.sum(axis=0).rename("expected_days")


def p_state_at(f: pd.DataFrame, states, hs=(5, 10, 21)) -> pd.Series:
    states = [states] if isinstance(states, str) else list(states)
    return pd.Series({h: float(f.loc[h, states].sum()) for h in hs if h in f.index}, name="p_" + "+".join(states))


# ----------------------------------------------------------------------------- diagnostics
def implied_vs_observed_duration(P: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Geometric holding time 1/(1-P_ii) against the observed mean run length — the first-order check."""
    d = durations(labels)
    rows = []
    for s in P.index:
        obs = d.get(s, [])
        rows.append({"state": s, "implied_days": 1 / max(1e-9, 1 - P.loc[s, s]), "observed_mean_days": float(np.mean(obs)) if obs else np.nan,
                     "episodes": len(obs)})
    return pd.DataFrame(rows).set_index("state")


def simulate_tilted_chain(P0: pd.DataFrame, psi: np.ndarray, beta: float, seed: int = 0, start: int = 0) -> np.ndarray:
    """A state path whose daily matrix is tilt(P0, psi_t, beta): ground truth for the fit and for the synthetic world."""
    rng = np.random.default_rng(seed)
    T = _tilt_array(P0.values, ranks(P0.index), np.asarray(psi, dtype=float), beta)
    s = np.empty(len(psi), dtype=int)
    s[0] = start
    for t in range(1, len(psi)):
        s[t] = rng.choice(len(P0), p=T[t - 1, s[t - 1]])
    return s
