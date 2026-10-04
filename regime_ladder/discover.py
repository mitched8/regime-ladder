"""Sub-state discovery inside a named state.

Within one named state, cluster days on the standardised characteristics vector. A sub-state
exists only if (1) a k > 1 model is preferred by BIC, (2) every cluster has at least
`min_episodes` contiguous runs, and (3) the clustering is stable: models fitted on the first and
second halves of the state's history agree on at least `stability` of days. Otherwise k = 1 and
the named state stands alone. Sub-states are descriptive until, separately, conditioning on them
beats the parent state's ladder out of sample.

Deliberately dependency-free: a small k-means and a BIC proxy for spherical clusters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .profile import distinguishing, state_profile


def _kmeans(X, k, rng, n_init=6, iters=60):
    best = None
    for _ in range(n_init):
        c = X[rng.choice(len(X), 1)]
        for _ in range(1, k):  # k-means++ seeding
            d2 = ((X[:, None, :] - c[None]) ** 2).sum(-1).min(1)
            c = np.vstack([c, X[rng.choice(len(X), 1, p=d2 / d2.sum())]])
        for _ in range(iters):
            lab = ((X[:, None, :] - c[None]) ** 2).sum(-1).argmin(1)
            new = np.vstack([X[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
            if np.allclose(new, c):
                break
            c = new
        inertia = float(((X - c[lab]) ** 2).sum())
        if best is None or inertia < best[2]:
            best = (lab, c, inertia)
    return best


def _bic(X, inertia, k):
    n, d = X.shape
    return n * d * np.log(max(inertia, 1e-12) / (n * d)) + k * d * np.log(n)


def _runs(mask: pd.Series) -> int:
    v = mask.values.astype(int)
    return int(((v[1:] == 1) & (v[:-1] == 0)).sum() + (v[0] == 1))


def _match(c_a, c_b):
    """Greedy centroid matching; returns permutation mapping b's clusters onto a's."""
    k = len(c_a)
    D = ((c_a[:, None, :] - c_b[None]) ** 2).sum(-1)
    perm, used = {}, set()
    for _ in range(k):
        i, j = np.unravel_index(np.argmin(np.where(np.isin(np.arange(k), list(used))[None, :] | np.isin(np.arange(k), list(perm))[:, None], np.inf, D)), D.shape)
        perm[int(i)] = int(j); used.add(int(j))
    return perm


def discover(chars: pd.DataFrame, labels: pd.Series, state: str, kmax: int = 4, min_episodes: int = 5,
             stability: float = 0.7, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    days = labels.index[labels == state]
    Xdf = chars.loc[days].dropna()
    mu, sd = Xdf.mean(), Xdf.std().replace(0, 1)
    X = ((Xdf - mu) / sd).values
    result = {"state": state, "k": 1, "labels": pd.Series(state, index=Xdf.index), "mu": mu, "sd": sd, "centroids": X.mean(0, keepdims=True),
              "bic": {1: _bic(X, float(((X - X.mean(0)) ** 2).sum()), 1)}, "stability": {}, "episodes": {}, "names": {state: state}}
    if len(X) < 20 * kmax:
        return result
    half = len(X) // 2
    for k in range(2, kmax + 1):
        lab, c, inertia = _kmeans(X, k, rng)
        result["bic"][k] = _bic(X, inertia, k)
        eps = [_runs(pd.Series(lab == j, index=Xdf.index)) for j in range(k)]
        la, ca, _ = _kmeans(X[:half], k, rng)
        lb, cb, _ = _kmeans(X[half:], k, rng)
        perm = _match(ca, cb)
        pa = ((X[:, None, :] - ca[None]) ** 2).sum(-1).argmin(1)
        pb = ((X[:, None, :] - cb[None]) ** 2).sum(-1).argmin(1)
        agree = float(np.mean([perm[int(a)] == int(b) for a, b in zip(pa, pb)]))
        result["stability"][k] = agree
        result["episodes"][k] = eps
        feasible = min(eps) >= min_episodes and agree >= stability
        if feasible and result["bic"][k] < result["bic"][result["k"]]:
            result.update(k=k, labels=pd.Series([f"{state}-{j + 1}" for j in lab], index=Xdf.index), centroids=c)
    if result["k"] > 1:
        prof = state_profile(chars.loc[Xdf.index], result["labels"])
        result["names"] = {s: s + " · " + " · ".join(f"{col} {'↑' if r.d > 0 else '↓'}" for col, r in distinguishing(prof, s, 2).iterrows())
                           for s in result["labels"].unique()}
    return result


def assign(chars_row: pd.Series, model: dict) -> str:
    """Nearest sub-state for a new day's characteristics under a fitted model."""
    if model["k"] == 1:
        return model["state"]
    x = ((chars_row[model["mu"].index] - model["mu"]) / model["sd"]).values.astype(float)
    j = int(((model["centroids"] - x) ** 2).sum(1).argmin())
    return f"{model['state']}-{j + 1}"
