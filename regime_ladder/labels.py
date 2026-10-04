"""Regime labelling: composite score -> smoothing -> two thresholds with hysteresis.

Deliberately transparent. Boundaries are estimated by a two-kink piecewise-linear
fit of a forward target on the score (grid search), inside whatever training
window the caller passes. The labeller is stateful (hysteresis), so labels are
produced by a forward loop and are point-in-time by construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NAMES = ("carry", "transition", "crisis")


def ewma(s: pd.Series, halflife: float) -> pd.Series:
    return s.ewm(halflife=halflife, adjust=False).mean()


def composite(features: pd.DataFrame, weights: dict | None = None) -> pd.Series:
    """Weighted average of feature columns (expected already on a comparable 0-100 scale)."""
    w = pd.Series(weights) if weights else pd.Series(1.0, index=features.columns)
    w = w.reindex(features.columns).fillna(0.0)
    w = w / w.sum()
    return (features * w).sum(axis=1, min_count=1)


def threshold_labels(score: pd.Series, b1: float, b2: float, delta: float = 0.0, names=NAMES) -> pd.Series:
    """Hard labels with hysteresis band +/- delta around each boundary."""
    out = []
    state = 0
    for v in score.values:
        if np.isnan(v):
            out.append(names[state])
            continue
        if state == 0 and v > b1 + delta:
            state = 1
        if state == 1:
            if v > b2 + delta:
                state = 2
            elif v < b1 - delta:
                state = 0
        elif state == 2 and v < b2 - delta:
            state = 1
        out.append(names[state])
    return pd.Series(out, index=score.index, name="regime")


def estimate_breaks(score: pd.Series, target: pd.Series, grid: np.ndarray | None = None,
                    min_gap: float = 10.0) -> tuple[float, float]:
    """Two-kink piecewise-linear fit: target ~ score + (score-b1)+ + (score-b2)+. Returns argmin SSE."""
    df = pd.concat([score.rename("s"), target.rename("y")], axis=1).dropna()
    s, y = df["s"].values, df["y"].values
    grid = grid if grid is not None else np.quantile(s, np.linspace(0.1, 0.9, 33))
    best, best_sse = (np.nan, np.nan), np.inf
    for i, b1 in enumerate(grid):
        for b2 in grid[i + 1:]:
            if b2 - b1 < min_gap:
                continue
            X = np.column_stack([np.ones_like(s), s, np.maximum(s - b1, 0), np.maximum(s - b2, 0)])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            sse = float(((y - X @ beta) ** 2).sum())
            if sse < best_sse:
                best, best_sse = (float(b1), float(b2)), sse
    return best


def switches(labels: pd.Series) -> int:
    return int((labels.values[1:] != labels.values[:-1]).sum())


def transition_matrix(labels: pd.Series, names=NAMES, prior_strength: float = 0.0, stickiness: float = 0.95) -> pd.DataFrame:
    """Row-stochastic transition matrix from day-to-day counts, optional Dirichlet prior toward stickiness."""
    idx = {n: i for i, n in enumerate(names)}
    C = np.zeros((3, 3))
    v = labels.map(idx).values
    for a, b in zip(v[:-1], v[1:]):
        C[a, b] += 1
    prior = prior_strength * np.where(np.eye(3, dtype=bool), stickiness, (1 - stickiness) / 2)
    M = C + prior
    return pd.DataFrame(M / M.sum(axis=1, keepdims=True), index=names, columns=names)


def durations(labels: pd.Series) -> dict:
    """Observed run lengths per regime (to compare against the geometric a constant chain implies)."""
    out = {n: [] for n in set(labels)}
    run, cur = 0, None
    for v in labels.values:
        if v == cur:
            run += 1
        else:
            if cur is not None:
                out[cur].append(run)
            cur, run = v, 1
    out[cur].append(run)
    return out
