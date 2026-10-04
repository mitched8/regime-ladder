"""Regime labelling: level x direction -> a small set of named states, with calibration.

Level: a composite stress score, smoothed, cut at two boundaries with hysteresis -> low/mid/high.
Direction: the smoothed score's slope, cut at two thresholds with hysteresis -> down/flat/up.
A partition table maps (level, direction) to a named state; "keep" means stay in the previous
state (used for the ambiguous mid/flat cell). The labeller is a forward loop, so labels are
point-in-time by construction.

`calibrate_states` is the state finder: given the score history and a forward target (forward
realised vol, forward surface change, or forward archetype outcome), it estimates boundaries by
regression kink, direction thresholds from the slope distribution, chooses smoothing and
hysteresis by out-of-sample separation of the target, and compares the 3-state (level only) and
5-state (level x direction) partitions on the same criterion. It returns a state spec and a
report; it does not decide for you.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LEVELS = ("low", "mid", "high")
DIRS = ("down", "flat", "up")
STATES5 = ("carry", "rising", "crisis", "normalising", "settling")
STATES3 = ("carry", "transition", "crisis")

# (level, direction) -> state, or a dict keyed by the previous state ("_" = default). "keep" = stay.
PARTITION5 = {
    ("low", "flat"): "carry",
    ("low", "up"): "rising",
    ("low", "down"): {"carry": "carry", "_": "settling"},                       # carry drifting down is still carry
    ("mid", "flat"): "keep",
    ("mid", "up"): {"crisis": "crisis", "normalising": "crisis", "_": "rising"},  # re-acceleration from a fading crisis
    ("mid", "down"): {"crisis": "normalising", "normalising": "normalising", "carry": "keep", "_": "settling"},
    ("high", "up"): "crisis", ("high", "flat"): "crisis", ("high", "down"): "normalising",
}
PARTITION3 = {(lv, d): {"low": "carry", "mid": "transition", "high": "crisis"}[lv] for lv in LEVELS for d in DIRS}


# ----------------------------------------------------------------------------- primitives
def ewma(s: pd.Series, halflife: float) -> pd.Series:
    return s.ewm(halflife=halflife, adjust=False).mean()


def composite(features: pd.DataFrame, weights: dict | None = None) -> pd.Series:
    w = pd.Series(weights) if weights else pd.Series(1.0, index=features.columns)
    w = w.reindex(features.columns).fillna(0.0)
    return (features * (w / w.sum())).sum(axis=1, min_count=1)


def direction_score(score: pd.Series, window: int = 5) -> pd.Series:
    """Slope of the smoothed score, in score points per day."""
    return (score - score.shift(window)) / window


def _three_way(x: pd.Series, lo: float, hi: float, delta: float, names) -> pd.Series:
    """Hysteresis labeller onto three ordered classes: below lo, between, above hi."""
    out, state = [], 1
    for v in x.values:
        if not np.isnan(v):
            if state == 0 and v > lo + delta:
                state = 1
            if state == 1:
                if v > hi + delta:
                    state = 2
                elif v < lo - delta:
                    state = 0
            elif state == 2 and v < hi - delta:
                state = 1
        out.append(names[state])
    return pd.Series(out, index=x.index)


def threshold_labels(score: pd.Series, b1: float, b2: float, delta: float = 0.0, names=STATES3) -> pd.Series:
    """Level-only labels (the 3-state design). Kept for the benchmark and for tests."""
    s = _three_way(score, b1, b2, delta, names)
    # start in the lowest class, as before
    return s.rename("regime")


def level_labels(score, b1, b2, delta=0.0):
    return _three_way(score, b1, b2, delta, LEVELS)


def direction_labels(dscore, d_down, d_up, delta=0.0):
    return _three_way(dscore, d_down, d_up, delta, DIRS)


def two_axis_labels(level: pd.Series, direction: pd.Series, partition: dict = PARTITION5, start: str = "carry") -> pd.Series:
    out, prev = [], start
    for lv, d in zip(level.values, direction.values):
        s = partition.get((lv, d), "keep")
        if isinstance(s, dict):
            s = s.get(prev, s.get("_", "keep"))
        prev = prev if s == "keep" else s
        out.append(prev)
    return pd.Series(out, index=level.index, name="regime")


def estimate_breaks(score: pd.Series, target: pd.Series, grid: np.ndarray | None = None, min_gap: float = 10.0) -> tuple[float, float]:
    """Two-kink piecewise-linear fit of target on score; returns the SSE-minimising (b1, b2)."""
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


def transition_matrix(labels: pd.Series, names=None, prior_strength: float = 0.0, stickiness: float = 0.95) -> pd.DataFrame:
    names = list(names) if names is not None else list(dict.fromkeys(n for n in STATES5 + STATES3 if n in set(labels))) or sorted(set(labels))
    idx = {n: i for i, n in enumerate(names)}
    k = len(names)
    C = np.zeros((k, k))
    v = labels.map(idx).values
    for a, b in zip(v[:-1], v[1:]):
        C[a, b] += 1
    prior = prior_strength * np.where(np.eye(k, dtype=bool), stickiness, (1 - stickiness) / max(1, k - 1))
    M = C + prior
    M = M / np.where(M.sum(axis=1, keepdims=True) == 0, 1, M.sum(axis=1, keepdims=True))
    return pd.DataFrame(M, index=names, columns=names)


def durations(labels: pd.Series) -> dict:
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


def episodes(labels: pd.Series) -> dict:
    return {k: len(v) for k, v in durations(labels).items()}


# ----------------------------------------------------------------------------- the state finder
def oos_separation(labels: pd.Series, target: pd.Series, folds: int = 4) -> float:
    """Out-of-sample R^2 of the forward target on state dummies: state means from the training
    part of each time-ordered fold, applied to the test part. Negative means worse than the mean."""
    df = pd.concat([labels.rename("s"), target.rename("y")], axis=1).dropna()
    n = len(df)
    cuts = np.linspace(int(0.4 * n), n, folds + 1).astype(int)
    sse = sst = 0.0
    for a, b in zip(cuts[:-1], cuts[1:]):
        tr, te = df.iloc[:a], df.iloc[a:b]
        m, m_all = tr.groupby("s")["y"].mean(), tr["y"].mean()
        pred = te["s"].map(m).fillna(m_all)
        sse += float(((te["y"] - pred) ** 2).sum())
        sst += float(((te["y"] - m_all) ** 2).sum())
    return 1 - sse / sst if sst > 0 else np.nan


def apply_spec(raw_score: pd.Series, spec: dict) -> pd.Series:
    """Labels from a raw (unsmoothed) composite and a spec returned by calibrate_states."""
    sc = ewma(raw_score, spec["halflife"])
    lv = level_labels(sc, spec["b1"], spec["b2"], spec["delta"])
    if spec["partition"] == "3":
        return two_axis_labels(lv, pd.Series("flat", index=sc.index), PARTITION3)
    d = direction_labels(direction_score(sc, spec["dir_window"]), spec["d_down"], spec["d_up"], spec["delta_d"])
    return two_axis_labels(lv, d, PARTITION5)


def calibrate_states(raw_score: pd.Series, target: pd.Series, halflives=(2, 3, 5), deltas=(0, 2, 3, 5),
                     dir_ks=(0.5, 0.75, 1.0), dir_window: int = 5, folds: int = 4, min_episodes: int = 5,
                     max_switches_per_year: float = 30.0, r2_tolerance: float = 0.015) -> dict:
    """Estimate boundaries, direction thresholds, smoothing and hysteresis; compare 3 vs 5 states.

    Call this on a training window only. The returned spec is applied forward with `apply_spec`.
    `target` is a forward-looking series aligned to the decision date (e.g. realised vol over the
    next 21 days); it is used only to score candidates, never to label.
    """
    years = max(1e-9, len(raw_score) / 252)
    report = []
    for hl in halflives:
        sc = ewma(raw_score, hl)
        b1, b2 = estimate_breaks(sc, target)
        ds = direction_score(sc, dir_window)
        sd = float(ds.std())
        for delta in deltas:
            lv = level_labels(sc, b1, b2, delta)
            l3 = two_axis_labels(lv, pd.Series("flat", index=sc.index), PARTITION3)
            cands = [("3", 0.0, 0.0, l3)]
            for k in dir_ks:
                d_up, delta_d = k * sd, 0.2 * k * sd
                d = direction_labels(ds, -d_up, d_up, delta_d)
                cands.append(("5", k, delta_d, two_axis_labels(lv, d, PARTITION5)))
            for part, k, delta_d, lab in cands:
                ep = episodes(lab)
                report.append(dict(halflife=hl, delta=delta, partition=part, dir_k=k, b1=b1, b2=b2, d_up=k * sd, d_down=-k * sd,
                                   delta_d=delta_d, dir_window=dir_window, oos_r2=oos_separation(lab, target, folds),
                                   switches=switches(lab), switches_per_year=switches(lab) / years,
                                   min_episodes=min(ep.values()), episodes=ep))
    rep = pd.DataFrame(report)
    ok = rep[(rep.min_episodes >= min_episodes) & (rep.switches_per_year <= max_switches_per_year)]
    if ok.empty:
        ok = rep[rep.min_episodes >= min_episodes]

    def pick(df):
        if df.empty:
            return None
        top = df.oos_r2.max()
        return dict(df[df.oos_r2 >= top - r2_tolerance].sort_values("switches").iloc[0])  # near-best, fewest switches

    best3, best5 = pick(ok[ok.partition == "3"]), pick(ok[ok.partition == "5"])
    spec = best5 if (best5 and (not best3 or best5["oos_r2"] >= best3["oos_r2"])) else best3
    return {"spec": spec, "spec3": best3, "spec5": best5, "report": rep,
            "best3_r2": best3["oos_r2"] if best3 else np.nan, "best5_r2": best5["oos_r2"] if best5 else np.nan}
