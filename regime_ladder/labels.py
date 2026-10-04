"""Regime labelling: level x direction -> named states, with calibration (the state finder).

Level: a composite stress score, smoothed, cut at k ordered boundaries with hysteresis ->
low / mid / high (three bands) or low / mid / high / extreme (four bands).
Direction: the smoothed score's slope, cut at two thresholds with hysteresis -> down / flat / up.
A partition table maps (level, direction) — and where it matters the previous state — to a named
state. Six named states: carry, rising, agitated, stressed, normalising, settling. The optional
fourth band `extreme` is the tail; it is kept as a state only if it has enough episodes to
condition on, otherwise it is merged into stressed for the ladder and surfaced as a flag.

`calibrate_states` is the state finder: on a training window it step-fits the level boundaries on
a forward vol level, sets the extreme bound as a tail quantile, sets direction thresholds from the
slope distribution, chooses smoothing and hysteresis by out-of-sample separation of forward P&L
under a switch budget, and compares partitions: "3" (level only), "6" (level x direction, three
bands), "6x" (four bands, extreme kept or merged by the episode rule). It returns a spec and a
report; the owner decides.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LEVELS3 = ("low", "mid", "high")
LEVELS4 = ("low", "mid", "high", "extreme")
DIRS = ("down", "flat", "up")
STATES6 = ("carry", "rising", "agitated", "stressed", "normalising", "settling")
STATES3 = ("carry", "transition", "crisis")
EXTREME = "extreme"
ALL_STATES = STATES6 + (EXTREME,)

# (level, direction) -> state, or a dict keyed by previous state ("_" = default). "keep" = stay.
PARTITION6 = {
    ("low", "flat"): "carry",
    ("low", "up"): "rising",
    ("low", "down"): {"carry": "carry", "_": "settling"},                          # carry drifting down is still carry
    ("mid", "flat"): "agitated",                                                    # elevated, choppy, going nowhere: a state, not a transit lounge
    ("mid", "up"): {"stressed": "stressed", "normalising": "stressed", "extreme": "stressed", "_": "rising"},
    ("mid", "down"): {"stressed": "normalising", "normalising": "normalising", "extreme": "normalising", "carry": "keep", "_": "settling"},
    ("high", "up"): "stressed", ("high", "flat"): "stressed", ("high", "down"): "normalising",
    ("extreme", "up"): EXTREME, ("extreme", "flat"): EXTREME, ("extreme", "down"): EXTREME,
}
PARTITION3 = {(lv, d): {"low": "carry", "mid": "transition", "high": "crisis", "extreme": "crisis"}[lv] for lv in LEVELS4 for d in DIRS}


# ----------------------------------------------------------------------------- primitives
def ewma(s: pd.Series, halflife: float) -> pd.Series:
    return s.ewm(halflife=halflife, adjust=False).mean()


def composite(features: pd.DataFrame, weights: dict | None = None) -> pd.Series:
    w = pd.Series(weights) if weights else pd.Series(1.0, index=features.columns)
    w = w.reindex(features.columns).fillna(0.0)
    return (features * (w / w.sum())).sum(axis=1, min_count=1)


def direction_score(score: pd.Series, window: int = 5) -> pd.Series:
    return (score - score.shift(window)) / window


def ordered_labels(x: pd.Series, bounds, delta: float, names, start: int = 0) -> pd.Series:
    """Hysteresis labeller onto len(bounds)+1 ordered classes; starts in class `start` (NaNs keep the class)."""
    bounds = list(bounds)
    out, state = [], start
    for v in x.values:
        if not np.isnan(v):
            while state < len(bounds) and v > bounds[state] + delta:
                state += 1
            while state > 0 and v < bounds[state - 1] - delta:
                state -= 1
        out.append(names[state])
    return pd.Series(out, index=x.index)


def level_labels(score, bounds, delta=0.0):
    bounds = list(bounds)
    return ordered_labels(score, bounds, delta, LEVELS4 if len(bounds) == 3 else LEVELS3)


def direction_labels(dscore, d_down, d_up, delta=0.0):
    out, state = [], 1
    for v in dscore.values:
        if not np.isnan(v):
            if state == 0 and v > d_down + delta:
                state = 1
            if state == 1:
                if v > d_up + delta:
                    state = 2
                elif v < d_down - delta:
                    state = 0
            elif state == 2 and v < d_up - delta:
                state = 1
        out.append(DIRS[state])
    return pd.Series(out, index=dscore.index)


def threshold_labels(score: pd.Series, b1: float, b2: float, delta: float = 0.0, names=STATES3) -> pd.Series:
    """Level-only three-state labels (the benchmark design)."""
    return ordered_labels(score, [b1, b2], delta, names).rename("regime")


def two_axis_labels(level: pd.Series, direction: pd.Series, partition: dict = PARTITION6, start: str = "carry") -> pd.Series:
    out, prev = [], start
    for lv, d in zip(level.values, direction.values):
        s = partition.get((lv, d), "keep")
        if isinstance(s, dict):
            s = s.get(prev, s.get("_", "keep"))
        prev = prev if s == "keep" else s
        out.append(prev)
    return pd.Series(out, index=level.index, name="regime")


def merge_rare(labels: pd.Series, rare: str = EXTREME, into: str = "stressed", min_episodes: int = 5) -> tuple[pd.Series, dict]:
    """Merge a rare state into its parent for the ladder if it has too few episodes; report what happened."""
    ep = episodes(labels)
    n = ep.get(rare, 0)
    if 0 < n < min_episodes:
        return labels.replace({rare: into}), {"merged": True, "rare_episodes": n, "rare_days": int((labels == rare).sum())}
    return labels, {"merged": False, "rare_episodes": n, "rare_days": int((labels == rare).sum())}


def estimate_breaks(score: pd.Series, target: pd.Series, n_breaks: int = 2, min_share: float = 0.08,
                    grid_n: int = 40) -> tuple:
    """Step fit: the n_breaks boundaries on `score` whose bands best explain `target` by band means
    (SSE-minimising piecewise-CONSTANT fit, every band holding at least `min_share` of the days).

    This is the estimator consistent with how bands are used downstream — the ladder conditions on
    band means — so it places boundaries at the steps between bands. A piecewise-linear (kink) fit
    does not: it puts its breaks at the ends of a ramp, not in the middle of it.
    """
    df = pd.concat([score.rename("s"), target.rename("y")], axis=1).dropna().sort_values("s")
    s, y = df["s"].values, df["y"].values
    n = len(s)
    if n < 50:
        return None
    cs, cs2 = np.concatenate([[0.0], np.cumsum(y)]), np.concatenate([[0.0], np.cumsum(y ** 2)])
    grid = np.quantile(s, np.linspace(min_share, 1 - min_share, grid_n))
    cuts = np.searchsorted(s, grid)
    min_n = int(min_share * n)

    def sse(a, b):
        tot = cs[b] - cs[a]
        return (cs2[b] - cs2[a]) - tot * tot / (b - a)

    best, best_sse = None, np.inf
    for combo in _combinations(range(len(cuts)), n_breaks):
        pts = [0] + [int(cuts[c]) for c in combo] + [n]
        if min(np.diff(pts)) < min_n:
            continue
        v = sum(sse(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
        if v < best_sse:
            best, best_sse = tuple(float(grid[c]) for c in combo), v
    return best


def _combinations(seq, k):
    from itertools import combinations
    return combinations(seq, k)


def extreme_bound(score: pd.Series, q: float = 0.975) -> float:
    """The extreme band is a tail by definition, not a step: its lower bound is a training-window quantile
    of the smoothed score. Whether it survives as a state is decided by the episode rule (`merge_rare`)."""
    return float(score.quantile(q))


def switches(labels: pd.Series) -> int:
    return int((labels.values[1:] != labels.values[:-1]).sum())


def state_order(labels) -> list:
    """The states present, in canonical order (named states, then the three-state names, then anything else)."""
    present = set(labels)
    return list(dict.fromkeys(n for n in ALL_STATES + STATES3 if n in present)) + sorted(present - set(ALL_STATES) - set(STATES3))


def transition_matrix(labels: pd.Series, names=None, prior_strength: float = 0.0, stickiness: float = 0.95) -> pd.DataFrame:
    names = list(names) if names is not None else state_order(labels)
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
    """Out-of-sample R^2 of the forward target on state dummies: means from the training part of each
    time-ordered fold applied to its test part. Negative means worse than the unconditional mean."""
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
    """Labels from a raw (unsmoothed) composite and a spec from calibrate_states. Applies the merge rule."""
    sc = ewma(raw_score, spec["halflife"])
    lv = level_labels(sc, spec["bounds"], spec["delta"])
    if spec["partition"] == "3":
        return two_axis_labels(lv, pd.Series("flat", index=sc.index), PARTITION3)
    d = direction_labels(direction_score(sc, spec["dir_window"]), spec["d_down"], spec["d_up"], spec["delta_d"])
    lab = two_axis_labels(lv, d, PARTITION6)
    if spec.get("extreme_merged"):
        lab = lab.replace({EXTREME: "stressed"})
    return lab


def calibrate_states(raw_score: pd.Series, target: pd.Series, level_target: pd.Series | None = None,
                     halflives=(2, 3, 5), deltas=(0, 2, 3, 5), dir_ks=(0.5, 0.75, 1.0), dir_window: int = 5,
                     folds: int = 4, min_episodes: int = 5, max_switches_per_year: float = 30.0,
                     r2_tolerance: float = 0.015, try_extreme: bool = True, extreme_quantile: float = 0.975) -> dict:
    """Estimate boundaries, direction thresholds, smoothing and hysteresis; compare partitions 3 / 6 / 6x.

    Two targets, two jobs. LEVEL boundaries are a vol-surface fact — where the surface clusters — so
    they are step-fitted on `level_target` (default: a forward vol level such as ATM a week ahead,
    available on the whole price history, not just the backtested window). PARTITION, direction
    thresholds, smoothing and hysteresis are a P&L fact, so they are chosen by out-of-sample
    separation of `target` (forward archetype P&L). Fitting level bounds on P&L fails where two bands
    earn alike for opposite reasons (stressed earns, normalising loses: the high band looks like the
    mid band on P&L and the boundary is lost). If `level_target` is None, `target` is used for both.

    Call on a training window only; apply forward with `apply_spec`. Targets are forward outcomes
    aligned to the decision date, used only to score candidates, never to label.
    """
    years = max(1e-9, len(raw_score) / 252)
    level_target = target if level_target is None else level_target
    report = []
    for hl in halflives:
        sc = ewma(raw_score, hl)
        b2 = estimate_breaks(sc, level_target, 2)
        b3 = (b2 + (extreme_bound(sc, extreme_quantile),)) if (try_extreme and b2) else None
        ds = direction_score(sc, dir_window)
        sd = float(ds.std())
        for delta in deltas:
            lv3 = level_labels(sc, b2, delta)
            cands = [("3", 0.0, 0.0, b2, two_axis_labels(lv3, pd.Series("flat", index=sc.index), PARTITION3), {})]
            lv4 = level_labels(sc, b3, delta) if b3 else None
            for k in dir_ks:
                d_up, delta_d = k * sd, 0.2 * k * sd
                d = direction_labels(ds, -d_up, d_up, delta_d)
                cands.append(("6", k, delta_d, b2, two_axis_labels(lv3, d, PARTITION6), {}))
                if lv4 is not None:
                    lab, info = merge_rare(two_axis_labels(lv4, d, PARTITION6), min_episodes=min_episodes)
                    cands.append(("6x", k, delta_d, b3, lab, info))
            for part, k, delta_d, bounds, lab, info in cands:
                ep = episodes(lab)
                report.append(dict(halflife=hl, delta=delta, partition=part, dir_k=k, bounds=tuple(bounds), b1=bounds[0], b2=bounds[1],
                                   d_up=k * sd, d_down=-k * sd, delta_d=delta_d, dir_window=dir_window,
                                   oos_r2=oos_separation(lab, target, folds), switches=switches(lab), switches_per_year=switches(lab) / years,
                                   min_episodes=min(ep.values()), episodes=ep, extreme_merged=info.get("merged", False),
                                   extreme_episodes=info.get("rare_episodes", 0), extreme_days=info.get("rare_days", 0)))
    rep = pd.DataFrame(report)
    ok = rep[(rep.min_episodes >= min_episodes) & (rep.switches_per_year <= max_switches_per_year)]
    if ok.empty:
        ok = rep[rep.min_episodes >= min_episodes]

    def pick(df):
        if df.empty:
            return None
        top = df.oos_r2.max()
        return dict(df[df.oos_r2 >= top - r2_tolerance].sort_values("switches").iloc[0])

    specs = {p: pick(ok[ok.partition == p]) for p in ("3", "6", "6x")}
    cands = [s for s in specs.values() if s]
    spec = max(cands, key=lambda s: s["oos_r2"]) if cands else None
    return {"spec": spec, "specs": specs, "report": rep,
            "r2": {p: (s["oos_r2"] if s else np.nan) for p, s in specs.items()}}
