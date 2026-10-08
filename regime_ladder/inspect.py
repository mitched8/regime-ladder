"""Every derived input, written out so it can be looked at on its own.

`dump(market, ...)` runs each component of the pipeline from a market frame alone — no backtester
needed — and writes one file per quantity: the state features and composite, the finder's full
candidate table and the chosen spec, labels, episodes, durations and the transition matrix, tags
and their cells, the characteristics vector and state profiles, sub-state discovery per state, the
shock components, the leading features, their screen, the per-fold tilt fits, the pressure index
and the fans. A README in the output folder names, for each file, the function that produced it, so
the same quantity can be recomputed in a notebook with one call. The trade-day table is optional;
with it, the forward-P&L target is used where the config asks for it and the ladder is included.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import discover, evaluate, explore, features, gates, labels, ladder, leading, profile, schema, shock, tags, transitions

FILES = []  # (filename, produced by, what it is) — filled as dump runs, written to README.md


def _w(out: Path, name: str, obj, by: str, what: str, **kw):
    p = out / name
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        obj.to_csv(p, **kw)
    else:
        p.write_text(json.dumps(obj, indent=2, default=lambda v: float(v) if isinstance(v, (np.floating, np.integer)) else str(v)))
    FILES[:] = [f for f in FILES if f[0] != name] + [(name, by, what)]


def dump(market: pd.DataFrame, cfg: dict, pcfg: dict, lcfg: dict, gates_cfg: dict, pair: str, out: str | Path,
         td: pd.DataFrame | None = None, target: pd.Series | None = None, events=None, target_name: str | None = None,
         targets: dict | None = None) -> dict:
    out = Path(out); out.mkdir(parents=True, exist_ok=True); FILES.clear()
    c = cfg["labeller"]

    # 1 · state features and the composite
    F = features.build(market)
    _w(out, "features.csv", F, "features.build(market)", "every registered state feature, trailing percentiles, point-in-time")
    comp = labels.composite(F[c["features"]])
    sc = labels.ewma(comp, c["halflife"]); ds = labels.direction_score(sc)
    # 2 · the finder
    if target is None:
        target = market["rv_1m"].shift(-21)
        tname = "forward 21d realised vol (no trade table given)"
    else:
        tname = target_name or "forward 5d archetype P&L from the trade table"
    lvl = market[c.get("level_target", "atm_1m")].shift(-int(c.get("level_horizon", 5)))
    cal = labels.calibrate_states(comp, target.reindex(comp.index), level_target=lvl, min_episodes=c["min_episodes"], max_switches_per_year=c["max_switches_per_year"])
    _w(out, "finder_report.csv", cal["report"].drop(columns=["episodes"]), "labels.calibrate_states(...)['report']",
       f"every candidate (half-life × hysteresis × direction k × partition) with OOS R² on the target [{tname}], switches, episodes, extreme rule")
    want = str(c.get("states", "auto")); spec = (cal["specs"].get(want) if want != "auto" else None) or cal["spec"]
    _w(out, "finder_spec.json", {k: v for k, v in spec.items() if k != "episodes"}, "labels.calibrate_states(...)['specs'][states]", "the chosen spec: bounds, half-life, hysteresis, direction thresholds, partition, extreme rule")
    _w(out, "finder_r2.json", {k: (None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)) for k, v in cal["r2"].items()}, "labels.calibrate_states(...)['r2']", "best OOS R² per partition 3 / 6 / 6x")
    # 3 · labels, episodes, durations, matrix
    lab = labels.apply_spec(comp, spec)
    sc = labels.ewma(comp, spec["halflife"]); ds = labels.direction_score(sc, spec["dir_window"])
    lv = labels.level_labels(sc, spec["bounds"], spec["delta"]); dr = labels.direction_labels(ds, spec["d_down"], spec["d_up"], spec["delta_d"])
    comp_frame = pd.DataFrame({"composite": comp, "smoothed": sc, "direction_score": ds, "level": lv, "direction": dr, "regime": lab})
    _w(out, "composite.csv", comp_frame,
       "labels.composite / ewma / direction_score / level_labels / direction_labels / apply_spec", "the labeller step by step: raw composite, EWMA, slope, level band, direction band, named state")
    _w(out, "finder_inputs.csv", pd.DataFrame({"target": target.reindex(comp.index), "level_target": lvl.reindex(comp.index)}),
       "the finder's two targets", f"what the finder scored against, per day: `target` [{tname}] for partition / direction / hysteresis, `level_target` (forward {c.get('level_target', 'atm_1m')}, {int(c.get('level_horizon', 5))}d ahead) for the level bounds")
    names = labels.state_order(lab)
    P = labels.transition_matrix(lab, names=names, prior_strength=lcfg["transitions"]["prior_strength"])
    _w(out, "transition_matrix.csv", P, "labels.transition_matrix(labels, prior_strength)", "daily matrix with the sticky prior; rows = from, columns = to")
    _w(out, "transition_counts.csv", labels.transition_matrix(lab, names=names, prior_strength=0.0), "labels.transition_matrix(labels, prior_strength=0)", "the raw empirical matrix, no prior")
    _w(out, "durations.csv", transitions.implied_vs_observed_duration(P, lab), "transitions.implied_vs_observed_duration(P, labels)", "1/(1-p_ii) against the observed mean run, with episode counts")
    _w(out, "episodes.json", labels.episodes(lab), "labels.episodes(labels)", "episodes per state")
    # 4 · tags
    tcfg = cfg.get("tags", {})
    if events:
        tcfg = {**tcfg, "events": list(events)}
    T = tags.build(market, tcfg, pair)
    if len(T.columns):
        _w(out, "tags.csv", T, "tags.build(market, cfg['tags'], pair)", "every configured tag per day")
        cells = {}
        for col in T.columns:
            _, info = tags.tag_split(lab, T[col], tcfg.get("min_episodes", 5)); cells[col] = info
        _w(out, "tag_cells.json", cells, "tags.tag_split(labels, tag)", "episodes and days per state·tag cell and whether it is kept")
    # 5 · characteristics, profiles, discovery
    ch = profile.characteristics(market, pcfg)
    _w(out, "characteristics.csv", ch, "profile.characteristics(market, cfg)", "the daily characteristics vector: own-pair, cross-pair, cross-asset")
    prof = profile.state_profile(ch, lab)
    _w(out, "state_profile.csv", prof, "profile.state_profile(characteristics, labels)", "per state: mean, standardised difference vs all days, for every characteristic")
    _w(out, "today_placement.csv", profile.today_placement(ch, lab), "profile.today_placement(characteristics, labels)", "where today sits inside its state's history, per characteristic")
    disc = {}
    for s in sorted(set(lab)):
        r = discover.discover(ch, lab, s, kmax=pcfg["discovery"]["kmax"], min_episodes=pcfg["discovery"]["min_episodes"], stability=pcfg["discovery"]["stability"])
        disc[s] = {"k": r["k"], "bic": r["bic"], "stability": r["stability"], "episodes": r["episodes"], "names": r["names"]}
        if r["k"] > 1:
            _w(out, f"substates_{s}.csv", r["labels"].rename("substate"), f"discover.discover(characteristics, labels, '{s}')['labels']", f"sub-state per day inside {s}")
    _w(out, "discovery.json", disc, "discover.discover(characteristics, labels, state)", "per state: k, BIC by k, stability by k, episodes per cluster, auto-names")
    # 6 · shocks
    S = shock.shock_score(market, cfg=lcfg["shock"]); S["har_vol_forecast"] = shock.har_series(market)
    _w(out, "shock.csv", S, "shock.shock_score(market, cfg) + shock.har_series(market)", "surprise, CUSUM pressure, BOCD P(short run), recent |surprise|, score, HAR forecast")
    _w(out, "shock_lead_profile.csv", shock.lead_profile(S["score"], lab), "shock.lead_profile(score, labels)", "mean score before escalations vs before de-escalations vs all days", index=False)
    # 7 · leading features, screen, tilt fits, pressure, fans
    lc = lcfg["leading"]
    L = leading.build(market, names=list(lc["features"]), events=events or lc.get("events") or None, **lc["features"])
    _w(out, "leading_features.csv", L, "leading.build(market, names, events, **params)", "every leading feature per day (0..100); stored_energy = gap_pct × complacency / 100")
    gcfg = gates_cfg["phase4_leading"]
    ys = {k: v.reindex(comp.index) for k, v in (targets or {}).items()} or target.reindex(comp.index)
    hd = lc.get("diag_horizon"); diag = {a: f"{a}_h{hd}" for a in (targets or {}) if hd and f"{a}_h{hd}" in (targets or {})}
    kw = dict(k=int(lc.get("k", 5)), ks=tuple(lc.get("ks", (10, 21))), feature_targets=lc.get("feature_targets") or {},
              interact=tuple(lc.get("interact", ("carry", "settling"))) or None, diag_targets=diag)
    scr = leading.screen(L, lab, comp, ys, gcfg, **kw)
    lagged = leading.screen(L, lab, comp, ys, gcfg, lag=int(lc.get("lag_check", 1)), holdout_frac=0.0, n_perm=0, **kw)
    scr["delta_r2_lagged"] = scr["feature"].map(lagged.set_index("feature")["delta_r2"])
    _w(out, "leading_screen.csv", scr, "leading.screen(L, labels, composite, targets, gates['phase4_leading'], k, ks, feature_targets)",
       "one row per candidate: outcome test (and its low-band and lagged variants), transition test at the declared k and the diagnostic horizons, hold-out confirmation, retain", index=False)
    lf = pd.DataFrame([{"feature": c, **leading.lift(L[c], lab, **(lc.get("lift") or {}))} for c in L.columns])
    _w(out, "leading_lift.csv", lf, "leading.lift(L[c], labels, horizon, quantile)", "event-study form: P(move into the high band within the horizon | feature in its top decile) vs the base rate, with a block-bootstrap interval", index=False)
    cz = (comp - comp.mean()) / comp.std()
    fits = []
    for col in L.columns:
        z = (L[col] - L[col].mean()) / L[col].std()
        if not np.isfinite(z.std()) or z.std() == 0:
            continue
        g = transitions.oos_gain(lab, z, folds=4, k=5, base=cz, prior_strength=lcfg["transitions"]["prior_strength"])
        g.insert(0, "feature", col); fits.append(g)
    if fits:
        _w(out, "tilt_fits.csv", pd.concat(fits), "transitions.oos_gain(labels, psi, k=5, base=composite_z)", "per feature and fold: fitted β (feature) and γ (composite baseline), OOS k-step gain per observation", index=False)
    kept = list(scr.loc[scr["retain"], "feature"])
    use = {k: 1.0 for k in kept} or {col: 1.0 for col in L.columns}
    psi = leading.pressure(L, use, zscore_window=lc.get("pressure_zscore_window"))
    _w(out, "pressure.csv", psi, "leading.pressure(L, weights)", f"pressure index from {'the retained features ' + str(kept) if kept else 'ALL candidates (none retained — illustrative only)'}")
    pg = None
    if kept:
        pg = leading.pressure_gain(psi, lab, comp, prior_strength=lcfg["transitions"]["prior_strength"])
    _w(out, "gate_phase4.json", gates.gate_phase4(scr, pg, gates_cfg), "gates.gate_phase4(screen, oos_gain(pressure).attrs)", "Gate 4: retained features and the pressure index's OOS transition gain")
    fit = transitions.fit_tilt(lab, psi, P, k=5, base=cz)
    _w(out, "tilt_pressure.json", {k: v for k, v in fit.items()}, "transitions.fit_tilt(labels, pressure, P, k=5, base=composite_z)", "β and γ for the pressure index on the whole history, with log-likelihoods")
    today = lab.iloc[-1]; H = lcfg["transitions"]["horizon"]
    f0 = transitions.fan(P, pd.Series({today: 1.0}), psi_path=np.zeros(H))
    path = transitions.covariate_path(float(psi.iloc[-1]) if np.isfinite(psi.iloc[-1]) else 0.0, H, lcfg["transitions"]["pressure_policy"], lcfg["transitions"]["pressure_halflife"])
    f1 = transitions.fan(P, pd.Series({today: 1.0}), psi_path=path, beta=fit["beta"])
    _w(out, "fan_constant.csv", f0, "transitions.fan(P, pi0, psi_path=zeros)", f"P(state at h) from today's state ({today}) under the constant matrix")
    _w(out, "fan_tilted.csv", f1, "transitions.fan(P, pi0, psi_path, beta)", f"the same under the pressure tilt with the declared {lcfg['transitions']['pressure_policy']} path")
    # 8 · the ladder, if trades were given
    if td is not None:
        frame = schema.labels_frame(pair, lab.index, lab.values)
        cum = ladder.attach_entry_labels(ladder.cumulative(td, cfg["horizons"]), frame)
        lad = ladder.shrink(ladder.ladder(cum, frame, alpha=cfg["alpha"], ci=cfg["ci"], n_boot=cfg.get("n_boot", 1000)), kappa=cfg["kappa"])
        _w(out, "ladder.csv", lad, "ladder.ladder(cum, labels) -> ladder.shrink", "the entry-conditional ladder, every group × state × h", index=False)
        _w(out, "increments.csv", ladder.increments(lad), "ladder.increments(lad)", "earn per day between consecutive horizons", index=False)
        g2 = gates.gate_phase2(lad, gates_cfg); g2["cells"] = {"|".join(map(str, k)): v for k, v in g2["cells"].items()}
        _w(out, "gate_phase2.json", g2, "gates.gate_phase2(lad)", "Gate 2: separation between states at h <= max_h")
        wf = evaluate.walk_forward(cum, n_splits=cfg["eval"]["n_splits"], kappa=cfg["eval"].get("kappa", 0.0))
        _w(out, "walk_forward.csv", wf, "evaluate.walk_forward(cum)", "out-of-sample state ladder vs unconditional, per group and h", index=False)
        _w(out, "gate_phase3.json", gates.gate_phase3(wf, gates_cfg), "gates.gate_phase3(walk_forward)", "Gate 3: out-of-sample improvement, rank IC, calibration slope")
    # 9 · the explorer's tables: per-entry normalised earn, and every statistic a trader can condition on
    ent = explore.entries(td, horizons=tuple(h for h in cfg["horizons"] if h <= 20) or explore.HORIZONS) if td is not None else None
    if ent is not None:
        _w(out, "entries.csv", ent, "explore.entries(td)", f"one row per entry date × horizon per group: cumulative earn of a fresh unit, total and by component, normalised by {ent.attrs['normalised_by']} at inception", index=False)
    stats, sdefs = explore.statistics(market, F, comp_frame, L, S, ch, ent, cfg=cfg.get("explore"))
    _w(out, "statistics.csv", stats, "explore.statistics(market, features, composite, leading, shock, characteristics, entries)", "every daily statistic the explorer can condition on (point-in-time); definitions in statistics_defs.json")
    _w(out, "statistics_defs.json", {**sdefs, "normalised_by": ent.attrs["normalised_by"] if ent is not None else None, "components": ent.attrs["components"] if ent is not None else []},
       "explore.statistics(...)[1]", "one-line definition and group per statistic")
    if cfg.get("filters"):
        FT = explore.filters_to_tags(stats, cfg["filters"])
        _w(out, "filter_tags.csv", FT, "explore.filters_to_tags(statistics, cfg['filters'])", "saved explorer filters as tags ('on' / 'none'), one column per filter")
        fcells = {}
        for col in FT.columns:
            _, info = tags.tag_split(lab, FT[col], cfg.get("tags", {}).get("min_episodes", 5)); fcells[col] = info
        prior = json.loads((out / "tag_cells.json").read_text()) if (out / "tag_cells.json").exists() else {}
        _w(out, "tag_cells.json", {**prior, **fcells}, "tags.tag_split(labels, tag)", "episodes and days per state·tag cell (configured tags and saved filters) and whether it is kept")
    _w(out, "meta.json", {"pair": pair, "first_date": str(market.index[0].date()), "last_date": str(market.index[-1].date()), "days": len(market),
                          "finder_target": tname, "leading_target": tname, "trade_table": td is not None, "today_state": lab.iloc[-1]},
       "inspect.dump", "what this dump was run on")
    # README
    lines = ["# Inspection dump", "", f"Pair {pair}. Every file is one derived quantity, produced by the function named; call it on the same market frame to reproduce it in a notebook.", "",
             "| file | produced by | what it is |", "|---|---|---|"] + [f"| `{n}` | `{b}` | {w} |" for n, b, w in FILES]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    return {"files": [n for n, _, _ in FILES], "spec": spec, "today": today, "retained": kept}
