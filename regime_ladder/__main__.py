"""CLI.  python -m regime_ladder {synth,check,labels,ladder,evaluate,validate,shock,transitions,leading,inspect,pack,demo} ...

`pack` turns an `inspect` folder (or the trade table) into one markdown review packet per stage for a reviewer without the code.

`inspect` writes every derived input — features, composite, finder table, labels, matrix, tags, characteristics,
profiles, discovery, shocks, leading features, tilt fits, pressure, fans — one file each, with a README naming
the function behind each one, so any piece can be looked at and recomputed on its own.

`demo` runs the whole pipeline on synthetic data — calibrated six-state labels, ladder, profile,
sub-state discovery, gates — and writes a card. It is the smoke test for a fresh environment.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import checks, discover, evaluate, features, gates, inspect as inspect_, labels, ladder, leading, pack, profile, report, schema, shock, synth, tags, transitions, view


def _cfg(path):
    return yaml.safe_load(Path(path).read_text())


def _forward_pnl_target(td: pd.DataFrame, archetype: str, h: int = 5) -> pd.Series:
    """Forward h-day cumulative P&L of fresh entries of one archetype, indexed by entry date."""
    cum = ladder.cumulative(td[td["archetype"] == archetype], horizons=(h,), include_expiry=False)
    return cum.groupby("entry_date")["cum_pnl"].mean()


def make_labels(market: pd.DataFrame, cfg: dict, pair: str, target: pd.Series | None = None):
    """Composite -> (calibrated) six-, six-plus-extreme- or three-state labels. Returns (label frame, series, spec).

    `cfg["labeller"]["states"]`: 6 (level x direction), "6x" (plus the extreme band, merged by the
    episode rule when rare), 3 (level only), or "auto" (whichever partition scores best on the target).
    """
    c = cfg["labeller"]
    f = features.build(market, names=c["features"])
    comp = labels.composite(f)
    if c.get("calibrate") and target is not None:
        lvl_col = c.get("level_target", "atm_1m")
        level_target = market[lvl_col].shift(-int(c.get("level_horizon", 5))).reindex(comp.index) if lvl_col in market else None
        cal = labels.calibrate_states(comp, target.reindex(comp.index), level_target=level_target,
                                      min_episodes=c["min_episodes"], max_switches_per_year=c["max_switches_per_year"])
        want = str(c.get("states", "auto"))
        spec = (cal["specs"].get(want) if want != "auto" else None) or cal["spec"]
    else:
        sc = labels.ewma(comp, c["halflife"])
        sd = float(labels.direction_score(sc).std())
        part = str(c.get("states", 6)); part = "6" if part == "auto" else part
        spec = dict(halflife=c["halflife"], bounds=tuple(c["bounds"]), delta=c["delta"], partition=part,
                    dir_window=5, d_up=c["dir_k"] * sd, d_down=-c["dir_k"] * sd, delta_d=0.2 * c["dir_k"] * sd,
                    extreme_merged=False)
    lab_s = labels.apply_spec(comp, spec)
    return schema.labels_frame(pair, lab_s.index, lab_s.values), lab_s, spec


def cmd_synth(a):
    m = synth.simulate_market(a.days, seed=a.seed, energy_beta=a.energy_beta)
    td = synth.simulate_trades(m, tenor_days=a.tenor, seed=a.seed + 1)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    m.to_parquet(out / "market.parquet"); td.to_parquet(out / "trade_days.parquet")
    synth.true_ladder(tenor_days=a.tenor).to_csv(out / "true_ladder.csv", index=False)
    print(f"wrote {len(td):,} trade-day rows and {len(m):,} market days to {out}")


def cmd_check(a):
    td = schema.coerce(pd.read_parquet(a.td))
    rep = checks.check_trade_days(td)
    if a.labels:
        rep.update(checks.check_label_coverage(td, pd.read_parquet(a.labels)))
    print(checks.format_report(rep))
    checks.assert_clean({k: v for k, v in rep.items() if not k.startswith("info_")})


def cmd_labels(a):
    cfg = _cfg(a.config)
    m = pd.read_parquet(a.market)
    target = None
    if a.td and cfg["labeller"].get("target") == "forward_pnl":
        target = _forward_pnl_target(schema.coerce(pd.read_parquet(a.td)), a.target_archetype)
    elif cfg["labeller"].get("calibrate"):
        target = m["rv_1m"].shift(-21)
    lab, lab_s, spec = make_labels(m, cfg, a.pair, target)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); lab.to_parquet(a.out)
    print(f"{len(lab)} labels, {labels.switches(lab_s)} switches, episodes {labels.episodes(lab_s)}")
    print("spec:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in spec.items() if k not in ("episodes",)})


def _run_ladder(td, lab, cfg):
    cum = ladder.attach_entry_labels(ladder.cumulative(td, cfg["horizons"]), lab)
    lad = ladder.shrink(ladder.ladder(cum, lab, alpha=cfg["alpha"], ci=cfg["ci"], n_boot=cfg.get("n_boot", 1000)), kappa=cfg["kappa"])
    return cum, lad, ladder.increments(lad), ladder.persistence_split(td, lab, cfg["horizons"])


def cmd_ladder(a):
    cfg = _cfg(a.config)
    td = schema.coerce(pd.read_parquet(a.td)); lab = pd.read_parquet(a.labels)
    cum, lad, inc, split = _run_ladder(td, lab, cfg)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lad.to_csv(out / "ladder.csv", index=False); inc.to_csv(out / "increments.csv", index=False); split.to_csv(out / "persistence_split.csv", index=False)
    for g in lad[ladder.GROUP].drop_duplicates().itertuples(index=False):
        report.plot_ladder(lad, tuple(g), out / f"ladder_{'_'.join(map(str, g))}.png")
        report.plot_increments(inc, tuple(g), out / f"increments_{'_'.join(map(str, g))}.png")
    print(lad[lad["h"] <= 5].to_string(index=False, float_format=lambda v: f"{v:+.2f}"))


def cmd_evaluate(a):
    cfg = _cfg(a.config)
    td = schema.coerce(pd.read_parquet(a.td)); lab = pd.read_parquet(a.labels)
    cum = ladder.attach_entry_labels(ladder.cumulative(td, cfg["horizons"]), lab)
    wf = evaluate.walk_forward(cum, n_splits=cfg["eval"]["n_splits"], kappa=cfg["eval"].get("kappa", 0.0))
    Path(a.out).mkdir(parents=True, exist_ok=True); wf.to_csv(Path(a.out) / "walk_forward.csv", index=False)
    g3 = gates.gate_phase3(wf, gates.load_gates(a.gates)); gates.write_result(g3, Path(a.out) / "gate_phase3.json")
    print(wf.to_string(index=False, float_format=lambda v: f"{v:+.3f}")); print("phase3 gate:", g3["pass"])


def cmd_validate(a):
    """Recover the analytic truth from synthetic data with TRUE labels: the test of the statistics themselves."""
    cfg = _cfg(a.config); truth = synth.true_ladder(horizons=cfg["horizons"]); rows = []
    for seed in range(a.seeds):
        m = synth.simulate_market(a.days, seed=seed); td = synth.simulate_trades(m, seed=seed + 100)
        lab = schema.labels_frame("EURUSD", m.index, m["state_true"].values)
        lad = ladder.ladder(ladder.attach_entry_labels(ladder.cumulative(td, cfg["horizons"]), lab), lab, ci=cfg["ci"], n_boot=cfg.get("n_boot", 1000))
        x = lad[lad["regime"] != "ALL"].merge(truth, on=["archetype", "regime", "h"]); x = x[x["n_eff"] >= 20]
        rows.append((seed, ((x["ci_lo"] <= x["true_ev"]) & (x["true_ev"] <= x["ci_hi"])).mean(), ((x["mean"] - x["true_ev"]).abs() / x["se"]).median()))
    r = pd.DataFrame(rows, columns=["seed", "coverage_90", "median_abs_err_over_se"])
    print(r.to_string(index=False, float_format=lambda v: f"{v:.3f}")); print(f"mean coverage {r.coverage_90.mean():.3f} (nominal 0.90)")


def _labels_series(path: str) -> pd.Series:
    lab = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, parse_dates=["date"])
    return lab.set_index("date")["regime"]


def cmd_shock(a):
    """Shock components and score for a market frame; lead profile against labels if given."""
    cfg = _cfg(a.leading_config)["shock"]
    m = pd.read_parquet(a.market)
    sc = shock.shock_score(m, cfg=cfg)
    sc["har_vol_forecast"] = shock.har_series(m)
    Path(a.out).mkdir(parents=True, exist_ok=True); sc.to_csv(Path(a.out) / "shock.csv")
    print(sc.dropna().tail(5).round(3).to_string())
    if a.labels:
        print(shock.lead_profile(sc["score"], _labels_series(a.labels)).round(2).to_string(index=False))


def cmd_transitions(a):
    """Constant matrix, duration check, and the fan from today's state — tilted by a pressure series if given."""
    tcfg = _cfg(a.leading_config)["transitions"]
    lab = _labels_series(a.labels)
    names = labels.state_order(lab)
    P = labels.transition_matrix(lab, names=names, prior_strength=tcfg["prior_strength"])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); P.to_csv(out / "transition_matrix.csv")
    print(P.round(3).to_string()); print(transitions.implied_vs_observed_duration(P, lab).round(1).to_string())
    pi0 = pd.Series({lab.iloc[-1]: 1.0})
    f0 = transitions.fan(P, pi0, psi_path=np.zeros(tcfg["horizon"]))
    print("constant matrix — expected days next", tcfg["horizon"], ":", transitions.expected_days(f0).round(1).to_dict())
    if a.pressure:
        psi = pd.read_csv(a.pressure, index_col=0, parse_dates=True).iloc[:, 0]
        g = transitions.oos_gain(lab, psi, folds=4, prior_strength=tcfg["prior_strength"])
        fit = transitions.fit_tilt(lab, psi, P)
        path = transitions.covariate_path(float(psi.reindex(lab.index).iloc[-1]), tcfg["horizon"], tcfg["pressure_policy"], tcfg["pressure_halflife"])
        f1 = transitions.fan(P, pi0, psi_path=path, beta=fit["beta"])
        hi = [x for x in ("stressed", "extreme", "crisis") if x in P.index]
        print(f"tilt beta {fit['beta']:+.2f} · OOS gain/transition {g.attrs['total_gain_per_transition']:+.4f} in {g.attrs['folds_positive']}/4 folds")
        print("P(" + "+".join(hi) + ") constant vs tilted:", transitions.p_state_at(f0, hi).round(3).to_dict(), transitions.p_state_at(f1, hi).round(3).to_dict())
        f1.to_csv(out / "fan_tilted.csv")
    f0.to_csv(out / "fan_constant.csv")


def _leading_targets(m: pd.DataFrame, lcfg: dict, td: pd.DataFrame | None) -> dict:
    """Named forward outcomes for the screen: one per archetype in `targets` from the trade table, or the
    surface fallback (realised minus implied over 21d) when there is no trade table."""
    if lcfg["target"] == "forward_pnl" and td is not None:
        archs = [a for a in lcfg.get("targets", [lcfg["target_archetype"]]) if a in set(td["archetype"])] or [lcfg["target_archetype"]]
        h, hd = int(lcfg.get("target_horizon", 5)), lcfg.get("diag_horizon")
        out = {a: _forward_pnl_target(td, a, h) for a in archs}
        if hd:
            out.update({f"{a}_h{hd}": _forward_pnl_target(td, a, int(hd)) for a in archs})
        return out
    return {"realised_minus_implied": (m["rv_1m"].shift(-21) - m["atm_1m"]).rename("realised_minus_implied")}


def _screen_kwargs(lcfg: dict, targets: dict | None = None) -> dict:
    hd = lcfg.get("diag_horizon")
    diag = {a: f"{a}_h{hd}" for a in (targets or {}) if hd and f"{a}_h{hd}" in (targets or {})}
    return dict(k=int(lcfg.get("k", 5)), ks=tuple(lcfg.get("ks", (10, 21))), feature_targets=lcfg.get("feature_targets") or {},
                interact=tuple(lcfg.get("interact", ("carry", "settling"))) or None, diag_targets=diag)


def cmd_leading(a):
    """Screen the leading features one at a time, build the pressure index from the retained ones, fit the tilt, evaluate Gate 4."""
    cfg = _cfg(a.config); lcfg_all = _cfg(a.leading_config); lcfg = lcfg_all["leading"]
    m = pd.read_parquet(a.market); lab = _labels_series(a.labels)
    comp = labels.composite(features.build(m, names=cfg["labeller"]["features"]))
    L = leading.build(m, names=list(lcfg["features"]), events=lcfg.get("events") or None, **lcfg["features"])
    td = schema.coerce(pd.read_parquet(a.td)) if a.td else None
    ys = _leading_targets(m, lcfg, td)
    gc = gates.load_gates(a.gates); kw = _screen_kwargs(lcfg, ys)
    sc = leading.screen(L, lab, comp, ys, gc["phase4_leading"], **kw)
    lagged = leading.screen(L, lab, comp, ys, gc["phase4_leading"], lag=int(lcfg.get("lag_check", 1)), holdout_frac=0.0, n_perm=0, **kw)
    sc["delta_r2_lagged"] = sc["feature"].map(lagged.set_index("feature")["delta_r2"])
    lf = pd.DataFrame([{"feature": c, **leading.lift(L[c], lab, **(lcfg.get("lift") or {}))} for c in L.columns])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); sc.to_csv(out / "leading_screen.csv", index=False); L.to_csv(out / "leading_features.csv"); lf.to_csv(out / "leading_lift.csv", index=False)
    print(sc.round(4).to_string(index=False)); print(lf.round(3).to_string(index=False))
    kept = [f for f in sc.loc[sc["retain"], "feature"]]
    pg = None
    if kept:
        psi = leading.pressure(L, lcfg.get("pressure_weights") or {k: 1.0 for k in kept}, zscore_window=lcfg.get("pressure_zscore_window"))
        psi.to_csv(out / "pressure.csv")
        pg = leading.pressure_gain(psi, lab, comp, prior_strength=lcfg_all["transitions"]["prior_strength"])
        print(f"pressure index from {kept}: OOS transition gain {pg['total_gain_per_transition']:+.4f} in {pg['folds_positive']}/4 folds")
    g4 = gates.gate_phase4(sc, pg, gc); gates.write_result(g4, out / "gate_phase4.json")
    print("phase4 (leading) gate:", g4["pass"], "· retained:", kept or "none")


def cmd_inspect(a):
    """Every derived input from a market frame (and optionally the trade table), one file each, plus a README."""
    cfg = _cfg(a.config); pcfg = _cfg(a.profile_config); lcfg = _cfg(a.leading_config); gc = gates.load_gates(a.gates)
    m = pd.read_parquet(a.market) if a.market.endswith(".parquet") else pd.read_csv(a.market, index_col=0, parse_dates=True)
    td = schema.coerce(pd.read_parquet(a.td)) if a.td else None
    target = _forward_pnl_target(td, cfg["labeller"].get("target_archetype", "straddle_atm")) if (td is not None and cfg["labeller"].get("target") == "forward_pnl") else None
    res = inspect_.dump(m, cfg, pcfg, lcfg, gc, a.pair, a.out, td=td, target=target, targets=_leading_targets(m, lcfg["leading"], td) if td is not None else None)
    print(f"{len(res['files'])} files in {a.out} · today {res['today']} · spec {res['spec']['partition']} bounds {tuple(round(b) for b in res['spec']['bounds'])} · retained leading {res['retained'] or 'none'}")
    print((Path(a.out) / "README.md").read_text())


def cmd_pack(a):
    """One self-describing markdown review packet per stage (templates/CHALLENGER_PROMPT.md is written against it)."""
    td = schema.coerce(pd.read_parquet(a.td)) if a.td else None
    m = None
    if a.market:
        m = pd.read_parquet(a.market) if a.market.endswith(".parquet") else pd.read_csv(a.market, index_col=0, parse_dates=True)
    if a.stage == "data" and td is None:
        raise SystemExit("--stage data needs --td")
    if a.stage != "data" and not a.src:
        raise SystemExit(f"--stage {a.stage} needs --src (an inspect folder)")
    cfgs = [a.config, a.gates, a.leading_config, a.profile_config, "configs/archetypes.yaml"]
    p = pack.build(a.stage, Path(a.out) / a.stage, src=a.src, td=td, market=m, label=a.label, config_paths=cfgs,
                   archetypes_cfg=pack.load_yaml("configs/archetypes.yaml"))
    print(f"wrote {p} ({len(p.read_text().splitlines())} lines, {p.stat().st_size / 1024:.0f} KB)")


def cmd_view(a):
    """One self-contained HTML page over one or more inspect folders: pivots over the ladder, one pair's series per tab."""
    p = view.build(a.src, a.out, label=a.label)
    print(f"wrote {p} ({p.stat().st_size / 1024:.0f} KB) · open it in a browser")


def cmd_demo(a):
    cfg = _cfg(a.config); pcfg = _cfg(a.profile_config)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    # 10 years of market for labelling and profiles; the last 5 carry trades (as in the real plan)
    m = synth.simulate_market(a.days, seed=a.seed, energy_beta=a.energy_beta)
    m_trade = m.iloc[-min(len(m), 1260):]
    td = synth.simulate_trades(m_trade, seed=a.seed + 1)
    rep = checks.check_trade_days(td); print(checks.format_report(rep)); checks.assert_clean(rep)
    target = _forward_pnl_target(td, "straddle_atm") if cfg["labeller"].get("target") == "forward_pnl" else m["rv_1m"].shift(-21)
    lab, lab_s, spec = make_labels(m, cfg, "EURUSD", target)
    acc = (lab_s.values == m["state_true"].values).mean()
    print(f"states: {spec['partition']} · agreement with true state {acc:.0%} · switches/yr {labels.switches(lab_s) / (len(m) / 252):.0f} · episodes {labels.episodes(lab_s)}")
    cum, lad, inc, split = _run_ladder(td, lab, cfg)
    wf = evaluate.walk_forward(cum, n_splits=cfg["eval"]["n_splits"], kappa=cfg["eval"].get("kappa", 0.0))
    # profile and discovery
    ch = profile.characteristics(m, pcfg); prof = profile.state_profile(ch, lab_s)
    today = lab_s.iloc[-1]
    disc = discover.discover(ch, lab_s, today, kmax=pcfg["discovery"]["kmax"], min_episodes=pcfg["discovery"]["min_episodes"], stability=pcfg["discovery"]["stability"])
    place = profile.today_placement(ch, lab_s)
    lines = [profile.describe(prof, today)]
    if disc["k"] > 1:
        sub = discover.assign(ch.iloc[-1], disc); lines.append(f"sub-state today: {disc['names'].get(sub, sub)} (k={disc['k']}, stability {disc['stability'][disc['k']]:.2f})")
    else:
        lines.append(f"no stable sub-states within {today} (stability {max(disc['stability'].values()) if disc['stability'] else float('nan'):.2f} < threshold)")
    top = place.loc[profile.distinguishing(prof, today, 3).index]
    lines += [f"{c}: today {r.today:+.2f}, {r.pct_within_state:.0%} percentile within {today}" for c, r in top.iterrows()]
    # tags: split today's state by spot-vol correlation sign where the cell has enough episodes
    tcfg = cfg.get("tags", {})
    tg = tags.build(m, tcfg, "EURUSD")
    if "corr_sign" in tg:
        split_lab, tinfo = tags.split_labels_frame(lab, tg["corr_sign"], tcfg.get("min_episodes", 5))
        kept = [k for k, v in tinfo["EURUSD"].items() if v["kept"]]
        lines.append(f"tags today: corr_sign={tg['corr_sign'].iloc[-1]} · split cells with enough episodes: {', '.join(kept) or 'none'}")
        if any(k.startswith(today + tags.SEP) for k in kept):
            cum_t = ladder.attach_entry_labels(ladder.cumulative(td, cfg["horizons"]), split_lab)
            lad_t = ladder.ladder(cum_t, split_lab, alpha=cfg["alpha"], ci="hac")
            g5 = lad_t[(lad_t["archetype"] == "rr_25d") & (lad_t["h"] == 5) & lad_t["regime"].str.startswith(today)]
            lines.append("rr_25d 5d EV by tag: " + " · ".join(f"{r.regime} {r['mean']:+.2f} (ep {int(r.episodes)})" for _, r in g5.iterrows()))
    # shock, transitions and leading features
    lcfg = _cfg(a.leading_config)
    sc = shock.shock_score(m, cfg=lcfg["shock"])
    lines.append(f"shock score today {sc['score'].iloc[-1]:.0f}/100 (CUSUM {sc['cusum_pressure'].iloc[-1]:.2f}, BOCD P(short run) {sc['bocd_p_short'].iloc[-1]:.2f})")
    names = labels.state_order(lab_s)
    P = labels.transition_matrix(lab_s, names=names, prior_strength=lcfg["transitions"]["prior_strength"])
    hi = [x for x in ("stressed", "extreme") if x in P.index]
    f0 = transitions.fan(P, pd.Series({today: 1.0}), psi_path=np.zeros(21))
    L = leading.build(m, names=list(lcfg["leading"]["features"]), **lcfg["leading"]["features"])
    ys = _leading_targets(m, lcfg["leading"], td)
    scr = leading.screen(L, lab_s, labels.composite(features.build(m, names=cfg["labeller"]["features"])), ys,
                         gates.load_gates(a.gates)["phase4_leading"], **_screen_kwargs(lcfg["leading"], ys))
    kept = list(scr.loc[scr["retain"], "feature"])
    p_const = transitions.p_state_at(f0, hi)
    if kept:
        psi = leading.pressure(L, {k: 1.0 for k in kept}, zscore_window=lcfg["leading"].get("pressure_zscore_window"))
        fit = transitions.fit_tilt(lab_s, psi, P)
        path = transitions.covariate_path(float(psi.iloc[-1]), 21, lcfg["transitions"]["pressure_policy"], lcfg["transitions"]["pressure_halflife"])
        p_tilt = transitions.p_state_at(transitions.fan(P, pd.Series({today: 1.0}), psi_path=path, beta=fit["beta"]), hi)
        lines.append(f"leading: retained {kept}; pressure today {psi.iloc[-1]:+.2f}, tilt β {fit['beta']:+.2f}; P(stressed/extreme) at 5/10/21d "
                     + " / ".join(f"{p_const[h]:.0%}→{p_tilt[h]:.0%}" for h in (5, 10, 21)))
    else:
        lines.append("leading: no feature retained by the incremental-value test; P(stressed/extreme) at 5/10/21d "
                     + " / ".join(f"{p_const[h]:.0%}" for h in (5, 10, 21)) + " under the constant matrix")
    scr.to_csv(out / "leading_screen.csv", index=False); sc.to_csv(out / "shock.csv"); P.to_csv(out / "transition_matrix.csv")
    g = ("EURUSD", "rr_25d", 21)
    card = report.card_md(lad, inc, split, g, today, profile_lines=lines)
    (out / "card.md").write_text(card); lad.to_csv(out / "ladder.csv", index=False); wf.to_csv(out / "walk_forward.csv", index=False)
    prof.to_csv(out / "state_profile.csv"); lab.to_csv(out / "labels.csv", index=False)
    report.plot_ladder(lad, g, out / "ladder.png"); report.plot_increments(inc, g, out / "increments.png")
    gc = gates.load_gates(a.gates)
    print(card); print("phase2 gate:", gates.gate_phase2(lad, gc)["pass"], "· phase3 gate:", gates.gate_phase3(wf, gc)["pass"])


def main(argv=None):
    p = argparse.ArgumentParser(prog="regime_ladder")
    p.add_argument("--config", default="configs/default.yaml"); p.add_argument("--gates", default="configs/gates.yaml")
    p.add_argument("--profile-config", default="configs/profile.yaml"); p.add_argument("--leading-config", default="configs/leading.yaml")
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("synth"); s.add_argument("--out", default="data/synth"); s.add_argument("--days", type=int, default=2520); s.add_argument("--energy-beta", type=float, default=0.0, dest="energy_beta")
    s.add_argument("--tenor", type=int, default=21); s.add_argument("--seed", type=int, default=0); s.set_defaults(f=cmd_synth)
    c = sp.add_parser("check"); c.add_argument("--td", required=True); c.add_argument("--labels"); c.set_defaults(f=cmd_check)
    l = sp.add_parser("labels"); l.add_argument("--market", required=True); l.add_argument("--pair", required=True); l.add_argument("--out", required=True)
    l.add_argument("--td", help="trade-day table, needed when labeller.target is forward_pnl"); l.add_argument("--target-archetype", default="straddle_atm"); l.set_defaults(f=cmd_labels)
    d = sp.add_parser("ladder"); d.add_argument("--td", required=True); d.add_argument("--labels", required=True); d.add_argument("--out", default="out/ladder"); d.set_defaults(f=cmd_ladder)
    e = sp.add_parser("evaluate"); e.add_argument("--td", required=True); e.add_argument("--labels", required=True); e.add_argument("--out", default="out/eval"); e.set_defaults(f=cmd_evaluate)
    v = sp.add_parser("validate"); v.add_argument("--seeds", type=int, default=8); v.add_argument("--days", type=int, default=1260); v.set_defaults(f=cmd_validate)
    k = sp.add_parser("shock"); k.add_argument("--market", required=True); k.add_argument("--labels"); k.add_argument("--out", default="out/shock"); k.set_defaults(f=cmd_shock)
    t = sp.add_parser("transitions"); t.add_argument("--labels", required=True); t.add_argument("--pressure"); t.add_argument("--out", default="out/transitions"); t.set_defaults(f=cmd_transitions)
    g = sp.add_parser("leading"); g.add_argument("--market", required=True); g.add_argument("--labels", required=True); g.add_argument("--td"); g.add_argument("--out", default="out/leading"); g.set_defaults(f=cmd_leading)
    i = sp.add_parser("inspect"); i.add_argument("--market", required=True); i.add_argument("--td"); i.add_argument("--pair", default="EURUSD"); i.add_argument("--out", default="out/inspect"); i.set_defaults(f=cmd_inspect)
    q = sp.add_parser("pack"); q.add_argument("--stage", required=True, choices=["data", "states", "ladder", "leading"]); q.add_argument("--src", help="folder written by inspect")
    q.add_argument("--td"); q.add_argument("--market"); q.add_argument("--label", default=""); q.add_argument("--out", default="out/packets"); q.set_defaults(f=cmd_pack)
    w = sp.add_parser("view"); w.add_argument("--src", action="append", required=True, help="inspect folder, or a parent holding one per pair; repeatable"); w.add_argument("--out", default="out/view.html"); w.add_argument("--label", default=""); w.set_defaults(f=cmd_view)
    m = sp.add_parser("demo"); m.add_argument("--out", default="out/demo"); m.add_argument("--days", type=int, default=2520); m.add_argument("--seed", type=int, default=0); m.add_argument("--energy-beta", type=float, default=0.0, dest="energy_beta"); m.set_defaults(f=cmd_demo)
    a = p.parse_args(argv); a.f(a)


if __name__ == "__main__":
    main()
