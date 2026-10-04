"""CLI.  python -m regime_ladder {synth,check,labels,ladder,evaluate,validate,demo} ...

`demo` runs the whole pipeline on synthetic data — calibrated six-state labels, ladder, profile,
sub-state discovery, gates — and writes a card. It is the smoke test for a fresh environment.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import checks, discover, evaluate, features, gates, labels, ladder, profile, report, schema, synth, tags


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
    m = synth.simulate_market(a.days, seed=a.seed)
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


def cmd_demo(a):
    cfg = _cfg(a.config); pcfg = _cfg(a.profile_config)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    # 10 years of market for labelling and profiles; the last 5 carry trades (as in the real plan)
    m = synth.simulate_market(a.days, seed=a.seed)
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
    p.add_argument("--profile-config", default="configs/profile.yaml")
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("synth"); s.add_argument("--out", default="data/synth"); s.add_argument("--days", type=int, default=2520)
    s.add_argument("--tenor", type=int, default=21); s.add_argument("--seed", type=int, default=0); s.set_defaults(f=cmd_synth)
    c = sp.add_parser("check"); c.add_argument("--td", required=True); c.add_argument("--labels"); c.set_defaults(f=cmd_check)
    l = sp.add_parser("labels"); l.add_argument("--market", required=True); l.add_argument("--pair", required=True); l.add_argument("--out", required=True)
    l.add_argument("--td", help="trade-day table, needed when labeller.target is forward_pnl"); l.add_argument("--target-archetype", default="straddle_atm"); l.set_defaults(f=cmd_labels)
    d = sp.add_parser("ladder"); d.add_argument("--td", required=True); d.add_argument("--labels", required=True); d.add_argument("--out", default="out/ladder"); d.set_defaults(f=cmd_ladder)
    e = sp.add_parser("evaluate"); e.add_argument("--td", required=True); e.add_argument("--labels", required=True); e.add_argument("--out", default="out/eval"); e.set_defaults(f=cmd_evaluate)
    v = sp.add_parser("validate"); v.add_argument("--seeds", type=int, default=8); v.add_argument("--days", type=int, default=1260); v.set_defaults(f=cmd_validate)
    m = sp.add_parser("demo"); m.add_argument("--out", default="out/demo"); m.add_argument("--days", type=int, default=2520); m.add_argument("--seed", type=int, default=0); m.set_defaults(f=cmd_demo)
    a = p.parse_args(argv); a.f(a)


if __name__ == "__main__":
    main()
