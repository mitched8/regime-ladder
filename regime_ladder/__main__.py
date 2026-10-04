"""CLI.  python -m regime_ladder {synth,check,labels,ladder,evaluate,demo} ...

`demo` runs the whole pipeline on synthetic data and writes a card: the smoke test
for a fresh environment, and the thing to run first on any new machine.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from . import checks, evaluate, features, gates, labels, ladder, report, schema, synth


def _cfg(path):
    return yaml.safe_load(Path(path).read_text())


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
    cfg = _cfg(a.config)["labeller"]
    m = pd.read_parquet(a.market)
    f = features.build(m, names=cfg["features"])
    score = labels.ewma(labels.composite(f), cfg["halflife"])
    lab = labels.threshold_labels(score, cfg["b1"], cfg["b2"], cfg["delta"])
    out = schema.labels_frame(a.pair, lab.index, lab.values)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out)
    print(f"{len(out)} labels, {labels.switches(lab)} switches, episodes {ladder.count_episodes(out)[a.pair]}")


def cmd_ladder(a):
    cfg = _cfg(a.config)
    td = schema.coerce(pd.read_parquet(a.td)); lab = pd.read_parquet(a.labels)
    cum = ladder.attach_entry_labels(ladder.cumulative(td, cfg["horizons"]), lab)
    lad = ladder.shrink(ladder.ladder(cum, lab, alpha=cfg["alpha"], ci=cfg["ci"], n_boot=cfg.get("n_boot", 1000)), kappa=cfg["kappa"])
    inc = ladder.increments(lad); split = ladder.persistence_split(td, lab, cfg["horizons"])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lad.to_csv(out / "ladder.csv", index=False); inc.to_csv(out / "increments.csv", index=False)
    split.to_csv(out / "persistence_split.csv", index=False)
    for g in lad[ladder.GROUP].drop_duplicates().itertuples(index=False):
        report.plot_ladder(lad, tuple(g), out / f"ladder_{'_'.join(map(str, g))}.png")
        report.plot_increments(inc, tuple(g), out / f"increments_{'_'.join(map(str, g))}.png")
    print(lad[lad["h"] <= 5].to_string(index=False, float_format=lambda v: f"{v:+.2f}"))


def cmd_evaluate(a):
    cfg = _cfg(a.config)
    td = schema.coerce(pd.read_parquet(a.td)); lab = pd.read_parquet(a.labels)
    cum = ladder.attach_entry_labels(ladder.cumulative(td, cfg["horizons"]), lab)
    wf = evaluate.walk_forward(cum, n_splits=cfg["eval"]["n_splits"])
    Path(a.out).mkdir(parents=True, exist_ok=True); wf.to_csv(Path(a.out) / "walk_forward.csv", index=False)
    g3 = gates.gate_phase3(wf, gates.load_gates(a.gates)); gates.write_result(g3, Path(a.out) / "gate_phase3.json")
    print(wf.to_string(index=False, float_format=lambda v: f"{v:+.3f}")); print("phase3 gate:", g3["pass"])


def cmd_demo(a):
    cfg = _cfg(a.config)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    m = synth.simulate_market(a.days, seed=a.seed); td = synth.simulate_trades(m, seed=a.seed + 1)
    rep = checks.check_trade_days(td); print(checks.format_report(rep)); checks.assert_clean(rep)
    f = features.build(m, names=cfg["labeller"]["features"])
    score = labels.ewma(labels.composite(f), cfg["labeller"]["halflife"])
    b1, b2 = cfg["labeller"]["b1"], cfg["labeller"]["b2"]
    if cfg["labeller"].get("estimate_breaks"):
        b1, b2 = labels.estimate_breaks(score, m["rv_1m"].shift(-21)); print(f"estimated boundaries: {b1:.1f} / {b2:.1f}")
    lab_s = labels.threshold_labels(score, b1, b2, cfg["labeller"]["delta"])
    lab = schema.labels_frame("EURUSD", lab_s.index, lab_s.values)
    acc = (lab_s.values == m["regime_true"].values).mean()
    print(f"labeller agreement with true regime: {acc:.0%}; switches {labels.switches(lab_s)} vs true {labels.switches(m['regime_true'])}")
    cum = ladder.attach_entry_labels(ladder.cumulative(td, cfg["horizons"]), lab)
    lad = ladder.shrink(ladder.ladder(cum, lab, alpha=cfg["alpha"], ci=cfg["ci"], n_boot=cfg.get("n_boot", 1000)), kappa=cfg["kappa"])
    inc = ladder.increments(lad); split = ladder.persistence_split(td, lab, cfg["horizons"])
    wf = evaluate.walk_forward(cum, n_splits=cfg["eval"]["n_splits"])
    g = ("EURUSD", "rr_25d", 21)
    card = report.card_md(lad, inc, split, g, lab_s.iloc[-1])
    (out / "card.md").write_text(card); lad.to_csv(out / "ladder.csv", index=False); wf.to_csv(out / "walk_forward.csv", index=False)
    report.plot_ladder(lad, g, out / "ladder.png"); report.plot_increments(inc, g, out / "increments.png")
    gc = gates.load_gates(a.gates)
    print(card); print("phase2 gate:", gates.gate_phase2(lad, gc)["pass"], "· phase3 gate:", gates.gate_phase3(wf, gc)["pass"])


def cmd_validate(a):
    """Recover the analytic truth from synthetic data with TRUE labels: the test of the statistics themselves."""
    cfg = _cfg(a.config); truth = synth.true_ladder(horizons=cfg["horizons"]); rows = []
    for seed in range(a.seeds):
        m = synth.simulate_market(a.days, seed=seed); td = synth.simulate_trades(m, seed=seed + 100)
        lab = schema.labels_frame("EURUSD", m.index, m["regime_true"].values)
        lad = ladder.ladder(ladder.attach_entry_labels(ladder.cumulative(td, cfg["horizons"]), lab), lab, ci=cfg["ci"], n_boot=cfg.get("n_boot", 1000))
        x = lad[lad["regime"] != "ALL"].merge(truth, on=["archetype", "regime", "h"]); x = x[x["n_eff"] >= 20]
        rows.append((seed, ((x["ci_lo"] <= x["true_ev"]) & (x["true_ev"] <= x["ci_hi"])).mean(), ((x["mean"] - x["true_ev"]).abs() / x["se"]).median()))
    r = pd.DataFrame(rows, columns=["seed", "coverage_90", "median_abs_err_over_se"])
    print(r.to_string(index=False, float_format=lambda v: f"{v:.3f}")); print(f"mean coverage {r.coverage_90.mean():.3f} (nominal 0.90)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="regime_ladder")
    p.add_argument("--config", default="configs/default.yaml"); p.add_argument("--gates", default="configs/gates.yaml")
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("synth"); s.add_argument("--out", default="data/synth"); s.add_argument("--days", type=int, default=1260)
    s.add_argument("--tenor", type=int, default=21); s.add_argument("--seed", type=int, default=0); s.set_defaults(f=cmd_synth)
    c = sp.add_parser("check"); c.add_argument("--td", required=True); c.add_argument("--labels"); c.set_defaults(f=cmd_check)
    l = sp.add_parser("labels"); l.add_argument("--market", required=True); l.add_argument("--pair", required=True)
    l.add_argument("--out", required=True); l.set_defaults(f=cmd_labels)
    d = sp.add_parser("ladder"); d.add_argument("--td", required=True); d.add_argument("--labels", required=True)
    d.add_argument("--out", default="out/ladder"); d.set_defaults(f=cmd_ladder)
    e = sp.add_parser("evaluate"); e.add_argument("--td", required=True); e.add_argument("--labels", required=True)
    e.add_argument("--out", default="out/eval"); e.set_defaults(f=cmd_evaluate)
    v = sp.add_parser("validate"); v.add_argument("--seeds", type=int, default=8); v.add_argument("--days", type=int, default=1260); v.set_defaults(f=cmd_validate)
    m = sp.add_parser("demo"); m.add_argument("--out", default="out/demo"); m.add_argument("--days", type=int, default=1260)
    m.add_argument("--seed", type=int, default=0); m.set_defaults(f=cmd_demo)
    a = p.parse_args(argv); a.f(a)


if __name__ == "__main__":
    main()
