"""Build the reviewer calibration packets on synthetic data where the truth is known.

    PYTHONPATH=. python docs/calibration/make_calibration.py      (from the repo root)

Three worlds, each with one extra "desk series" (`desk_flow`) registered the way a real desk series would be:

  A  energy world, desk_flow = the world's own stored-energy rule (the true pressure) -> should be retained; Gate 4 passes
  B  null world,   desk_flow = the same construction, but pressure drives nothing   -> nothing retained; Gate 4 fails
  C  null world,   desk_flow built with a timestamp bug: it is the trailing percentile of realised vol TEN DAYS
     AHEAD, presented as if it were trailing                                       -> it will be retained; the
     reviewer must catch it from the packet alone

The answer key (ANSWER_KEY.md) is for the owner; never give it to the reviewer.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import yaml

from regime_ladder import gates, inspect, leading, pack, synth
from regime_ladder.features import rolling_percentile  # noqa: F401  (used by flow_leaky)

OUT = Path(__file__).parent
CFGS = ["configs/default.yaml", "configs/gates.yaml", "configs/leading.yaml", "configs/profile.yaml", "configs/archetypes.yaml"]


def flow_true(m, asof=None):
    """The world's own stored-energy rule applied to the path so far, on the 0..100 scale the pressure index
    expects (50 = normal, +25 per unit). In world A it drives transitions; in world B the same rule drives nothing."""
    mm = m if asof is None else m.loc[:asof]
    gz = leading.gap_z(mm)
    x = pd.Series([synth.energy_pressure(g, a) if np.isfinite(g) else np.nan for g, a in zip(gz, mm["atm_1m"])], index=mm.index)
    return (50 + 25 * x).clip(0, 100).rename("desk_flow")


def flow_leaky(m, asof=None, window=756):
    x = m["rv_1w"].shift(-10)                       # the bug: a future value stamped on today
    x = x if asof is None else x.loc[:asof]
    return rolling_percentile(x, window).rename("desk_flow")


def expected_forward_pnl(m, h=5, tenor=21):
    """The noiseless forward target the unit tests use: the earn the true state path implies. The noisy
    simulated trades are kept for the ladder and data packets; with them, ten years of one pair has too
    little power for the outcome test (see ANSWER_KEY.md)."""
    s = m["state_true"].values
    return pd.Series([sum(synth.earn("straddle_atm", s[i + a], tenor - a) for a in range(1, h + 1)) if i + h < len(m) else np.nan
                      for i in range(len(m))], index=m.index)


def run(name, energy_beta, flow, stages):
    m = synth.simulate_market(2520, seed=0, energy_beta=energy_beta)
    td = synth.simulate_trades(m, seed=1)
    cfg, pcfg, lcfg = (yaml.safe_load(open(CFGS[i])) for i in (0, 3, 2))
    leading.LEADING["desk_flow"] = flow
    lcfg["leading"]["features"]["desk_flow"] = {}
    src = Path("out/calibration") / name
    target = expected_forward_pnl(m)
    res = inspect.dump(m, cfg, pcfg, lcfg, gates.load_gates(), "EURUSD", src, td=td, target=target,
                       target_name="expected forward 5d straddle P&L given the true state path (synthetic, noiseless)")
    for st in stages:
        p = pack.build(st, OUT / name / st, src=src, td=td if st == "data" else None, market=m,
                       label=f"calibration {name} (synthetic)", config_paths=CFGS, archetypes_cfg=yaml.safe_load(open(CFGS[4])))
        print(name, st, p, f"{p.stat().st_size / 1024:.0f} KB")
    print(name, "retained:", res["retained"])
    del leading.LEADING["desk_flow"]


if __name__ == "__main__":
    run("A_energy", 3.0, flow_true, ["leading", "states", "ladder", "data"])
    run("B_null", 0.0, flow_true, ["leading"])
    run("C_planted", 0.0, flow_leaky, ["leading"])
