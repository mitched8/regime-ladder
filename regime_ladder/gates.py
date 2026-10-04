"""Pre-registered gates, evaluated against results files.

Thresholds live in configs/gates.yaml and are changed only by the research
owner, with a logged reason. The code here only reads and compares.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml


def load_gates(path="configs/gates.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def gate_phase1(check_report: dict, golden: pd.DataFrame | None, cfg: dict) -> dict:
    """All integrity checks pass; golden trades reproduce within tolerance."""
    res = {"integrity_all_pass": all(ok for ok, _ in check_report.values())}
    if golden is not None and len(golden):
        tol = cfg["phase1"]["golden_abs_tol"]
        res["golden_trades_match"] = bool((golden["abs_error"] <= tol).all())
    res["pass"] = all(v for k, v in res.items() if k != "pass")
    return res


def gate_phase2(lad: pd.DataFrame, cfg: dict) -> dict:
    """Entry labels separate outcomes: regime means differ in sign-consistent ways at short horizons.

    Minimal form: at each h <= max_h, the spread between best and worst regime mean exceeds
    `min_spread_se` times the larger of their standard errors.
    """
    c = cfg["phase2"]
    ok_h = {}
    for (pair, arch, tenor, h), g in lad[lad["regime"] != "ALL"].groupby(["pair", "archetype", "tenor_days", "h"]):
        if h > c["max_h"]:
            continue
        g = g.dropna(subset=["mean", "se"])
        if len(g) < 2:
            continue
        hi, lo = g.loc[g["mean"].idxmax()], g.loc[g["mean"].idxmin()]
        spread = hi["mean"] - lo["mean"]
        ok_h[(pair, arch, tenor, int(h))] = bool(spread > c["min_spread_se"] * max(hi["se"], lo["se"]))
    frac = sum(ok_h.values()) / len(ok_h) if ok_h else 0.0
    return {"cells": ok_h, "fraction_separated": frac, "pass": frac >= c["min_fraction_cells"]}


def gate_phase3(wf: pd.DataFrame, cfg: dict) -> dict:
    """Ladder beats the unconditional benchmark out of sample at short horizons."""
    c = cfg["phase3"]
    w = wf[wf["h"] <= c["max_h"]]
    res = {
        "improvement_min": float(w["improvement"].min()) if len(w) else float("nan"),
        "rank_ic_min": float(w["rank_ic"].min()) if len(w) else float("nan"),
        "calib_slope_range": (float(w["calib_slope"].min()), float(w["calib_slope"].max())) if len(w) else (None, None),
    }
    res["pass"] = bool(len(w) and res["improvement_min"] > c["min_improvement"] and res["rank_ic_min"] > c["min_rank_ic"]
                       and c["calib_slope"][0] <= res["calib_slope_range"][0] and res["calib_slope_range"][1] <= c["calib_slope"][1])
    return res


def write_result(obj: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str))
