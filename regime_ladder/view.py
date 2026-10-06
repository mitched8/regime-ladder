"""One self-contained HTML page over `inspect` output, for looking at results without a notebook.

    python -m regime_ladder view --src out/inspect [--src out/inspect_usdjpy ...] [--out out/view.html]

`--src` is an inspect folder, or a parent folder holding one inspect folder per pair; several may be
given. The page embeds the data as JSON and draws with plain SVG: no server, no network, no library,
so it opens from a file on a locked-down machine. It reads only files `inspect` writes; nothing is
recomputed here, so a number on the page is a number in those files.

The Ladder, Matrix and Out-of-sample tabs are pivots over the ladder tables: any of pair, strategy,
tenor and entry state can be the lines, the panels or a filter, and the horizon or the tenor is the
x axis. That is the comparison the owner asked for; the other tabs show one pair's series at a time.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

TEMPLATE = Path(__file__).with_name("view_template.html")
LADDER_COLS = ["pair", "archetype", "tenor_days", "regime", "h", "to_expiry", "n_eff", "mean", "se", "ci_lo", "ci_hi",
               "p_profit", "q05", "q50", "q95", "es05", "episodes", "w_shrink", "mean_shrunk"]
SERIES = {  # file -> columns to embed as daily series (whatever of them exists)
    "composite.csv": ["composite", "smoothed", "direction_score", "level", "direction", "regime"],
    "finder_inputs.csv": ["target", "level_target"],
    "shock.csv": ["surprise", "cusum_pressure", "bocd_p_short", "recent_abs", "score", "har_vol_forecast"],
    "leading_features.csv": None,  # all columns
    "pressure.csv": ["pressure"],
    "features.csv": None,
}


def folders(srcs) -> list[Path]:
    """Inspect folders under the given paths: a folder is one if it holds composite.csv."""
    out = []
    for s in srcs:
        s = Path(s)
        if (s / "composite.csv").exists():
            out.append(s)
        else:
            out += sorted(p for p in s.iterdir() if p.is_dir() and (p / "composite.csv").exists())
    if not out:
        raise SystemExit(f"no inspect output under {', '.join(map(str, srcs))} (looked for composite.csv)")
    return out


def _csv(p: Path, **kw):
    return pd.read_csv(p, **kw) if p.exists() else None


def _json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _clean(o):
    """JSON-safe: numpy scalars to Python, NaN/inf to None, pandas to lists/dicts."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return [_clean(r) for r in o.to_dict("records")]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (float, np.floating)):
        return None if (math.isnan(o) or math.isinf(o)) else round(float(o), 5)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def _pair_name(f: Path) -> str:
    meta = _json(f / "meta.json")
    if meta and meta.get("pair"):
        return str(meta["pair"])
    rd = f / "README.md"
    if rd.exists():
        for line in rd.read_text().splitlines():
            if line.startswith("Pair "):
                return line.split()[1].rstrip(".")
    return f.name


def one_pair(f: Path) -> tuple[str, dict, dict, dict]:
    """(pair, daily series, per-pair tables, ladder tables) for one inspect folder."""
    pair = _pair_name(f)
    comp = pd.read_csv(f / "composite.csv", index_col=0, parse_dates=True)
    idx = comp.index
    series = {"dates": [d.strftime("%Y-%m-%d") for d in idx], "_groups": {}}
    for name, cols in SERIES.items():
        df = _csv(f / name, index_col=0, parse_dates=True)
        if df is None:
            continue
        df = df.reindex(idx)
        for c in (cols or list(df.columns)):
            if c in df.columns and c not in series:
                series[c] = _clean(df[c].tolist())
                series["_groups"].setdefault(name, []).append(c)
    tm = _csv(f / "transition_matrix.csv", index_col=0)
    tc = _csv(f / "transition_counts.csv", index_col=0)
    fc = _csv(f / "fan_constant.csv")
    ft = _csv(f / "fan_tilted.csv")
    sp = _csv(f / "state_profile.csv", index_col=0)
    lp = _csv(f / "shock_lead_profile.csv")
    meta = _json(f / "meta.json") or {}
    tables = {
        "meta": meta,
        "today": {"date": series["dates"][-1], "state": meta.get("today_state") or str(comp["regime"].iloc[-1])},
        "spec": _json(f / "finder_spec.json"),
        "finder_r2": _json(f / "finder_r2.json"),
        "durations": _csv(f / "durations.csv"),
        "matrix": {"states": list(tm.index), "P": tm.values} if tm is not None else None,
        "counts": {"states": list(tc.index), "P": tc.values} if tc is not None else None,
        "fan": {"h": fc["h"].tolist(), "constant": {c: fc[c].tolist() for c in fc.columns if c != "h"},
                "tilted": ({c: ft[c].tolist() for c in ft.columns if c != "h"} if ft is not None else None)} if fc is not None else None,
        "state_profile": {"rows": list(sp.index), "cols": list(sp.columns), "values": sp.values} if sp is not None else None,
        "lead_profile": lp.iloc[0].to_dict() if lp is not None and len(lp) else None,
        "screen": _csv(f / "leading_screen.csv"),
        "finder_report": _csv(f / "finder_report.csv", index_col=0),
        "tilt_fits": _csv(f / "tilt_fits.csv"),
        "gates": {k: _json(f / f"gate_{k}.json") for k in ("phase2", "phase3", "phase4") if (f / f"gate_{k}.json").exists()},
        "tag_cells": _json(f / "tag_cells.json"),
        "discovery": _json(f / "discovery.json"),
    }
    lad = _csv(f / "ladder.csv")
    ladders = {
        "ladder": lad.reindex(columns=[c for c in LADDER_COLS if c in lad.columns]) if lad is not None else None,
        "increments": _csv(f / "increments.csv"),
        "walk_forward": _csv(f / "walk_forward.csv"),
    }
    return pair, series, tables, ladders


def sweep_payload(sweep_dir: str | Path, dates: list[str]) -> dict:
    """sweep.csv plus one regime strip per variant, aligned to the page's dates."""
    d = Path(sweep_dir)
    rows = pd.read_csv(d / "sweep.csv")
    idx = pd.DatetimeIndex(dates)
    strips = {}
    for r in rows.itertuples():
        f = d / f"labels_{r.name}.parquet"
        if f.exists():
            lab = pd.read_parquet(f).set_index("date")["regime"]
            strips[r.name] = lab.reindex(idx).tolist()
    readme = (d / "README.md").read_text() if (d / "README.md").exists() else ""
    fit_end = readme.split("dates <= ")[1].split(";")[0] if "dates <= " in readme else None
    return {"rows": rows, "strips": strips, "fit_end": fit_end}


def payload(srcs, label: str = "", sweep: str | None = None) -> dict:
    fs = folders(srcs)
    out = {"label": label, "generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "sources": [str(f) for f in fs],
           "pairs": [], "series": {}, "tables": {}, "ladder": [], "increments": [], "walk_forward": []}
    parts = {"ladder": [], "increments": [], "walk_forward": []}
    for f in fs:
        pair, series, tables, ladders = one_pair(f)
        if pair in out["series"]:
            raise SystemExit(f"pair {pair} appears twice ({f}); give one inspect folder per pair")
        out["pairs"].append(pair)
        out["series"][pair] = series
        out["tables"][pair] = tables
        for k, df in ladders.items():
            if df is not None:
                parts[k].append(df)
    for k, dfs in parts.items():
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            keys = [c for c in ("pair", "archetype", "tenor_days", "regime", "h", "bucket") if c in df.columns]
            out[k] = df.drop_duplicates(subset=keys, keep="last")
    if sweep:
        out["sweep"] = sweep_payload(sweep, out["series"][out["pairs"][0]]["dates"])
    return _clean(out)


def build(srcs, out: str | Path = "out/view.html", label: str = "", sweep: str | None = None) -> Path:
    data = payload(srcs, label, sweep)
    js = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    html = TEMPLATE.read_text().replace("__DATA__", js).replace("__TITLE__", (label or "regime ladder") + " · view")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return out
