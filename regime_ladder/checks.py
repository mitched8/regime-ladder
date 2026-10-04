"""Integrity checks on the trade-day table and label coverage.

Run on every data build. Each check returns (passed, detail). `assert_clean`
raises on any failure so an agent cannot proceed on a broken table without
noticing. These are integrity checks, not research gates (see gates.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import COMPONENTS, REQUIRED


def check_trade_days(td: pd.DataFrame, component_tol: float = 1e-6) -> dict:
    r = {}
    missing = [c for c in REQUIRED if c not in td.columns]
    r["required_columns"] = (not missing, f"missing: {missing}" if missing else "ok")
    if missing:
        return r
    dup = td.duplicated(["trade_id", "date"]).sum()
    r["no_duplicate_trade_date"] = (dup == 0, f"{dup} duplicate rows")
    nan = int(td["pnl"].isna().sum())
    r["no_nan_pnl"] = (nan == 0, f"{nan} NaN pnl rows")
    g = td.sort_values(["trade_id", "age"]).groupby("trade_id")["age"]
    bad_start = int((g.first() != 1).sum())
    bad_step = int(g.apply(lambda a: (np.diff(a.values) != 1).any()).sum())
    r["ages_consecutive_from_1"] = (bad_start + bad_step == 0, f"{bad_start} trades not starting at age 1, {bad_step} with gaps")
    over = int((td["age"] > td["tenor_days"]).sum())
    r["age_within_tenor"] = (over == 0, f"{over} rows with age > tenor")
    mono = int(td.sort_values(["trade_id", "age"]).groupby("trade_id")["date"].apply(lambda d: (np.diff(d.values) <= np.timedelta64(0, "ns")).any()).sum())
    r["dates_increase_with_age"] = (mono == 0, f"{mono} trades with non-increasing dates")
    before = int((td["date"] <= td["entry_date"]).sum())
    r["date_after_entry"] = (before == 0, f"{before} rows dated on/before entry")
    if all(c in td.columns for c in COMPONENTS):
        gap = (td["pnl"] - td[COMPONENTS].sum(axis=1)).abs()
        n_bad = int((gap > component_tol).sum())
        r["components_sum_to_pnl"] = (n_bad == 0, f"{n_bad} rows off by > {component_tol}; max gap {gap.max():.3g}")
    per_trade = td.groupby("trade_id")["age"].max()
    short = int((per_trade < td.groupby("trade_id")["tenor_days"].first()).sum())
    r["info_trades_not_yet_expired"] = (True, f"{short} trades shorter than tenor (live or truncated)")
    return r


def check_label_coverage(entries: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Every (pair, entry_date) must have a label."""
    e = entries[["pair", "entry_date"]].drop_duplicates()
    m = e.merge(labels.rename(columns={"date": "entry_date"}), on=["pair", "entry_date"], how="left")
    n_missing = int(m["regime"].isna().sum())
    return {"label_coverage": (n_missing == 0, f"{n_missing} of {len(e)} entry dates unlabelled")}


def assert_clean(report: dict) -> None:
    failed = {k: v[1] for k, v in report.items() if not v[0]}
    if failed:
        raise AssertionError("integrity checks failed: " + "; ".join(f"{k}: {v}" for k, v in failed.items()))


def format_report(report: dict) -> str:
    return "\n".join(f"[{'PASS' if ok else 'FAIL'}] {k}: {detail}" for k, (ok, detail) in report.items())
