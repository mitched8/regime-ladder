"""Out-of-sample evaluation of the ladder against benchmarks.

Walk-forward by entry date: estimate per-regime (and unconditional) means on
the training window, score the test window. Reports, per horizon:
  mse_uncond, mse_regime, improvement, rank_ic (Spearman), calibration slope.

This is the skeleton of the Phase 3 go/no-go. It is intentionally simple:
expanding windows, no purging beyond the horizon itself. Extend, do not replace.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .ladder import GROUP


def _splits(dates: pd.Series, n_splits: int, min_train_frac: float = 0.4):
    u = np.array(sorted(dates.unique()))
    cuts = np.linspace(int(len(u) * min_train_frac), len(u) - 1, n_splits + 1).astype(int)
    for a, b in zip(cuts[:-1], cuts[1:]):
        yield u[a], u[b]


def walk_forward(cum_lab: pd.DataFrame, n_splits: int = 4, min_train_frac: float = 0.4) -> pd.DataFrame:
    rows = []
    for gkey, g in cum_lab.groupby(GROUP):
        for h, gh in g.groupby("h"):
            preds_r, preds_u, ys = [], [], []
            for t0, t1 in _splits(gh["entry_date"], n_splits, min_train_frac):
                # purge: training outcomes must be realised before the test window starts
                train = gh[gh["entry_date"] < t0 - pd.tseries.offsets.BDay(int(h))]
                test = gh[(gh["entry_date"] >= t0) & (gh["entry_date"] < t1)]
                if len(train) < 20 or len(test) == 0:
                    continue
                m_u = train["cum_pnl"].mean()
                m_r = train.groupby("regime")["cum_pnl"].mean()
                preds_u.append(np.full(len(test), m_u))
                preds_r.append(test["regime"].map(m_r).fillna(m_u).values)
                ys.append(test["cum_pnl"].values)
            if not ys:
                continue
            y, pu, pr = map(np.concatenate, (ys, preds_u, preds_r))
            mse_u, mse_r = float(((y - pu) ** 2).mean()), float(((y - pr) ** 2).mean())
            ic = float(spearmanr(pr, y).statistic) if np.std(pr) > 0 else np.nan
            slope = float(np.polyfit(pr, y, 1)[0]) if np.std(pr) > 0 else np.nan
            rows.append((*gkey, int(h), len(y), mse_u, mse_r, 1 - mse_r / mse_u, ic, slope))
    return pd.DataFrame(rows, columns=GROUP + ["h", "n_test", "mse_uncond", "mse_regime", "improvement", "rank_ic", "calib_slope"])
