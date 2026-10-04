"""Tables, plots and the per-component card. Research output, not a UI."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .ladder import GROUP  # noqa: E402

COLOURS = {"carry": "#3B6FD4", "rising": "#2EAE7A", "transition": "#2EAE7A", "crisis": "#E8743B",
           "normalising": "#8B5CF6", "settling": "#0EA5E9", "ALL": "#6B7280"}


def _sel(df: pd.DataFrame, group: tuple) -> pd.DataFrame:
    m = pd.Series(True, index=df.index)
    for c, v in zip(GROUP, group):
        m &= df[c] == v
    return df[m]


def ladder_table_md(lad: pd.DataFrame, group: tuple, regime: str) -> str:
    g = _sel(lad, group)
    g = g[g["regime"] == regime].sort_values("h")
    lines = ["| h | EV | 90% interval | P(profit) | ES 5% | n_eff | episodes |", "|---|---|---|---|---|---|---|"]
    for _, r in g.iterrows():
        h = f"{int(r['h'])}d" + (" (expiry)" if r["to_expiry"] else "")
        lines.append(f"| {h} | {r['mean']:+.2f} | [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] | {r['p_profit']:.0%} | {r['es05']:+.1f} | {r['n_eff']:.0f} | {int(r['episodes'])} |")
    return "\n".join(lines)


def card_md(lad: pd.DataFrame, inc: pd.DataFrame, split: pd.DataFrame, group: tuple, regime_today: str,
            profile_lines: list | None = None) -> str:
    pair, arch, tenor = group
    out = [f"## {pair} · {arch} · {tenor}d · entry regime today: {regime_today.upper()}",
           "Reference: hold-for-h from today, no exit rule; cash per unit standard notional.", "",
           ladder_table_md(lad, group, regime_today), ""]
    s = _sel(split, group)
    s = s[s["regime"] == regime_today].sort_values("h")
    if len(s):
        r = s.iloc[-1]
        out += [f"**Regime dependence at {int(r['h'])}d:** stayed {r['ev_stayed']:+.1f} · broke {r['ev_broke']:+.1f} · "
                f"historical break frequency {r['q_broke']:.0%}", ""]
    i = _sel(inc, group)
    i = i[i["regime"] == regime_today].sort_values("h_from")
    wide = i[(i["h_to"] - i["h_from"]) >= 2]  # ignore one-day buckets (d1 and a lone expiry day)
    if len(i) and len(wide):
        worst = wide.loc[wide["earn_per_day"].abs().idxmax()]
        first = i.iloc[0]
        out += [f"**Where it earns:** {worst['bucket']}: {worst['earn_per_day']:+.2f}/day (vs {first['bucket']}: {first['earn_per_day']:+.2f}/day)", ""]
    eps = int(_sel(lad, group).query("regime == @regime_today")["episodes"].iloc[0])
    if eps < 10:
        out += [f"_Evidence note: only {eps} independent {regime_today} episodes behind these numbers._", ""]
    if profile_lines:
        out += ["**State profile (today vs this state's history, vs all days):**"] + [f"- {l}" for l in profile_lines] + [""]
    return "\n".join(out)


def plot_ladder(lad: pd.DataFrame, group: tuple, path: str) -> None:
    g = _sel(lad, group)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for reg, gr in g.groupby("regime"):
        if reg == "ALL":
            continue
        gr = gr.sort_values("h")
        ax.fill_between(gr["h"], gr["ci_lo"], gr["ci_hi"], color=COLOURS.get(reg, "#999"), alpha=0.15, lw=0)
        ax.plot(gr["h"], gr["mean"], "-o", color=COLOURS.get(reg, "#999"), ms=4, label=f"{reg} entry")
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xlabel("horizon h (trading days)")
    ax.set_ylabel("cumulative EV per unit notional")
    ax.set_title(" · ".join(map(str, group)))
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_increments(inc: pd.DataFrame, group: tuple, path: str) -> None:
    g = _sel(inc, group)
    g = g[g["regime"] != "ALL"]
    buckets = list(dict.fromkeys(g["bucket"]))
    regs = sorted(g["regime"].unique())
    fig, ax = plt.subplots(figsize=(7, 3.0))
    w = 0.8 / max(1, len(regs))
    for j, reg in enumerate(regs):
        gr = g[g["regime"] == reg].set_index("bucket").reindex(buckets)
        ax.bar([b + (j - (len(regs) - 1) / 2) * w for b in range(len(buckets))], gr["earn_per_day"], w,
               color=COLOURS.get(reg, "#999"), label=reg)
    ax.set_xticks(range(len(buckets)))
    ax.set_xticklabels(buckets)
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_ylabel("earn per day")
    ax.set_title("increments by age bucket · " + " · ".join(map(str, group)))
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


TENOR_LABELS = {5: "1W", 10: "2W", 21: "1M", 42: "2M", 63: "3M", 126: "6M", 252: "1Y"}


def plot_tenor_curve(lad: pd.DataFrame, pair: str, archetype: str, h: int, path: str, regimes=None) -> None:
    """EV at a fixed horizon across tenors, one line per entry regime: which tenor is best paid from here."""
    g = lad[(lad["pair"] == pair) & (lad["archetype"] == archetype) & (lad["h"] == h) & (lad["regime"] != "ALL")]
    regimes = regimes or sorted(g["regime"].unique())
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for reg in regimes:
        gr = g[g["regime"] == reg].sort_values("tenor_days")
        if gr.empty:
            continue
        ax.errorbar(gr["tenor_days"], gr["mean"], yerr=[gr["mean"] - gr["ci_lo"], gr["ci_hi"] - gr["mean"]], fmt="-o", ms=4,
                    color=COLOURS.get(reg, "#999"), capsize=2, lw=1.4, label=f"{reg} entry")
    ax.set_xscale("log")
    ticks = sorted(g["tenor_days"].unique())
    ax.set_xticks(ticks)
    ax.set_xticklabels([TENOR_LABELS.get(int(t), str(int(t)) + "d") for t in ticks])
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xlabel("tenor")
    ax.set_ylabel(f"EV at h={h}d per unit vega")
    ax.set_title(f"{pair} · {archetype} · tenor curve at {h}d")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
