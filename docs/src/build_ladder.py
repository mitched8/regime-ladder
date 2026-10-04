"""Deep dive 6: ladder.html — The horizon ladder and its intervals."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt, yaml
from common import world, figure, save, table, page, COLOURS, BUILT, HYP, ROOT
import sys; sys.path.insert(0, str(ROOT))
from regime_ladder import synth, labels, ladder, schema, combine, evaluate, gates, report

W = world(); m, m5, td, st, est, lab_true, lab_est = W["m"], W["m5"], W["td"], W["st"], W["est"], W["lab_true"], W["lab_est"]
C = COLOURS; S = list(synth.STATES); S6 = list(labels.STATES6)
cfg = yaml.safe_load(open(ROOT / "configs/default.yaml")); gc = gates.load_gates(str(ROOT / "configs/gates.yaml"))
truth = synth.true_ladder(horizons=(1, 3, 5, 10, 20))
figs = {}
H = (1, 3, 5, 10, 20)
cum_t = ladder.attach_entry_labels(ladder.cumulative(td, H), lab_true); cum_e = ladder.attach_entry_labels(ladder.cumulative(td, H), lab_est)
lad_t = ladder.shrink(ladder.ladder(cum_t, lab_true, ci="boot", n_boot=800), kappa=20); lad_e = ladder.shrink(ladder.ladder(cum_e, lab_est, ci="boot", n_boot=800), kappa=20)
G = ("EURUSD", "rr_25d", 21)

# ---- F1 from trade-day rows to the ladder: one entry's path, many entries' paths in one state, the distribution at h=5
tdS = td[td.archetype == "straddle_atm"].copy(); tdS["cum"] = tdS.groupby("trade_id")["pnl"].cumsum()
ent = lab_true.set_index("date")["regime"]
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.2), gridspec_kw={"width_ratios": [1, 1.3, 1]})
one = tdS[tdS.trade_id == tdS.trade_id.iloc[5000]]
axs[0].plot(one.age, one.cum, "-o", ms=3, color="#222");
for h in H: axs[0].axvline(h, color="#4F46E5", lw=0.7, ls=":")
axs[0].set_xlabel("age (days since entry)"); axs[0].set_ylabel("cumulative P&L"); axs[0].set_title("One trade: the backtester's daily rows, summed", fontsize=9)
ids = tdS[tdS.entry_date.isin(ent[ent == "stressed"].index)].trade_id.unique()[:60]
for tid in ids:
    g = tdS[tdS.trade_id == tid]; axs[1].plot(g.age, g.cum, color=C["stressed"], alpha=0.25, lw=0.8)
mean_path = tdS[tdS.trade_id.isin(tdS[tdS.entry_date.isin(ent[ent == "stressed"].index)].trade_id)].groupby("age")["cum"].mean()
axs[1].plot(mean_path.index, mean_path.values, color="#222", lw=2, label="mean over all stressed entries"); axs[1].set_xlabel("age"); axs[1].set_title("Every straddle entered in stressed: the paths the ladder averages", fontsize=9); axs[1].legend(frameon=False, fontsize=7.5)
at5 = tdS[(tdS.age == 5)].merge(lab_true.rename(columns={"date": "entry_date"}), on=["pair", "entry_date"])
for s_, col in (("carry", C["carry"]), ("stressed", C["stressed"])):
    axs[2].hist(at5[at5.regime == s_]["cum"], bins=40, density=True, alpha=0.5, color=col, label=f"{s_} entries")
axs[2].set_xlabel("cumulative P&L at h = 5"); axs[2].set_yticks([]); axs[2].set_title("The outcome distribution at one rung", fontsize=9); axs[2].legend(frameon=False, fontsize=7.5)
figs["build"] = save(fig, "la_build.png")

# ---- F2 dependence: correlation of 5-day outcomes between entries at lag L, within vs across episodes
ep_ids = ladder.episode_ids(lab_true); c5 = cum_t[(cum_t.archetype == "straddle_atm") & (cum_t.h == 5)].merge(ep_ids, on=["pair", "entry_date"]).sort_values("entry_date").reset_index(drop=True)
rows = []
for L in range(1, 21):
    a, b = c5.iloc[:-L], c5.iloc[L:]
    same = (a["episode"].values == b["episode"].values) & (a["regime"].values == b["regime"].values)
    xa, xb = a["cum_pnl"].values, b["cum_pnl"].values
    rows.append(dict(lag=L, within=np.corrcoef(xa[same], xb[same])[0, 1] if same.sum() > 30 else np.nan, across=np.corrcoef(xa[~same], xb[~same])[0, 1] if (~same).sum() > 30 else np.nan))
dep = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(dep.lag, dep.within, "o-", ms=3.5, color="#E8743B", label="both entries in the same regime episode"); ax.plot(dep.lag, dep.across, "o-", ms=3.5, color="#9aa0a6", label="entries in different episodes")
ax.axvline(5, color="#4F46E5", ls=":", lw=1); ax.text(5.2, 0.85, "h = 5: overlap ends", fontsize=8, color="#4F46E5"); ax.axhline(0, color="#333", lw=0.7)
ax.set_xlabel("days between the two entries"); ax.set_ylabel("correlation of 5-day outcomes"); ax.set_title("Two kinds of dependence between entries"); ax.legend(frameon=False, fontsize=8)
figs["dep"] = save(fig, "la_dep.png")

# ---- F3 interval methods: coverage by h and by n_eff across seeds (true labels)
from regime_ladder.ladder import block_bootstrap_ci, episode_cluster_ci, hac_se
cells = []
for seed in range(8):
    mm = synth.simulate_market(1260, seed=seed); tt = schema.coerce(synth.simulate_trades(mm, seed=seed + 100)); lt = schema.labels_frame("EURUSD", mm.index, mm["state_true"].values)
    cum = ladder.attach_entry_labels(ladder.cumulative(tt, H), lt).merge(ladder.episode_ids(lt), on=["pair", "entry_date"]); rng = np.random.default_rng(seed)
    for (a_, reg, h), g in cum.groupby(["archetype", "regime", "h"]):
        g = g.sort_values("entry_date"); x = g["cum_pnl"].values; n = len(x); n_eff = n / h
        if n_eff < 10: continue
        tv = truth[(truth.archetype == a_) & (truth.regime == reg) & (truth.h == h)].true_ev.iloc[0]; mean = x.mean()
        lo_b, hi_b = block_bootstrap_ci(x, 2 * h, 500, 0.10, rng); lo_e, hi_e = episode_cluster_ci(x, g["episode"].values.astype(int), 500, 0.10, rng); se = hac_se(x, h)
        half_b, half_e = 0.5 * (hi_b - lo_b), 0.5 * (hi_e - lo_e) if np.isfinite(lo_e) else np.nan; half_h = 1.6449 * se
        cells.append(dict(seed=seed, archetype=a_, regime=reg, h=h, n_eff=n_eff, episodes=g["episode"].nunique(), true=tv, mean=mean, half_hac=half_h, half_block=half_b, half_episode=half_e, half_hybrid=np.nanmax([half_b, half_e])))
V = pd.DataFrame(cells)
for mth in ("hac", "block", "episode", "hybrid"): V[f"cov_{mth}"] = (np.abs(V["mean"] - V["true"]) <= V[f"half_{mth}"])
fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
byh = V[V.n_eff >= 20].groupby("h")[[f"cov_{m_}" for m_ in ("hac", "block", "episode", "hybrid")]].mean()
for mth, col, ls_ in (("hac", "#9aa0a6", "--"), ("block", "#0EA5E9", "-"), ("episode", "#D4A017", "-"), ("hybrid", "#4F46E5", "-")):
    axs[0].plot(byh.index, byh[f"cov_{mth}"], "o" + ls_, ms=4, color=col, lw=1.8 if mth == "hybrid" else 1.2, label=mth)
axs[0].axhline(0.9, color="#333", ls=":", lw=0.9); axs[0].set_xticks(H); axs[0].set_xlabel("h"); axs[0].set_ylabel("coverage of a nominal 90% interval"); axs[0].set_title("Coverage by horizon (cells with n_eff ≥ 20, 8 seeds, true labels)"); axs[0].legend(frameon=False, fontsize=8); axs[0].set_ylim(0.6, 1.02)
V["ep_bin"] = pd.cut(V.episodes, [0, 10, 15, 20, 30, 100], labels=["≤10", "11–15", "16–20", "21–30", ">30"])
bye = V[V.n_eff >= 20].groupby("ep_bin", observed=True)[[f"cov_{m_}" for m_ in ("block", "hybrid")]].mean()
xx = np.arange(len(bye)); axs[1].bar(xx - 0.18, bye["cov_block"], 0.36, color="#0EA5E9", label="block only"); axs[1].bar(xx + 0.18, bye["cov_hybrid"], 0.36, color="#4F46E5", label="hybrid")
axs[1].axhline(0.9, color="#333", ls=":", lw=0.9); axs[1].set_xticks(xx); axs[1].set_xticklabels(bye.index); axs[1].set_xlabel("episodes behind the cell"); axs[1].set_ylabel("coverage"); axs[1].set_title("Coverage by episode count: where the episode bootstrap matters"); axs[1].legend(frameon=False, fontsize=8); axs[1].set_ylim(0.6, 1.02)
figs["cov"] = save(fig, "la_cov.png")
cov_summary = V[V.n_eff >= 20][[f"cov_{m_}" for m_ in ("hac", "block", "episode", "hybrid")]].mean()
width_summary = (V[V.n_eff >= 20][[f"half_{m_}" for m_ in ("hac", "block", "episode", "hybrid")]].mean() / V[V.n_eff >= 20]["half_block"].mean())

# ---- F4 validation scatter (hybrid) — the figure from the framework page
fig, ax = plt.subplots(figsize=(5.2, 4.4))
Vv = V[V.n_eff >= 20]
for reg, g in Vv.groupby("regime"): ax.errorbar(g["true"], g["mean"], yerr=g["half_hybrid"], fmt="o", ms=3, color=C[reg], alpha=0.5, lw=0.8, label=f"{reg}")
lim = [min(Vv["true"].min(), Vv["mean"].min()) - 1, max(Vv["true"].max(), Vv["mean"].max()) + 1]; ax.plot(lim, lim, "k-", lw=0.8)
ax.set_xlabel("analytic truth"); ax.set_ylabel("ladder estimate ± 90% interval"); ax.set_title(f"{len(Vv)} cells, 8 seeds · coverage {cov_summary['cov_hybrid']:.0%}"); ax.legend(frameon=False, fontsize=7)
figs["valid"] = save(fig, "la_valid.png")

# ---- F5 shrinkage sweep
rows = []
for k in (0, 5, 10, 20, 40, 80):
    for ln, cum in (("true", cum_t), ("estimated", cum_e)):
        wf = evaluate.walk_forward(cum, n_splits=4, kappa=k); w = wf[wf.h <= 10]
        rows.append(dict(kappa=k, labels=ln, improvement=w.improvement.mean(), improvement_min=w.improvement.min(), slope_min=w.calib_slope.min(), slope_max=w.calib_slope.max()))
ksw = pd.DataFrame(rows)
fig, axs = plt.subplots(1, 2, figsize=(10, 3))
for ln, col in (("true", "#222"), ("estimated", "#4F46E5")):
    g = ksw[ksw.labels == ln]; axs[0].plot(g.kappa, g.improvement, "o-", ms=4, color=col, label=f"{ln} labels"); axs[1].fill_between(g.kappa, g.slope_min, g.slope_max, color=col, alpha=0.15); axs[1].plot(g.kappa, g.slope_min, color=col, lw=0.8); axs[1].plot(g.kappa, g.slope_max, color=col, lw=0.8, label=f"{ln} labels (range over cells)")
axs[0].set_xlabel("κ"); axs[0].set_ylabel("mean OOS MSE improvement, h ≤ 10"); axs[0].set_title("Shrinkage strength against out-of-sample improvement"); axs[0].legend(frameon=False, fontsize=8)
axs[1].axhspan(0.5, 1.5, color="#2EAE7A", alpha=0.08); axs[1].axhline(1, color="#333", lw=0.7); axs[1].set_xlabel("κ"); axs[1].set_ylabel("calibration slope"); axs[1].set_title("…and against calibration (gate band shaded)"); axs[1].legend(frameon=False, fontsize=8)
figs["kappa"] = save(fig, "la_kappa.png")

# ---- F6 ladder, increments, persistence split (rr_25d, estimated labels)
report.plot_ladder(lad_e, G, str(ROOT / "docs/src/_fig/la_ladder.png")); figs["ladder"] = str(ROOT / "docs/src/_fig/la_ladder.png")
inc = ladder.increments(lad_e); report.plot_increments(inc, G, str(ROOT / "docs/src/_fig/la_inc.png")); figs["inc"] = str(ROOT / "docs/src/_fig/la_inc.png")
sp = ladder.persistence_split(td, lab_est, horizons=H); spg = sp[(sp.archetype == "rr_25d")]
fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
for s_ in S6:
    g = spg[spg.regime == s_].sort_values("h")
    if len(g): axs[0].plot(g.h, g.q_broke, "o-", ms=3.5, color=C[s_], label=s_)
axs[0].set_xticks(H); axs[0].set_xlabel("h"); axs[0].set_ylabel("q = P(entry state broke by h)"); axs[0].set_title("Break frequency by entry state"); axs[0].legend(frameon=False, fontsize=7, ncol=2)
g = spg[(spg.regime == "agitated")].sort_values("h")
axs[1].plot(g.h, g.ev_stayed, "o-", ms=3.5, color="#2EAE7A", label="EV | state persisted"); axs[1].plot(g.h, g.ev_broke, "o-", ms=3.5, color="#E8743B", label="EV | state broke"); axs[1].plot(g.h, g.ev, "o-", ms=3.5, color="#222", label="EV (all)"); axs[1].plot(g.h, (1 - g.q_broke) * g.ev_stayed + g.q_broke * g.ev_broke, "x", ms=7, color="#4F46E5", label="(1−q)·stayed + q·broke")
axs[1].axhline(0, color="#333", lw=0.7); axs[1].set_xticks(H); axs[1].set_xlabel("h"); axs[1].set_ylabel("rr_25d cumulative EV, agitated entries"); axs[1].set_title("The split and its identity (crosses land on the dots)"); axs[1].legend(frameon=False, fontsize=7)
figs["split"] = save(fig, "la_split.png")

# ---- combinations and tenors: legs in vega units for three tenors
legs_frames = []
for T in (5, 21, 63):
    tdv = combine.to_vega_units(schema.coerce(synth.simulate_trades(m5, archetypes=combine.BASE_LEGS, tenor_days=T, seed=100 + T)))
    legs_frames += [tdv] + [combine.combine(tdv, w_, n_) for n_, w_ in combine.STANDARD_COMBOS.items()]
tdl = pd.concat(legs_frames)
cum_l = ladder.attach_entry_labels(ladder.cumulative(tdl, (1, 3, 5, 10, 20, 40, 60)), lab_true); lad_l = ladder.ladder(cum_l, lab_true, ci="hac")
report.plot_tenor_curve(lad_l, "EURUSD", "put_25d", 5, str(ROOT / "docs/src/_fig/la_tenor.png")); figs["tenor"] = str(ROOT / "docs/src/_fig/la_tenor.png")
# verification: package vs combination (synthetic rr_25d package from DEMO vs combine of legs) — in the synthetic world the package is generated independently, so show the exactness of combine on a leg-level identity instead
tdv21 = combine.to_vega_units(schema.coerce(synth.simulate_trades(m5, archetypes=combine.BASE_LEGS, tenor_days=21, seed=121)))
rr = combine.combine(tdv21, combine.STANDARD_COMBOS["rr_25d"], "rr_25d_check")
chk = rr.merge(tdv21[tdv21.archetype == "call_25d"][["entry_date", "age", "pnl"]].rename(columns={"pnl": "call"}), on=["entry_date", "age"]).merge(tdv21[tdv21.archetype == "put_25d"][["entry_date", "age", "pnl"]].rename(columns={"pnl": "put"}), on=["entry_date", "age"])
max_gap = float((chk["pnl"] - (chk["call"] - chk["put"])).abs().max())
cands = {**combine.STANDARD_COMBOS, "atm vega": {"straddle_atm": 1.0}, "put skew (25p − ATM)": {"put_25d": 1.0, "straddle_atm": -1.0}, "call skew (25c − ATM)": {"call_25d": 1.0, "straddle_atm": -1.0}, "10d put wing (10p − 25p)": {"put_10d": 1.0, "put_25d": -1.0}}
scr = combine.screen(lad_l, cands, "normalising", 5); scr = scr[scr.tenor_days == 21].reset_index(drop=True)

# ---- the card for the live demos: raw means, n_eff, all means per (state, h) for rr_25d 21d, estimated labels
rr21 = lad_e[(lad_e.archetype == "rr_25d") & (lad_e.tenor_days == 21)]
card_rows = {s_: {int(r.h): dict(mean=float(r["mean"]), n_eff=float(r.n_eff), lo=float(r.ci_lo), hi=float(r.ci_hi)) for _, r in rr21[rr21.regime == s_].iterrows()} for s_ in rr21.regime.unique()}
legs_ev = {}
for (a_, T), g in lad_l[lad_l.archetype.isin(combine.BASE_LEGS)].groupby(["archetype", "tenor_days"]):
    legs_ev[f"{a_}|{T}"] = {s_: {int(r.h): round(float(r["mean"]), 4) for _, r in gg.iterrows()} for s_, gg in g.groupby("regime")}
data = {"states": [s_ for s_ in S6 if s_ in card_rows], "colours": {s_: C[s_] for s_ in S}, "card": card_rows, "hs": list(H) + [21],
        "legs": legs_ev, "leg_names": list(combine.BASE_LEGS), "combos": combine.STANDARD_COMBOS,
        "valid": [dict(h=int(r.h), ep=int(r.episodes), neff=round(float(r.n_eff), 1), err=round(float(abs(r["mean"] - r["true"])), 4), hac=round(float(r.half_hac), 4), block=round(float(r.half_block), 4), episode=(round(float(r.half_episode), 4) if np.isfinite(r.half_episode) else None)) for _, r in V.iterrows()]}

fmt_scr = {"ev_linear": lambda v: f"{v:+.2f}", "episodes": lambda v: f"{int(v)}", "tenor_days": str}
fmt_k = {"kappa": str, "improvement": lambda v: f"{v:+.3f}", "improvement_min": lambda v: f"{v:+.3f}", "slope_min": lambda v: f"{v:.2f}", "slope_max": lambda v: f"{v:.2f}"}
wf_e = evaluate.walk_forward(cum_e, n_splits=4); g3e = gates.gate_phase3(wf_e, gc); wf_t = evaluate.walk_forward(cum_t, n_splits=4); g3t = gates.gate_phase3(wf_t, gc)
g2e = gates.gate_phase2(lad_e, gc); g2t = gates.gate_phase2(lad_t, gc)
fmt_wf = {"h": str, "n_test": str, "mse_uncond": lambda v: f"{v:.1f}", "mse_regime": lambda v: f"{v:.1f}", "improvement": lambda v: f"{v:+.3f}", "rank_ic": lambda v: f"{v:.3f}", "calib_slope": lambda v: f"{v:.2f}"}

body = f"""
<!-- 1 -->
<h2><span class="n">1</span>The estimand {BUILT}</h2>
<div class="eq">EV(h | s)  =  mean over past entries made on days labelled s  of  [ cumulative net P&amp;L from entry to day h ]
               held h days, no exit rule, hedged exactly as the backtester hedges, cash per unit of vega at inception</div>
<p>Everything on the card is this quantity or a statistic of the same sample: its quantiles, the probability it is positive, its expected shortfall, the count of independent episodes behind it. Three commitments follow from measuring it this way rather than estimating a daily rate per regime and adding days up.</p>
<ol>
<li><b>The exposure is right by construction.</b> Every observation starts from a fresh unit and ages through the backtester's own repricing and hedging; nothing is rescaled by a Greek that was true on a different day.</li>
<li><b>Transitions inside the window are already in the outcomes.</b> A unit entered in carry that lived through a break to stressed on day 4 contributes exactly what it earned; no switch term is added and none can be double-counted. (That is also why the transition matrix is not an input to any number here.)</li>
<li><b>The distribution is measured, not simulated.</b> Tails and probability of profit are empirical quantiles of the same entries, never shrunk.</li>
</ol>
<div class="box"><b>Read it as a market-maker.</b> The base legs are the components a smile-vega-neutral book decomposes into; the ladder is the marginal expected earn of carrying one unit of a component for the next h days from today's state — a number to tilt composition with, not a trade signal. Short horizons are primary for a book that is re-cut continuously; the increments between horizons (Section 8) are how the number applies to inventory that is already days old.</div>

<!-- 2 -->
<h2><span class="n">2</span>From trade-day rows to a rung {BUILT}</h2>
<p>The input is one row per trade per day: pair, archetype, tenor, entry date, date, age, net P&amp;L, and where available the components (trade, delta hedge, vega hedge) and daily Greeks. <code>cumulative</code> sums each trade's rows to every horizon h ≤ tenor and to expiry; <code>attach_entry_labels</code> joins the point-in-time label on the <i>entry</i> date and drops (and counts) entries without one. A rung is the set of cumulative outcomes of all entries labelled s, at one h, in one (pair, archetype, tenor).</p>
{figure(figs['build'], 1, "Left: one trade's daily rows summed, with the rungs marked. Centre: every straddle entered in stressed over five years, and their mean path — the ladder is that black line read at the rungs. Right: the outcome distribution at h = 5 for carry entries against stressed entries; the quantiles, P(profit) and ES on the card are read off this.")}
<p>Horizons are capped at the component's tenor and the to-expiry rung is always added, so a 1W component has rungs 1 / 3 / 5 and a 1Y component gains 40 and 60. Entries without a label — the warm-up of the features, a gap in the surface — are dropped from every rung and reported as a count, never filled.</p>

<!-- 3 -->
<h2><span class="n">3</span>Why the entries are not independent {BUILT}</h2>
<p>Entries are made every day and each lives h days, so consecutive entries share h − 1 of their days: the same daily shocks are in both outcomes. That is the <b>overlap</b> dependence and it has a known length, h. There is a second kind that outlasts it. Entries made inside the same regime episode share that episode's <i>fate</i> — when it ends and what it ends into — and for a quantity conditioned on the entry state, that is most of what varies between episodes. Two entries ten days apart in one long agitated episode are not independent at h = 5 even though their windows never touch.</p>
{figure(figs['dep'], 2, "Correlation of 5-day outcomes between two straddle entries L days apart, split by whether they fall in the same regime episode. Across episodes the correlation is the overlap and dies at h. Within an episode it stays well above zero for weeks. An interval sized for the overlap alone is too narrow for exactly the cells with few episodes.", maxw=640)}
<div class="eq">n_eff  ≈  n · min( 1 , entry spacing / h )          episodes  =  contiguous runs of the entry state in the label history</div>
<p>n_eff discounts the overlap; episodes count the fates. Both are on the card, because a cell with n_eff 40 from 4 episodes and a cell with n_eff 40 from 25 episodes are not equally trustworthy, and no single number says so.</p>

<!-- 4 -->
<h2><span class="n">4</span>Intervals {BUILT}</h2>
<p>Four ways to put a 90% interval on a rung's mean, and what the synthetic truth says about each:</p>
<ol>
<li><b>Newey–West (HAC)</b> with bandwidth h: a standard error that allows for autocorrelation up to lag h. Fast; exactly the overlap and nothing else.</li>
<li><b>Circular block bootstrap</b> with blocks of 2h: resample the ordered outcomes in blocks long enough to keep the overlap dependence inside a block.</li>
<li><b>Episode-cluster bootstrap</b>: resample whole regime episodes of entries with replacement — the unit the fate dependence lives in.</li>
<li><b>The hybrid</b> (default, <code>ci: boot</code>): the <i>wider</i> of 2 and 3, centred on the mean.</li>
</ol>
{figure(figs['cov'], 3, f"Coverage of a nominal 90% interval with true labels across eight synthetic histories, cells with n_eff ≥ 20. Left, by horizon: HAC and the block bootstrap fall to {byh['cov_block'].loc[5]:.0%}–{byh['cov_block'].loc[10]:.0%} at h = 5–10; the hybrid recovers to {byh['cov_hybrid'].loc[5]:.0%}–{byh['cov_hybrid'].loc[10]:.0%} there and sits at {byh['cov_hybrid'].loc[1]:.0%} at h = 1 (the h = 20 cells are few and over-covered). Right, by the number of episodes behind the cell: the hybrid's gain is largest where episodes are few. Overall: HAC {cov_summary['cov_hac']:.0%}, block {cov_summary['cov_block']:.0%}, episode {cov_summary['cov_episode']:.0%}, hybrid {cov_summary['cov_hybrid']:.0%}; the hybrid's intervals are {width_summary['half_hybrid']:.0%} the width of the block bootstrap's. The residual shortfall at h = 5–10 is what a dozen episodes can tell you: no resampling scheme invents the episodes that did not happen, which is why the episode count is on the card next to the interval.")}
{figure(figs['valid'], 4, "Every cell's estimate and hybrid interval against the analytic truth. This is the evidence that the statistics are implemented correctly; it says nothing about real markets.", maxw=520)}

<!-- 5 -->
<h2><span class="n">5</span>Shrinkage {BUILT}</h2>
<p>A rung with little evidence is pulled toward the unconditional mean at the same horizon — the ALL row — with weight n_eff / (n_eff + κ). The tails are never shrunk: a quantile of a thin sample is reported as a quantile of a thin sample, with its episode count beside it.</p>
<div class="eq">mean_shrunk  =  w · mean_s  +  (1 − w) · mean_ALL          w = n_eff / (n_eff + κ),   κ = 20 by default</div>
{figure(figs['kappa'], 5, "The walk-forward benchmark as κ varies, on the synthetic trades with true and estimated labels. Left: mean out-of-sample MSE improvement over the unconditional mean at h ≤ 10. Right: the range of the calibration slope over cells (the gate asks for 0.5–1.5). On this world raw means are hard to beat — the states are real by construction — and strong shrinkage over-flattens the long-horizon cells. On real data κ is to be chosen by leave-one-fold-out MSE (D7), and the evaluation can score either the raw or the shrunk estimate (config `eval.kappa`).")}
{table(ksw, fmt_k)}

<!-- 6 -->
<h2><span class="n">6</span>The ladder {BUILT}</h2>
{figure(figs['ladder'], 6, "The risk reversal at 1M with estimated labels: one line per entry state, shaded 90% hybrid intervals. The lines fan out from zero by construction and the intervals widen with h as the overlap grows.")}

<!-- 7 -->
<h2><span class="n">7</span>Increments: the aged unit {BUILT}</h2>
<p>The difference between consecutive rungs, divided by the days between them, is the expected earn <i>per day</i> of a unit that is already h days old over its next few days. A market-making book holds strikes of every age, so this row maps onto actual inventory where the ladder maps onto a fresh unit; and where a component stops earning per unit of risk, the roll point falls out of the chart.</p>
<div class="eq">increment(h₁→h₂)  =  ( EV(h₂) − EV(h₁) ) / (h₂ − h₁)          Σ increments · days  =  EV(T) exactly</div>
{figure(figs['inc'], 7, "Increments for the same component: earn per day in each age bucket, by entry state. The to-expiry bucket of the risk reversal is where the remaining-tenor effect of the age-dependence question lives.")}

<!-- 8 -->
<h2><span class="n">8</span>Regime dependence: the persistence split {BUILT}</h2>
<p>Each rung's outcomes are split by whether the entry state was still in force on every day to h (<i>persisted</i>) or had left at least once (<i>broke</i>), with the break frequency q. It is a decomposition for explanation, never a forecast: it tells apart a component that is hurt by a break from one that is insurance against it, which the EV column cannot. The identity holds exactly in sample; the two parts are never multiplied together and never shown as a ratio.</p>
<div class="eq">(1 − q) · EV_persisted  +  q · EV_broke  =  EV</div>
{figure(figs['split'], 8, "Left: break frequency by entry state and horizon — nearly every agitated entry has seen another state by expiry. Right: for agitated entries of the risk reversal, the two conditional means, the overall mean, and the identity's reconstruction (crosses on dots).")}

<!-- 9 -->
<h2><span class="n">9</span>Base legs, combinations and scaling {BUILT}</h2>
<p>The backtester runs five <b>base legs</b> per tenor — the ATM straddle, the 25d put and call, the 10d put and call — each hedged leg by leg. Because hedging is per leg and P&amp;L is per leg per day, the daily P&amp;L of any <b>combination</b> of legs entered on the same date is exactly the weighted sum of its legs', hedges included; a spread's ladder is the ladder of that summed series, with no new backtest run for any combination anyone cares to name. Rows are matched on (pair, tenor, entry date, age), so a combination row exists only where every leg has one.</p>
<p><b>Scaling.</b> Each base leg's P&amp;L is divided by its own vega at age 1 — the strategy's net vega excluding the vega-hedge leg, which is the leg's inception vega (from day 2 the daily vega hedge takes it to ~0, so inception is the only meaningful normaliser). Weights are then <i>vega units</i>: a zero-sum weight vector is smile-vega-neutral by construction (risk reversal = +1 call − 1 put; fly = +½ put + ½ call − 1 ATM) and a non-zero sum carries net vega, which is how a market-making book is decomposed. Packages the source also runs as packages are reconciled against the combination of their legs at WU-03 and are never themselves rescaled.</p>
<div class="eq">EV_combo(h | s)  =  Σ_i w_i · EV_leg i(h | s)          exact for the mean (linearity); intervals, quantiles and ES need the combined rows</div>
<p>On the synthetic legs the combination reproduces the leg-level identity to machine precision (max |rr − (call − put)| = {max_gap:.1e}). The linear EV is the cheap first pass over many candidates (<code>screen</code>); distributions are then computed for a shortlist.</p>
{table(scr, fmt_scr)}
<p class="dim">The screen at 1M, normalising entries, h = 5, synthetic legs in vega units. On this world the normalising state punishes owning vega and pays for owning smile against it — the generating tables say so; it is not a finding about markets.</p>

<!-- 10 -->
<h2><span class="n">10</span>Tenors {BUILT}</h2>
<p>Any major tenor from 1W to 1Y is a separate component; the ladder is per (pair, leg or combination, tenor) and horizons are capped at tenor. At a fixed short horizon, EV across tenors is a <b>tenor curve</b> per component — which tenor's skew is best paid from here — and it makes the remaining-tenor diagnostic of the age-dependence question directly testable.</p>
{figure(figs['tenor'], 9, "The 25d put at h = 5 per unit vega across 1W, 1M and 3M, one line per entry state. Phase 3 runs three tenors by default (config); all are available.", maxw=640)}

<!-- 11 -->
<h2><span class="n">11</span>What the card shows, and the gates it must pass {BUILT}</h2>
<p>Per component, per pair and tenor, for today's state: the rungs with EV, 90% interval, P(profit), ES 5%, n_eff and episodes; the persistence split at the longest rung; where it earns (the extreme increments); the state profile, tags, the shock score and the transition fan. Two pre-registered gates decide whether any of it is shown to a trader:</p>
<ul>
<li><b>Gate 2 · separation.</b> At h ≤ 10, the best and worst entry states differ by more than two standard errors in at least 75% of (component, h) cells. Synthetic: true labels {'pass' if g2t['pass'] else 'fail'} ({g2t['fraction_separated']:.0%} of cells), estimated labels {'pass' if g2e['pass'] else 'fail'} ({g2e['fraction_separated']:.0%}).</li>
<li><b>Gate 3 · go/no-go.</b> Walk-forward by entry date (four test windows after a 40% training block, purged by h): the state-conditional prediction must beat the unconditional mean out of sample at every h ≤ 10 — positive MSE improvement, positive rank IC, calibration slope in [0.5, 1.5] — and (WU-11) beat a regularised model on the continuous features. Synthetic: true labels {'pass' if g3t['pass'] else 'fail'}, estimated labels {'pass' if g3e['pass'] else 'fail'}.</li>
</ul>
{table(wf_e[wf_e.h <= 10][['archetype', 'h', 'n_test', 'improvement', 'rank_ic', 'calib_slope']], fmt_wf)}
<p class="dim">The walk-forward table with estimated labels, h ≤ 10. The labeller, not the ladder, is where the synthetic world fails Gate 3 — the same cells pass with true labels — which is why Gate 2 comes first.</p>

<!-- 12 -->
<h2><span class="n">12</span>Live · shrinkage, intervals and a combination builder {BUILT}</h2>
<div class="demo"><div class="hd">Live · κ on the risk reversal's ladder</div>
<p class="dim" style="margin:0 0 8px">The 1M risk reversal with estimated labels. Move κ and watch each state's rung pulled toward the ALL row (grey dashed) by n_eff / (n_eff + κ); thin cells move most. The faint dotted lines are the raw means (κ = 0).</p>
<div class="ctl"><div><label>κ: <b id="kv">20</b></label><input type="range" id="k" min="0" max="200" step="5" value="20"></div></div>
<canvas id="cv_k" width="1600" height="420"></canvas>
</div>
<div class="demo"><div class="hd">Live · which interval, and when it matters</div>
<p class="dim" style="margin:0 0 8px">The validation cells (eight synthetic histories, true labels): coverage of a nominal 90% interval by method, for the horizons and the minimum episode count you choose. The hybrid is the wider of block and episode per cell.</p>
<div class="ctl"><div><label>horizons</label><select id="v_h"><option value="all">all</option><option value="1">1</option><option value="3">3</option><option value="5" selected>5</option><option value="10">10</option><option value="20">20</option></select></div>
<div><label>minimum episodes behind the cell: <b id="v_ep_v">1</b></label><input type="range" id="v_ep" min="1" max="30" step="1" value="1"></div>
<div><label>maximum episodes: <b id="v_ep2_v">100</b></label><input type="range" id="v_ep2" min="5" max="100" step="5" value="100"></div></div>
<div class="stat"><div>cells<b id="v_n">—</b></div><div>HAC<b id="v_hac">—</b></div><div>block<b id="v_block">—</b></div><div>episode<b id="v_epi">—</b></div><div>hybrid<b id="v_hyb">—</b></div><div>hybrid width / block width<b id="v_w">—</b></div></div>
</div>
<div class="demo"><div class="hd">Live · build a combination in vega units</div>
<p class="dim" style="margin:0 0 8px">Weights on the five base legs; the linear EV per entry state at the chosen tenor and horizon, from the synthetic leg ladders (true labels). The sum of the weights is the net vega: zero means smile-vega-neutral.</p>
<div class="ctl">
  {''.join(f'<div><label>{l}: <b id="w_{l}_v">{w0}</b></label><input type="range" id="w_{l}" min="-2" max="2" step="0.25" value="{w0}"></div>' for l, w0 in zip(combine.BASE_LEGS, (0, -1, 1, 0, 0)))}
  <div><label>tenor</label><select id="c_T"><option value="5">1W</option><option value="21" selected>1M</option><option value="63">3M</option></select></div>
  <div><label>horizon</label><select id="c_h"><option>1</option><option>3</option><option selected>5</option><option>10</option><option>20</option></select></div>
  <div><label>preset</label><select id="c_pre"><option value="">—</option>{''.join(f'<option value="{n_}">{n_}</option>' for n_ in combine.STANDARD_COMBOS)}<option value="atm">atm vega</option></select></div>
</div>
<div class="stat"><div>net vega (Σ w)<b id="c_net">—</b></div><div>smile-vega-neutral?<b id="c_neu">—</b></div></div>
<canvas id="cv_c" width="1600" height="320"></canvas>
</div>

<!-- 13 -->
<h2><span class="n">13</span>On real data {BUILT} <span class="dim">WU-02, WU-03, WU-10, WU-11</span></h2>
<ol>
<li><b>The vertical slice first</b> (WU-02): one pair, the straddle, the ALL row at h = 5, reconciled against a direct pandas computation for one month of entries. Everything else is this, repeated.</li>
<li><b>Legs before packages</b> (WU-03): confirm per-day vega is available for the inception normaliser; reconcile the source's own RR and fly packages against <code>combine</code> of the legs — correlation ~1 and a stable ratio — before any combination is quoted.</li>
<li><b>Intervals: hybrid, always.</b> HAC is for speed in development. Read the episode column before the interval.</li>
<li><b>κ by leave-one-fold-out MSE</b>, and score the walk-forward at the κ the card shows (<code>eval.kappa</code>).</li>
<li><b>Gate 3 against layer 1 as well as layer 0</b> (WU-11): the ridge model on the continuous features is the benchmark that decides whether discretising into states adds anything. If it does not, the states become presentation language over a continuous model — an acceptable outcome, stated as such.</li>
<li><b>Failure modes.</b> Entries dropped for missing labels concentrated in one period (the feature warm-up or a data gap — check before Gate 2); a persistence split with q ≈ 1 at h = 5 (the labeller is switching faster than the horizon — back to the switch budget); increments that change sign between adjacent buckets in every state (an attribution artefact at drifted strikes, the age-dependence question).</li>
</ol>

<!-- 14 -->
<h2><span class="n">14</span>Code map</h2>
<table><tr><th>function</th><th>does</th></tr>
<tr><td><code>ladder.cumulative(td, horizons, include_expiry)</code></td><td>trade-day rows → cumulative P&amp;L at each rung</td></tr>
<tr><td><code>ladder.attach_entry_labels(cum, labels)</code></td><td>join the entry-date label; count the unlabelled</td></tr>
<tr><td><code>ladder.ladder(cum_lab, labels, alpha, ci, n_boot)</code></td><td>the ladder: mean, interval (hybrid or HAC), quantiles, P(profit), ES, n_eff, episodes; plus the ALL row</td></tr>
<tr><td><code>ladder.hac_se · block_bootstrap_ci · episode_cluster_ci · episode_ids</code></td><td>the interval machinery</td></tr>
<tr><td><code>ladder.shrink(lad, kappa)</code></td><td>mean_shrunk toward the ALL row at the same h</td></tr>
<tr><td><code>ladder.increments(lad) · persistence_split(td, labels)</code></td><td>the aged unit; the regime-dependence identity</td></tr>
<tr><td><code>combine.to_vega_units · combine · ev_linear · screen</code></td><td>scaling, exact aggregation, linear EV, the candidate screen</td></tr>
<tr><td><code>evaluate.walk_forward(cum_lab, n_splits, kappa)</code></td><td>the out-of-sample benchmark per horizon</td></tr>
<tr><td><code>gates.gate_phase2 · gate_phase3</code></td><td>the pre-registered tests, thresholds from <code>configs/gates.yaml</code></td></tr>
<tr><td><code>report.card_md · plot_ladder · plot_increments · plot_tenor_curve</code></td><td>the card and its plots</td></tr>
<tr><td><code>python -m regime_ladder ladder · evaluate · validate · demo</code></td><td>the CLI</td></tr>
</table>
"""

js = r"""
const S=D.states,COL=D.colours;
// ---- kappa
function runK(){const k=+document.getElementById('k').value;kv.textContent=k;const hs=D.hs;const all={};hs.forEach(h=>{let s=0,n=0;S.forEach(st=>{const c=D.card[st]&&D.card[st][h];if(c){s+=c.mean*c.n_eff;n+=c.n_eff;}});all[h]=n?s/n:0;});
  const series=S.filter(st=>D.card[st]).map(st=>({name:st,color:COL[st],y:hs.map(h=>{const c=D.card[st][h];if(!c)return null;const w=c.n_eff/(c.n_eff+k);return w*c.mean+(1-w)*all[h];})}));
  S.filter(st=>D.card[st]).forEach(st=>series.push({color:COL[st],lw:0.8,dash:[2,4],y:hs.map(h=>D.card[st][h]?D.card[st][h].mean:null)}));
  series.push({name:"ALL",color:"#6B7280",y:hs.map(h=>all[h]),dash:[6,4]});
  lines(document.getElementById('cv_k'),series,{pad:[16,10,40,60],xlabels:hs.map((h,i)=>[i,h+"d"])});}
document.getElementById('k').addEventListener('input',runK);runK();
// ---- intervals
function runV(){const hsel=document.getElementById('v_h').value,ep=+v_ep.value,ep2=+v_ep2.value;v_ep_v.textContent=ep;v_ep2_v.textContent=ep2;
  const cells=D.valid.filter(c=>(hsel==="all"||c.h===+hsel)&&c.ep>=ep&&c.ep<=ep2&&c.neff>=20);v_n.textContent=cells.length;if(!cells.length)return;
  const cov=f=>cells.filter(c=>{const hw=f(c);return hw!=null&&c.err<=hw;}).length/cells.length;const hyb=c=>Math.max(c.block,c.episode==null?0:c.episode);
  v_hac.textContent=(100*cov(c=>c.hac)).toFixed(0)+"%";v_block.textContent=(100*cov(c=>c.block)).toFixed(0)+"%";v_epi.textContent=(100*cov(c=>c.episode)).toFixed(0)+"%";v_hyb.textContent=(100*cov(hyb)).toFixed(0)+"%";
  const wb=cells.reduce((a,c)=>a+c.block,0),wh=cells.reduce((a,c)=>a+hyb(c),0);v_w.textContent=(wh/wb).toFixed(2);}
['v_h','v_ep','v_ep2'].forEach(id=>document.getElementById(id).addEventListener('input',runV));runV();
// ---- combination builder
const LEGS=D.leg_names;
function setPreset(){const p=document.getElementById('c_pre').value;if(!p)return;const w=p==="atm"?{straddle_atm:1}:D.combos[p];LEGS.forEach(l=>{document.getElementById('w_'+l).value=w[l]||0;});}
function runC(){const T=document.getElementById('c_T').value,h=+document.getElementById('c_h').value;const w={};let net=0;LEGS.forEach(l=>{w[l]=+document.getElementById('w_'+l).value;document.getElementById('w_'+l+'_v').textContent=w[l];net+=w[l];});
  c_net.textContent=net.toFixed(2);c_neu.innerHTML=Math.abs(net)<1e-9?'<span class="ok">yes</span>':'<span class="no">no — carries net vega</span>';
  const states=S.filter(s=>LEGS.every(l=>D.legs[l+"|"+T]&&D.legs[l+"|"+T][s]&&D.legs[l+"|"+T][s][h]!=null));
  const ev=states.map(s=>LEGS.reduce((a,l)=>a+w[l]*D.legs[l+"|"+T][s][h],0));
  const cv=document.getElementById('cv_c'),c=cv.getContext('2d'),W=cv.width,Hh=cv.height;c.clearRect(0,0,W,Hh);const mx=Math.max(1e-6,...ev.map(Math.abs));const bw=(W-80)/states.length;const Y=v=>Hh/2-(v/mx)*(Hh/2-40);
  c.strokeStyle="#333";c.beginPath();c.moveTo(40,Y(0));c.lineTo(W-20,Y(0));c.stroke();
  states.forEach((s,i)=>{const x=40+i*bw+bw*0.15;c.fillStyle=COL[s];c.fillRect(x,Math.min(Y(0),Y(ev[i])),bw*0.7,Math.abs(Y(ev[i])-Y(0)));c.fillStyle="#333";c.font="600 15px Inter,system-ui";c.textAlign="center";c.fillText(s,x+bw*0.35,Hh-10);c.fillText((ev[i]>=0?"+":"")+ev[i].toFixed(2),x+bw*0.35,ev[i]>=0?Y(ev[i])-8:Y(ev[i])+20);});
  c.textAlign="left";c.fillStyle="#666";c.font="13px Inter,system-ui";c.fillText("linear EV per unit vega, "+T+"d tenor, h = "+h+", by entry state",46,22);}
LEGS.forEach(l=>document.getElementById('w_'+l).addEventListener('input',()=>{document.getElementById('c_pre').value="";runC();}));['c_T','c_h'].forEach(id=>document.getElementById(id).addEventListener('input',runC));document.getElementById('c_pre').addEventListener('input',()=>{setPreset();runC();});runC();
"""

toc = [("1", "The estimand"), ("2", "From trade-day rows to a rung"), ("3", "Why entries are not independent"), ("4", "Intervals"), ("5", "Shrinkage"), ("6", "The ladder"), ("7", "Increments"),
       ("8", "The persistence split"), ("9", "Legs, combinations, scaling"), ("10", "Tenors"), ("11", "The card and the gates"), ("12", "Live · κ, intervals, combinations"), ("13", "On real data"), ("14", "Code map")]
out = page("ladder.html", "The horizon ladder and its intervals",
           "The product: what is estimated and why it is measured at fixed horizons from a fresh unit rather than rolled forward from a daily rate, the two kinds of dependence between entries and the interval that respects both, shrinkage and what it costs, the increments that map the number onto an aged book, the persistence split and its identity, exact aggregation of base legs into any combination in vega units, the tenor curve, and the two gates the card has to pass. Every figure is from synthetic data with a known answer.",
           toc, body, data, js)
print(out, cov_summary.round(3).to_dict(), max_gap)
