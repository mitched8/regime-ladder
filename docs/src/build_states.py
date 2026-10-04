"""Deep dive 1: states.html — States and the state finder."""
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from common import world, figure, save, table, page, COLOURS, BUILT, HYP, ROOT
import sys; sys.path.insert(0, str(ROOT))
from regime_ladder import synth, labels, features, ladder, schema

W = world(); m, comp, st, fwd5, cal, spec, est, F = W["m"], W["comp"], W["st"], W["fwd5"], W["cal"], W["spec"], W["est"], W["F"]
S = list(synth.STATES); S6 = list(labels.STATES6); C = COLOURS
x = np.arange(len(m)); seg = slice(1500, 2200)
ok = comp.notna()
agree = (est[ok].values == st[ok].values).mean()
sc = labels.ewma(comp, spec["halflife"]); ds = labels.direction_score(sc, spec["dir_window"])
lv = labels.level_labels(sc, spec["bounds"], spec["delta"]); dr = labels.direction_labels(ds, spec["d_down"], spec["d_up"], spec["delta_d"])
figs = {}

def ribbon(ax, series, y, h, seg):
    cols = [C.get(v, "#999") for v in series.values[seg]]
    ax.bar(x[seg], [h] * len(cols), bottom=y, width=1, color=cols, lw=0)

# ---- F1 features
fig, axs = plt.subplots(4, 1, figsize=(10, 5.2), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 0.5]})
for ax, col, nm in zip(axs[:3], ["vol_pct", "term_slope_pct", "rr_stress_pct"], ["ATM vol percentile", "term-slope percentile", "RR stress percentile"]):
    ax.plot(x[seg], F[col].values[seg], color="#333", lw=1); ax.set_ylabel(nm, fontsize=8); ax.set_ylim(0, 100)
ribbon(axs[3], st, 0, 1, seg); axs[3].set_yticks([]); axs[3].grid(False); axs[3].set_xlabel("trading day"); axs[3].set_title("true state", fontsize=8, loc="left")
axs[0].set_title("The three starting features over 700 days, with the true state below")
figs["features"] = save(fig, "st_features.png")

# ---- F2 the labeller step by step
fig, axs = plt.subplots(5, 1, figsize=(10, 7.2), sharex=True, gridspec_kw={"height_ratios": [2, 1.2, 0.5, 0.5, 1]})
axs[0].plot(x[seg], comp.values[seg], color="#bbb", lw=0.8, label="composite (equal-weight mean of the three)"); axs[0].plot(x[seg], sc.values[seg], color="#222", lw=1.3, label=f"EWMA, half-life {spec['halflife']}d")
for b in spec["bounds"]: axs[0].axhline(b, color="#888", ls="--", lw=0.8); axs[0].axhspan(b - spec["delta"], b + spec["delta"], color="#4F46E5", alpha=0.07)
axs[0].set_ylabel("score (0–100)"); axs[0].legend(frameon=False, fontsize=8, loc="upper left"); axs[0].set_title("1 · composite and its EWMA, with the two level boundaries and their hysteresis bands")
axs[1].plot(x[seg], ds.values[seg], color="#222", lw=1.1); axs[1].axhline(0, color="#333", lw=0.6)
for d in (spec["d_up"], spec["d_down"]): axs[1].axhline(d, color="#888", ls="--", lw=0.8)
axs[1].set_ylabel("slope / day"); axs[1].set_title("2 · direction score: 5-day slope of the EWMA, with its thresholds", fontsize=9)
lvcol = {"low": "#cbd5e1", "mid": "#94a3b8", "high": "#475569", "extreme": "#B91C1C"}; drcol = {"down": "#0EA5E9", "flat": "#e5e7eb", "up": "#2EAE7A"}
axs[2].bar(x[seg], 1, width=1, color=[lvcol[v] for v in lv.values[seg]], lw=0); axs[2].set_yticks([]); axs[2].grid(False); axs[2].set_title("3 · level band (low / mid / high)", fontsize=9, loc="left")
axs[3].bar(x[seg], 1, width=1, color=[drcol[v] for v in dr.values[seg]], lw=0); axs[3].set_yticks([]); axs[3].grid(False); axs[3].set_title("4 · direction band (down / flat / up)", fontsize=9, loc="left")
ribbon(axs[4], st, 1.1, 0.8, seg); ribbon(axs[4], est, 0.1, 0.8, seg)
axs[4].set_yticks([0.5, 1.5]); axs[4].set_yticklabels(["estimated", "true"]); axs[4].set_ylim(0, 2); axs[4].grid(False); axs[4].set_xlabel("trading day"); axs[4].set_title(f"5 · the partition table turns (level, direction, previous state) into a named state · agreement {agree:.0%}", fontsize=9, loc="left")
figs["steps"] = save(fig, "st_steps.png")

# ---- F3 kink vs step on a toy ramp + on the real composite
rng = np.random.default_rng(0)
s_toy = pd.Series(rng.uniform(0, 100, 4000)); y_toy = pd.Series(np.where(s_toy < 30, -5, np.where(s_toy < 70, 0, 5)) + rng.normal(0, 1.5, 4000))
step_b = labels.estimate_breaks(s_toy, y_toy, 2)
def kink_fit(s, y, grid=None):
    s, y = s.values, y.values; grid = np.quantile(s, np.linspace(0.08, 0.97, 30)); best, bs = np.inf, None
    for i, b1 in enumerate(grid):
        for b2 in grid[i + 1:]:
            if b2 - b1 < 10: continue
            X = np.column_stack([np.ones_like(s), s, np.maximum(s - b1, 0), np.maximum(s - b2, 0)]); beta, *_ = np.linalg.lstsq(X, y, rcond=None); v = ((y - X @ beta) ** 2).sum()
            if v < best: best, bs = v, (b1, b2, beta)
    return bs
kb1, kb2, kbeta = kink_fit(s_toy, y_toy)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
ax = axs[0]
bins = pd.cut(s_toy, np.arange(0, 101, 5)); mb = y_toy.groupby(bins, observed=True).mean(); ax.plot([b.mid for b in mb.index], mb.values, "o", ms=4, color="#333", label="binned mean of the outcome")
xx = np.linspace(0, 100, 200); ax.plot(xx, kbeta[0] + kbeta[1] * xx + kbeta[2] * np.maximum(xx - kb1, 0) + kbeta[3] * np.maximum(xx - kb2, 0), color="#E8743B", lw=1.4, label=f"kink fit: breaks at {kb1:.0f}, {kb2:.0f}")
for b in step_b: ax.axvline(b, color="#4F46E5", lw=1.4)
ax.text(step_b[0] + 1, 4.2, f"step fit: {step_b[0]:.0f}, {step_b[1]:.0f}", color="#4F46E5", fontsize=8); ax.set_xlabel("score"); ax.set_ylabel("outcome"); ax.set_title("A toy world: three bands with steps at 30 and 70"); ax.legend(frameon=False, fontsize=8, loc="upper left")
ax = axs[1]
by = pd.concat([sc.rename("sc"), st], axis=1).dropna(); order = ["carry", "settling", "rising", "agitated", "normalising", "stressed"]
ax.boxplot([by.loc[by.state_true == s, "sc"].values for s in order], tick_labels=order, vert=False, widths=0.6, showfliers=False, medianprops=dict(color="#222"), boxprops=dict(color="#555"), whiskerprops=dict(color="#999"), capprops=dict(color="#999"))
for b in spec["bounds"]: ax.axvline(b, color="#4F46E5", lw=1.4)
for b in (14.6, 51.5): ax.axvline(b, color="#E8743B", lw=1.1, ls="--")
pb = W["cal_pnl_only"]["specs"]["6"]["bounds"]
for b in pb: ax.axvline(b, color="#D4A017", lw=1.1, ls=":")
ax.set_xlabel("smoothed composite"); ax.set_title("The synthetic world: composite by true state")
ax.plot([], [], color="#4F46E5", lw=1.4, label=f"step on forward ATM: {spec['bounds'][0]:.0f} / {spec['bounds'][1]:.0f}"); ax.plot([], [], color="#E8743B", ls="--", label="kink on forward P&L: 15 / 51"); ax.plot([], [], color="#D4A017", ls=":", label=f"step on forward P&L: {pb[0]:.0f} / {pb[1]:.0f}"); ax.legend(frameon=False, fontsize=7.5, loc="lower right")
figs["kink"] = save(fig, "st_kink.png")

# ---- F4 the finder report: OOS R² by half-life × hysteresis for partition 6, and 3/6/6x bars
rep = cal["report"]
piv = rep[(rep.partition == "6")].groupby(["halflife", "delta"])["oos_r2"].max().unstack()
sw = rep[(rep.partition == "6")].groupby(["halflife", "delta"])["switches_per_year"].min().unstack()
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.2), gridspec_kw={"width_ratios": [1, 1, 1.1]})
im = axs[0].imshow(piv.values, cmap="Blues", aspect="auto"); axs[0].set_xticks(range(len(piv.columns))); axs[0].set_xticklabels(piv.columns); axs[0].set_yticks(range(len(piv.index))); axs[0].set_yticklabels(piv.index); axs[0].grid(False)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]): axs[0].text(j, i, f"{piv.values[i, j]:.3f}", ha="center", va="center", fontsize=8, color="#fff" if piv.values[i, j] > piv.values.max() - 0.01 else "#111")
axs[0].set_xlabel("hysteresis δ"); axs[0].set_ylabel("half-life"); axs[0].set_title("best OOS R² (partition 6)")
im2 = axs[1].imshow(sw.values, cmap="Oranges", aspect="auto"); axs[1].set_xticks(range(len(sw.columns))); axs[1].set_xticklabels(sw.columns); axs[1].set_yticks(range(len(sw.index))); axs[1].set_yticklabels(sw.index); axs[1].grid(False)
for i in range(sw.shape[0]):
    for j in range(sw.shape[1]): axs[1].text(j, i, f"{sw.values[i, j]:.0f}", ha="center", va="center", fontsize=8, color="#111")
axs[1].set_xlabel("hysteresis δ"); axs[1].set_title("switches per year"); axs[1].axhline(-0.5, color="none")
vals = [[W["cal_rv"]["r2"]["3"], W["cal_rv"]["r2"]["6"], W["cal_rv"]["r2"]["6x"]], [cal["r2"]["3"], cal["r2"]["6"], cal["r2"]["6x"]]]
xx = np.arange(2); w = 0.26
for j, (nm, col) in enumerate([("3: level only", "#9aa0a6"), ("6: level × direction", "#4F46E5"), ("6x: + extreme band", "#B91C1C")]):
    axs[2].bar(xx + (j - 1) * w, [v[j] for v in vals], w, color=col, label=nm)
    for i, v in enumerate(vals): axs[2].text(i + (j - 1) * w, v[j] + 0.004, f"{v[j]:.2f}", ha="center", fontsize=7)
axs[2].set_xticks(xx); axs[2].set_xticklabels(["target: fwd 21d RV", "target: fwd 5d P&L"], fontsize=8); axs[2].set_title("partition by target"); axs[2].legend(frameon=False, fontsize=7)
figs["report"] = save(fig, "st_report.png")

# ---- F5 extreme band
sc3 = labels.ewma(comp, cal["specs"]["6x"]["halflife"]); xb = cal["specs"]["6x"]["bounds"][2]
lab6x_raw = labels.two_axis_labels(labels.level_labels(sc3, cal["specs"]["6x"]["bounds"], cal["specs"]["6x"]["delta"]), labels.direction_labels(labels.direction_score(sc3, 5), cal["specs"]["6x"]["d_down"], cal["specs"]["6x"]["d_up"], cal["specs"]["6x"]["delta_d"]))
fig, axs = plt.subplots(2, 1, figsize=(10, 3.6), sharex=True, gridspec_kw={"height_ratios": [2, 0.6]})
axs[0].plot(x, sc3.values, color="#222", lw=0.7); axs[0].axhline(xb, color="#B91C1C", lw=1.2, ls="--"); axs[0].text(20, xb + 1.5, f"extreme bound = 97.5th percentile of the training window = {xb:.0f}", color="#B91C1C", fontsize=8)
for b in cal["specs"]["6x"]["bounds"][:2]: axs[0].axhline(b, color="#888", lw=0.8, ls="--")
axs[0].set_ylabel("smoothed composite"); axs[0].set_title(f"The extreme band over ten years: {cal['specs']['6x']['extreme_episodes']} episodes, {cal['specs']['6x']['extreme_days']} days → {'merged into stressed' if cal['specs']['6x']['extreme_merged'] else 'kept as a state'} (rule: ≥ 5 episodes)")
axs[1].bar(x, 1, width=1, color=[C.get(v, "#999") for v in lab6x_raw.values], lw=0); axs[1].set_yticks([]); axs[1].grid(False); axs[1].set_xlabel("trading day"); axs[1].set_title("labels with the band before the merge rule (dark red = extreme)", fontsize=8, loc="left")
figs["extreme"] = save(fig, "st_extreme.png")

# ---- F6 confusion matrix + detection lag
ct = pd.crosstab(st[ok], est[ok]).reindex(index=S, columns=S6, fill_value=0)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.1, 1]})
P = ct.div(ct.sum(axis=1).replace(0, 1), axis=0)
im = axs[0].imshow(P.values, cmap="Blues", vmin=0, vmax=1, aspect="auto"); axs[0].grid(False)
axs[0].set_xticks(range(len(S6))); axs[0].set_xticklabels(S6, rotation=25, fontsize=8); axs[0].set_yticks(range(len(S))); axs[0].set_yticklabels([f"{s} ({int(ct.loc[s].sum())})" for s in S], fontsize=8)
for i in range(len(S)):
    for j in range(len(S6)): axs[0].text(j, i, f"{P.values[i, j]:.0%}", ha="center", va="center", fontsize=7.5, color="#fff" if P.values[i, j] > 0.55 else "#111")
axs[0].set_xlabel("estimated"); axs[0].set_ylabel("true (days)"); axs[0].set_title("Row-normalised confusion: where each true state goes")
# detection lag
lags = {}
v = st.values; e = est.values
for t in range(1, len(v)):
    if v[t] != v[t - 1] and comp.notna().iloc[t]:
        s = v[t]; found = None
        for d in range(0, 31):
            if t + d < len(e) and e[t + d] == s: found = d; break
        lags.setdefault(s, []).append(found if found is not None else 31)
order2 = ["rising", "agitated", "stressed", "normalising", "settling", "carry"]
data_l = [lags.get(s, []) for s in order2]
ticks = [f"{s}\n{(np.array(lags.get(s, [31])) > 30).mean():.0%} never" for s in order2]
bp = axs[1].boxplot(data_l, tick_labels=ticks, showfliers=False, widths=0.55, medianprops=dict(color="#222"))
axs[1].tick_params(axis="x", labelsize=7.5)
axs[1].set_ylabel("days from true onset to first matching label"); axs[1].set_ylim(0, 33); axs[1].set_title("Detection lag by state (31 = not seen within 30 days)")
figs["confusion"] = save(fig, "st_confusion.png")
lag_summary = {s: (float(np.median(lags.get(s, [np.nan]))), float((np.array(lags.get(s, [31])) > 30).mean())) for s in order2}

# ---- F7 what label error costs the ladder (rr_25d)
td = W["td"]; truth = synth.true_ladder()
def run(lab):
    cum = ladder.attach_entry_labels(ladder.cumulative(td), lab); return ladder.ladder(cum, lab, ci="hac")
lad_t, lad_e = run(W["lab_true"]), run(W["lab_est"])
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, lad, title in [(axs[0], lad_t, "true labels"), (axs[1], lad_e, "estimated labels")]:
    g = lad[lad.archetype == "rr_25d"]
    for reg, gr in g.groupby("regime"):
        if reg == "ALL": continue
        gr = gr.sort_values("h"); ax.fill_between(gr.h, gr.ci_lo, gr.ci_hi, color=C[reg], alpha=0.12, lw=0); ax.plot(gr.h, gr["mean"], "-o", color=C[reg], ms=3.5, label=reg)
    for reg, gr in truth[(truth.archetype == "rr_25d") & (truth.regime != "extreme")].groupby("regime"): ax.plot(gr.h, gr.true_ev, ":", color=C[reg], lw=1.2)
    ax.axhline(0, color="#333", lw=0.8); ax.set_title(f"rr_25d ladder, {title}"); ax.set_xlabel("h"); ax.set_xticks([1, 3, 5, 10, 20])
axs[0].set_ylabel("cumulative EV"); axs[0].legend(frameon=False, fontsize=7, ncol=2)
figs["cost"] = save(fig, "st_cost.png")

# ---- seeds table
rows = []
for seed in (0, 1, 2):
    mm = synth.simulate_market(2520, seed=seed)
    cc = labels.composite(features.build(mm, names=["vol_pct", "term_slope_pct", "rr_stress_pct"])); ss = mm["state_true"]
    f5 = pd.Series([sum(synth.earn("straddle_atm", ss.values[i + a], 21 - a) for a in range(1, 6)) if i + 6 < len(mm) else np.nan for i in range(len(mm))], index=mm.index)
    cl = labels.calibrate_states(cc, f5, level_target=mm["atm_1m"].shift(-5)); cv = labels.calibrate_states(cc, mm["rv_1m"].shift(-21), level_target=mm["atm_1m"].shift(-5))
    s6 = cl["specs"]["6"]; e6 = labels.apply_spec(cc, s6); okk = cc.notna()
    rows.append(dict(seed=seed, b1=s6["bounds"][0], b2=s6["bounds"][1], halflife=s6["halflife"], delta=s6["delta"], dir_k=s6["dir_k"], chosen=cl["spec"]["partition"], r2_3=cl["r2"]["3"], r2_6=cl["r2"]["6"], r2_6x=cl["r2"]["6x"],
                     rv_chosen=cv["spec"]["partition"], agreement=(e6[okk].values == ss[okk].values).mean(), switches_per_year=s6["switches_per_year"], extreme=("merged" if cl["specs"]["6x"]["extreme_merged"] else "kept") + f" ({cl['specs']['6x']['extreme_episodes']} ep)"))
seeds = pd.DataFrame(rows)

# ---- data for the live demos
sd_by_hl = {hl: float(labels.direction_score(labels.ewma(comp, hl), 5).std()) for hl in range(1, 11)}
data = {"comp": [round(float(v), 2) if np.isfinite(v) else None for v in comp.values], "truth": list(st.values), "fwd5": [round(float(v), 3) if np.isfinite(v) else None for v in fwd5.values],
        "bounds": [round(float(b), 1) for b in spec["bounds"]], "sd_by_hl": sd_by_hl, "spec": {"halflife": int(spec["halflife"]), "delta": float(spec["delta"]), "dir_k": float(spec["dir_k"])},
        "states": S, "states6": S6, "colours": {s: C[s] for s in S}}

# ---- body
def partition_grid(editable=False):
    cells = [("low", "up", "rising", ""), ("mid", "up", "rising", "stressed if coming down from stressed / normalising / extreme"), ("high", "up", "stressed", ""),
             ("low", "flat", "carry", ""), ("mid", "flat", "agitated", ""), ("high", "flat", "stressed", ""),
             ("low", "down", "settling", "carry if already in carry"), ("mid", "down", "settling", "normalising if coming from stressed / normalising / extreme; keep if in carry"), ("high", "down", "normalising", "")]
    out = '<div class="grid">'
    for lvl, d, s, note in cells:
        if editable:
            opts = "".join(f'<option value="{o}"{" selected" if o == s else ""}>{o}</option>' for o in S6 + ["keep"])
            out += f'<div class="cell"><small>{lvl} · {d}</small><select data-cell="{lvl},{d}">{opts}</select></div>'
        else:
            colour = {"carry": "var(--carry)", "rising": "var(--rising)", "agitated": "var(--agit)", "stressed": "var(--crisis)", "normalising": "var(--norm)", "settling": "var(--settle)"}[s]
            out += f'<div class="cell"><small>{lvl} · {d}</small><b style="color:{colour}">{s}</b>' + (f'<small>{note}</small>' if note else "") + '</div>'
    return out + "</div>"

rep_show = rep[rep.partition.isin(["3", "6", "6x"])].sort_values("oos_r2", ascending=False).head(12)[["partition", "halflife", "delta", "dir_k", "bounds", "oos_r2", "switches_per_year", "min_episodes", "extreme_merged"]].copy()
rep_show["bounds"] = rep_show["bounds"].apply(lambda b: " / ".join(f"{v:.0f}" for v in b))
fmt_rep = {"oos_r2": lambda v: f"{v:.3f}", "switches_per_year": lambda v: f"{v:.0f}", "halflife": lambda v: f"{v:.0f}", "delta": lambda v: f"{v:.0f}", "dir_k": lambda v: f"{v:.2f}", "min_episodes": lambda v: f"{v:.0f}", "extreme_merged": lambda v: "merged" if v else "—"}
fmt_seeds = {"b1": lambda v: f"{v:.0f}", "b2": lambda v: f"{v:.0f}", "halflife": lambda v: f"{v:.0f}", "delta": lambda v: f"{v:.0f}", "dir_k": lambda v: f"{v:.2f}", "r2_3": lambda v: f"{v:.3f}", "r2_6": lambda v: f"{v:.3f}", "r2_6x": lambda v: f"{v:.3f}", "agreement": lambda v: f"{v:.0%}", "switches_per_year": lambda v: f"{v:.0f}", "seed": str}

body = f"""
<!-- 1 -->
<h2><span class="n">1</span>What a state has to be {BUILT}</h2>
<p>A state is the conditioning variable of the whole product: the ladder is <i>EV(h | state on the entry date)</i>, and nothing else in the design can rescue a state definition that does not describe anything the P&amp;L sees. That puts four constraints on it before any statistics are run.</p>
<ol>
<li><b>Few.</b> Ten years of one pair hold on the order of 15–40 independent episodes per state. Every extra state halves the evidence behind every number on the card.</li>
<li><b>Persistent.</b> A state that lasts three days is noise with a name; the horizon ladder needs states that outlive the short horizons (1, 3, 5 days) so that "entered in state s" means something at h = 5.</li>
<li><b>Point-in-time.</b> The label for date <i>t</i> uses only what was known at the close of <i>t</i>. Enforced by the truncation test, not by discipline.</li>
<li><b>Decided by the P&amp;L.</b> Whether direction matters, whether the agitated middle deserves its own cell, whether an extreme band is a state or a flag — these are answered by out-of-sample separation of forward outcomes, by the finder, not by argument.</li>
</ol>
<div class="box"><b>Vocabulary.</b> A <i>score</i> is a continuous number (the composite). A <i>band</i> is a cut of a score with hysteresis (low / mid / high; down / flat / up). A <i>state</i> is a named cell of the partition table over two bands, with a memory of the previous state. A <i>tag</i> qualifies a state without changing it; a <i>sub-state</i> is a cluster discovered inside one. The ladder conditions on states (and, where the evidence allows, on state·tag cells).</div>

<!-- 2 -->
<h2><span class="n">2</span>Features and the composite {BUILT}</h2>
<p>State features answer "where are we now". The contract is strict: a feature is a function <code>f(market, asof=None) → Series</code> that uses trailing windows only — no centred windows, no full-sample statistics — and is expressed as a <b>trailing percentile rank</b> (0–100) of the raw quantity within a window of ~3 years, so that features from different surfaces and eras are on one scale and the composite is a plain average. The starting set is three: the ATM-vol percentile, the term-slope percentile (1M − 1Y), and the risk-reversal stress percentile (a trailing z-score of the 25d RR, sign flipped so that more negative RR is more stress). Three more are registered — variance-risk-premium thinness, |spot-vol beta|, realised term ratio — and available to the composite by config.</p>
<div class="eq">composite_t  =  mean( vol_pct_t , term_slope_pct_t , rr_stress_pct_t )        each a trailing percentile in [0, 100]</div>
<p>Why equal weights and not a fitted combination: with 15–40 episodes per state, a weight vector fitted to forward outcomes is the first thing that overfits, and a composite that moves with the surface is easier to reason about on a trading floor than a projection. Fitting weights is a research question for after Gate 2, and it would be answered by the same finder with the same out-of-sample criterion.</p>
{figure(figs['features'], 1, 'The three starting features over 700 days of the synthetic history, with the true state underneath. Each is a trailing percentile, so the series are comparable across pairs and eras, and leak nothing from the future. Rising starts while all three are still low — the first days of any rise are indistinguishable from carry in the features, and that lag is inherited by everything downstream.')}

<!-- 3 -->
<h2><span class="n">3</span>Smoothing and direction {BUILT}</h2>
<p>The composite is smoothed with an EWMA whose half-life is a finder choice (2–5 days). The <b>direction score</b> is the 5-day slope of the smoothed score. Both are deliberately crude: the smoothing exists to stop a one-day spike from switching the state, and the slope exists only to separate "going up" from "going down" at the band edges. Everything that follows is a function of these two series.</p>
<div class="eq">s_t   = EWMA(composite, half-life hl)
d_t   = ( s_t − s_(t−5) ) / 5                      direction thresholds at ± k · sd(d), hysteresis 0.2 · k · sd(d)</div>
<p>The cost of smoothing is lag. A half-life of {spec['halflife']} days means a step change in the composite is 75% absorbed after about {int(round(2 * spec['halflife']))} days; the direction score, being a 5-day difference, is late by a further few days. Figure 6 measures this directly as days from a true onset to the first matching label.</p>

<!-- 4 -->
<h2><span class="n">4</span>Level bands, hysteresis and the extreme band {BUILT}</h2>
<p>The level band is the smoothed score cut at ordered boundaries with hysteresis: the band moves up only when the score exceeds the boundary by δ, and down only when it falls below by δ. It starts in the lowest band and keeps its band through missing values. Two boundaries give low / mid / high; a third, far out in the tail, gives the <b>extreme</b> band.</p>
<pre>def ordered_labels(x, bounds, delta, names, start=0):
    out, state = [], start
    for v in x.values:
        if not isnan(v):
            while state &lt; len(bounds) and v &gt; bounds[state] + delta: state += 1
            while state &gt; 0            and v &lt; bounds[state-1] - delta: state -= 1
        out.append(names[state])
    return out</pre>
<p>The two level boundaries are a <i>surface fact</i> and are estimated by the finder on a forward vol level (Section 8). The extreme bound is not estimated at all: it is the 97.5th percentile of the smoothed score over the training window — a tail by definition, not a step. Whether extreme becomes a state is decided afterwards by the <b>merge rule</b>: if it has fewer than five episodes in the training window it is merged into stressed for the ladder and surfaced on the card as a flag ("in the extreme band today"); with five or more it is a state like any other. Nothing in this design is allowed to condition a number on eleven days.</p>
{figure(figs['extreme'], 2, f"The extreme band over the ten synthetic years. The smoothed composite spends {cal['specs']['6x']['extreme_days']} days above the 97.5th-percentile bound in {cal['specs']['6x']['extreme_episodes']} episodes, so on this history the rule merges it into stressed. On another seed with six episodes it is kept. The decision is re-made every time the finder runs on a training window; the card always shows the flag.")}

<!-- 5 -->
<h2><span class="n">5</span>Direction bands {BUILT}</h2>
<p>The direction band is a three-way hysteresis cut of the slope at ±k·sd(d), where sd(d) is the standard deviation of the slope over the training window and k is a finder choice (0.5, 0.75 or 1.0). The hysteresis on direction is a fifth of the threshold. Direction starts <i>flat</i>, so a history that begins mid-episode is not mislabelled by its first few days.</p>
<p>Why direction at all: on one stress axis, vol rising from low and vol falling from high are the same score and opposite trades for every component. Section 8 shows that direction more than doubles the separation of forward P&amp;L relative to level alone, and adds little against forward vol. <b>Direction is a P&amp;L fact, not a vol fact</b>; a labeller that was tuned on vol would never have found it.</p>

<!-- 6 -->
<h2><span class="n">6</span>The partition table {BUILT}</h2>
<p>Nine cells of (level, direction) map to six names. Three cells also look at the previous state, for reasons a trader would recognise:</p>
{partition_grid()}
<ol>
<li><b>Carry drifting down is still carry.</b> (low, down) is <i>settling</i> only if we were not already in carry; otherwise a quiet market whose score is edging lower would flicker between carry and settling for no reason.</li>
<li><b>Coming down from the top is normalising, all the way down.</b> (mid, down) is <i>normalising</i> when the previous state was stressed, normalising or extreme — post-crisis normalisation is where short vol and short wings earn most of their year, and it does not stop being that when the score crosses into the mid band. From rising or agitated, the same cell is <i>settling</i>: a rise that fizzled, not a crisis that is unwinding.</li>
<li><b>A bounce inside normalisation is still stressed.</b> (mid, up) is <i>rising</i> from below, but from stressed / normalising / extreme it is <i>stressed</i>: the first up-tick after a crisis peak is a crisis, not a fresh rise.</li>
</ol>
<p>(mid, flat) is <b>agitated</b>: elevated, choppy, going nowhere. The first draft made this cell "keep the previous state", which was the design flaw the agitated state fixes — a market that sits in the middle for weeks is a state in its own right, not a transit lounge between carry and crisis, and in G10 it is where most of a market-making year is lived while a full crisis is rare.</p>
<div class="demo"><div class="hd">Live · edit the partition</div>
<p class="dim" style="margin:0 0 8px">Reassign any cell and watch the labels, the switch rate, the episode counts and the out-of-sample R² of forward 5-day straddle P&amp;L change. The previous-state rules can be switched off to see what they buy. Boundaries, smoothing and thresholds are the calibrated ones.</p>
{partition_grid(editable=True)}
<div class="ctl"><div><label><input type="checkbox" id="prevrules" checked> previous-state rules (the three above)</label></div><div><button id="resetpart" style="font:inherit;padding:4px 10px">reset to the design</button></div></div>
<div class="stat"><div>agreement with true state<b id="p_agree">—</b></div><div>switches / year<b id="p_sw">—</b></div><div>episodes (min over states)<b id="p_ep">—</b></div><div>OOS R² on forward 5d P&amp;L<b id="p_r2">—</b></div></div>
<canvas id="cv_part" width="1600" height="120"></canvas>
<p class="dim" style="margin-top:6px">Days 1500–2200: true (upper) and the edited labeller (lower).</p>
</div>

<!-- 7 -->
<h2><span class="n">7</span>Live · the labeller end to end {BUILT}</h2>
{figure(figs['steps'], 3, f"The labeller step by step on 700 days: composite and EWMA with the level boundaries and hysteresis bands; the direction score with its thresholds; the level band; the direction band; and the named state against the truth. Agreement {agree:.0%} over the ten years, {labels.switches(est) / 10:.0f} switches a year against {labels.switches(st) / 10:.0f} true.")}
<div class="demo"><div class="hd">Live · smoothing, hysteresis, direction sensitivity</div>
<div class="ctl">
  <div><label>EWMA half-life: <b id="hlv">{int(spec['halflife'])}</b> days</label><input type="range" id="hl" min="1" max="10" step="1" value="{int(spec['halflife'])}"></div>
  <div><label>Level hysteresis ±<b id="dv">{int(spec['delta'])}</b></label><input type="range" id="d" min="0" max="8" step="1" value="{int(spec['delta'])}"></div>
  <div><label>Direction threshold: <b id="kv">{spec['dir_k']}</b> × sd(slope)</label><input type="range" id="k" min="0.25" max="1.5" step="0.25" value="{spec['dir_k']}"></div>
</div>
<div class="stat"><div>agreement<b id="agree">—</b></div><div>switches / year<b id="sw">—</b></div><div>true switches / year<b>{labels.switches(st) / 10:.0f}</b></div><div>episodes (min)<b id="ep">—</b></div><div>OOS R² on fwd P&amp;L<b id="r2">—</b></div></div>
<canvas id="cv" width="1600" height="470"></canvas>
<p class="dim" style="margin-top:6px">Days 1500–2200. Top: smoothed score and level boundaries. Middle: direction. Ribbons: true (upper), estimated (lower). Blue carry, green rising, gold agitated, orange stressed, dark red extreme, violet normalising, light blue settling. Notice that pushing hysteresis up buys fewer switches at the cost of later detection, and that a short half-life with no hysteresis has the best agreement and an unusable switch rate — the trade the finder is built to make for you.</p>
</div>

<!-- 8 -->
<h2><span class="n">8</span>The finder I · two targets, two jobs {BUILT}</h2>
<p><code>calibrate_states</code> takes the raw composite over a <b>training window</b> and two forward outcomes aligned to the decision date, and never looks at either when labelling — only when scoring. The separation into two targets is the single most important design decision in the finder, and it was learnt the hard way.</p>
<p><b>Level boundaries are a surface fact.</b> They are fitted on a <i>forward vol level</i> — ATM a week ahead — by a <b>step fit</b>: the two boundaries that minimise the squared error of the target around band means, with every band holding at least 8% of days. This target is available on the whole price history (ten years and more), not just the five the backtester covers, and it is monotone in the composite, so the bands it produces are bands of the vol surface.</p>
<p><b>Partition, direction thresholds, smoothing and hysteresis are a P&amp;L fact.</b> They are chosen by out-of-sample separation of the archetype's forward 5-day P&amp;L (Section 9). Fitting the <i>bounds</i> on P&amp;L fails for a reason that is obvious in hindsight: stressed earns and normalising loses for a long straddle, so on P&amp;L the high band looks like the mid band and the upper boundary is lost; and the first draft's estimator — a piecewise-<i>linear</i> kink regression — puts its breaks at the ends of a ramp, not in the middle of it.</p>
{figure(figs['kink'], 4, f"Left: a toy world with steps at 30 and 70 and nothing else. The kink fit (orange) places its breaks at {kb1:.0f} and {kb2:.0f} — the ends of the ramps the noise creates between the steps — while the step fit (blue) lands at {step_b[0]:.0f} and {step_b[1]:.0f}. Right: the synthetic world's smoothed composite by true state. The kink fit on forward P&amp;L put the first boundary at 15 and labelled most of carry as agitated (18% agreement); a step fit on forward P&amp;L (gold) finds the first boundary and loses the second; the step fit on forward ATM (blue) lands between the bands at {spec['bounds'][0]:.0f} / {spec['bounds'][1]:.0f} and agreement rises to {agree:.0%}.")}
<div class="eq">step fit:   (b1, b2) = argmin  Σ_bands  Σ_(t in band) ( y_t − mean_band(y) )²      subject to each band ≥ 8% of days
            grid of 40 quantiles of the score; y = ATM_(t+5)  (level_target)       — labels.estimate_breaks
extreme:    b3 = quantile_0.975( s_t over the training window )                     — labels.extreme_bound</div>

<!-- 9 -->
<h2><span class="n">9</span>The finder II · smoothing, hysteresis, direction and the partition {BUILT}</h2>
<p>With the bounds fixed, the finder scores every combination of half-life ∈ {{2, 3, 5}}, hysteresis δ ∈ {{0, 2, 3, 5}}, direction sensitivity k ∈ {{0.5, 0.75, 1.0}} and partition ∈ {{3, 6, 6x}} — 120 candidates — on the <b>out-of-sample R²</b> of the forward outcome: state means are estimated on the first part of each of four time-ordered folds and applied to the rest. Negative means worse than the unconditional mean. Two constraints are applied before anything is ranked:</p>
<ol>
<li><b>Minimum episodes.</b> Every state must have at least five episodes in the training window, or the candidate is discarded. A state the ladder cannot condition on is not a state.</li>
<li><b>Switch budget.</b> At most 30 switches a year (config). Without it the finder picks zero hysteresis and labels that flip every few days, because out-of-sample R² does not penalise flicker — the labels were right on average and useless on any given day. Among candidates within 0.015 of the best R², the one with the fewest switches wins.</li>
</ol>
<p>The three partitions are then compared on the same criterion: <b>3</b> (level only: carry / transition / crisis), <b>6</b> (level × direction), <b>6x</b> (6 plus the extreme band, merged or kept by the episode rule). The finder returns all three specs, the full candidate table and the R² of each; it does not decide — the owner does, at Gate 2, with the table in front of them.</p>
{figure(figs['report'], 5, f"The finder's candidate table on the synthetic world, forward 5-day straddle P&amp;L as the target. Left: best OOS R² for partition 6 by half-life and hysteresis; centre: the switch rate of those candidates — the budget is what rules out the top-left corner. Right: the three partitions under two targets. Against forward realised vol the gain from direction is small ({W['cal_rv']['r2']['3']:.2f} → {W['cal_rv']['r2']['6']:.2f}); against forward P&amp;L it is large ({cal['r2']['3']:.2f} → {cal['r2']['6']:.2f}). The extreme band changes nothing here because it was merged.")}
<h3>The top of the candidate table</h3>
{table(rep_show, fmt_rep)}
<p class="dim">`finder_report.csv` from <code>python -m regime_ladder inspect</code> holds all 120 rows. The chosen spec on this history: partition {spec['partition']}, half-life {int(spec['halflife'])}, δ {int(spec['delta'])}, k {spec['dir_k']}, bounds {spec['bounds'][0]:.0f} / {spec['bounds'][1]:.0f}, {spec['switches_per_year']:.0f} switches a year.</p>

<!-- 10 -->
<h2><span class="n">10</span>Live · the boundary explorer {BUILT}</h2>
<div class="demo"><div class="hd">Live · move the level boundaries</div>
<p class="dim" style="margin:0 0 8px">Drag the two boundaries and watch agreement, episodes and the confusion matrix. The calibrated values are {spec['bounds'][0]:.0f} / {spec['bounds'][1]:.0f}; the kink fit's were 15 / 51. The confusion matrix is row-normalised: each row is a true state, each column where the labeller put those days.</p>
<div class="ctl">
  <div><label>lower boundary b1: <b id="b1v">{spec['bounds'][0]:.0f}</b></label><input type="range" id="b1" min="5" max="60" step="1" value="{spec['bounds'][0]:.0f}"></div>
  <div><label>upper boundary b2: <b id="b2v">{spec['bounds'][1]:.0f}</b></label><input type="range" id="b2" min="40" max="95" step="1" value="{spec['bounds'][1]:.0f}"></div>
</div>
<div class="stat"><div>agreement<b id="b_agree">—</b></div><div>switches / year<b id="b_sw">—</b></div><div>episodes (min)<b id="b_ep">—</b></div><div>OOS R² on fwd P&amp;L<b id="b_r2">—</b></div></div>
<div id="cm_live"></div>
</div>

<!-- 11 -->
<h2><span class="n">11</span>What estimation error looks like {BUILT}</h2>
<p>The labeller agrees with the true state about half the time on synthetic data, and the pattern of the disagreement is more informative than the number.</p>
{figure(figs['confusion'], 6, f"Left: row-normalised confusion of true state against estimated. Carry is mostly right; rising is split between carry (seen too late) and agitated (the mid band is reached before the slope threshold is); normalising, the rarest state, is often read as stressed or agitated. Right: detection lag — days from a true onset to the first day the labeller says the same state. Median lags: " + ", ".join(f"{s} {v[0]:.0f}d" for s, v in lag_summary.items() if np.isfinite(v[0])) + ". The share labelled 'never' is onsets not matched within 30 days — mostly episodes shorter than the lag.")}
<p>Three things follow for the rest of the design. First, every ladder cell on real data is a <i>mixture</i> of its neighbours, so the estimated-label ladder is a blurred version of the true one (Figure 7); the ladder itself is not the bottleneck, the labeller is, and that is why Gate 2 (separation) sits before Gate 3 (go/no-go). Second, the first days of every rise are labelled carry — a known, structural lag that shock detection (the fast path) and leading features exist to shorten. Third, anything scored against these labels — a sub-state, a tag, a leading feature — has to be given the continuous score as a baseline, or it will be rewarded for rediscovering the label error rather than for information; the transitions and leading-features pages return to this.</p>
{figure(figs['cost'], 7, "The same risk-reversal ladder with true labels (left) and the calibrated labeller's (right); dotted lines are the analytic truth. Estimated labels pull adjacent states toward each other because each bucket contains some of its neighbours. On real data there is no left panel; the right panel is the product, and its separation is what Gate 2 measures.")}

<!-- 12 -->
<h2><span class="n">12</span>Robustness across histories {BUILT}</h2>
<p>The same finder on three independent synthetic histories. The bounds move with the history (as they should — the composite's distribution differs), the direction partitions win on P&amp;L every time, level-only wins on forward vol on two of three, and the extreme rule goes both ways.</p>
{table(seeds, fmt_seeds)}
<p class="dim">r2_* are out-of-sample R² on forward 5-day straddle P&amp;L; rv_chosen is the partition the finder prefers when the target is forward realised vol instead. Agreement is with the true state over days with a defined composite.</p>

<!-- 13 -->
<h2><span class="n">13</span>On real data {BUILT} <span class="dim">WU-08</span></h2>
<ol>
<li><b>Calibrate inside walk-forward training windows.</b> The finder runs on each training window; labels for the test window come from <code>apply_spec</code> with that window's spec. The label history that reaches the ladder is therefore itself out of sample, and the fold boundaries are logged. Pooling the whole history into one calibration is the look-ahead this scaffold is built to prevent.</li>
<li><b>Level target first.</b> On the ten-year surface, step-fit the bounds on forward ATM (<code>level_target</code>, <code>level_horizon</code> in config). Look at <code>composite.csv</code> and the boxplot of the composite by estimated state before anything else: if the bands do not look like bands of the surface, nothing downstream is worth running.</li>
<li><b>Partition on P&amp;L where the backtester covers, forward vol elsewhere.</b> The config's <code>target</code> picks; the five years of trades decide the partition, the longer history the bounds.</li>
<li><b>Read the candidate table, not just the winner.</b> If the best partition-6 candidate beats partition 3 by less than 0.03 of OOS R², the direction states are not earning their place on that pair — a legitimate finding, and the three-state fallback is one config line.</li>
<li><b>Expect the lag.</b> Median detection lags of 5–10 days are what this labeller is. The question for Gate 2 is not whether the labels are right on a given day but whether entries labelled s separate h ≤ 10 outcomes with stable sign — the separation test, with the confusion matrix beside it.</li>
<li><b>Failure modes to look for.</b> A state with fewer than five episodes in any training window (the finder discards it; if it keeps happening, the state is not real on this pair); switch rates near the budget (hysteresis too low, or the composite too noisy — try the half-life grid upward before adding features); bounds that jump between folds (the composite's distribution is drifting — check the percentile windows); extreme kept in one fold and merged in the next (expected; the card's flag handles it).</li>
</ol>
<div class="box warn"><b>Gate 2.</b> At h ≤ 10, entry states must separate outcomes with stable sign in at least 75% of (component, h) cells, out of sample. Fail → look first at detection lag against the mark moves, then at feature choice; never at <code>gates.yaml</code>.</div>

<!-- 14 -->
<h2><span class="n">14</span>Code map</h2>
<table><tr><th>function</th><th>does</th></tr>
<tr><td><code>features.build(m, names)</code></td><td>the registered features, each <code>f(market, asof)</code>; <code>features.truncation_agrees</code> is the point-in-time test</td></tr>
<tr><td><code>labels.composite(F, weights)</code></td><td>weighted mean of the chosen features (equal by default)</td></tr>
<tr><td><code>labels.ewma · direction_score</code></td><td>the smoothed score and its 5-day slope</td></tr>
<tr><td><code>labels.level_labels · direction_labels</code></td><td>hysteresis cuts into bands (via <code>ordered_labels</code>)</td></tr>
<tr><td><code>labels.two_axis_labels(level, direction, PARTITION6)</code></td><td>the partition table with previous-state rules → named states</td></tr>
<tr><td><code>labels.estimate_breaks · extreme_bound · merge_rare</code></td><td>step fit of the bounds, the tail bound, the episode rule</td></tr>
<tr><td><code>labels.calibrate_states(comp, target, level_target, ...)</code></td><td>the finder: returns <code>spec</code>, <code>specs</code> {{3, 6, 6x}}, <code>report</code>, <code>r2</code></td></tr>
<tr><td><code>labels.apply_spec(comp, spec)</code></td><td>labels from a raw composite and a spec; applies the merge rule</td></tr>
<tr><td><code>labels.transition_matrix · durations · episodes · switches</code></td><td>diagnostics on a label series</td></tr>
<tr><td><code>python -m regime_ladder inspect</code></td><td>writes <code>features.csv</code>, <code>composite.csv</code>, <code>finder_report.csv</code>, <code>finder_spec.json</code>, …</td></tr>
</table>
"""

js = r"""
const S=D.states, COL=D.colours, N=D.comp.length, MASK=D.comp.map(v=>v!=null);
const a0=1500,b0=2200;
function stats(L){return {ag:agreement(L,D.truth,MASK), sw:switches(L)/(N/252), ep:Math.min(...Object.values(episodes(L,D.states6))), r2:oosR2(L,D.fwd5,4)};}
// ---- demo 7: the labeller
function run(){const h=+hl.value,d=+document.getElementById('d').value,k=+document.getElementById('k').value;hlv.textContent=h;dv.textContent=d;kv.textContent=k.toFixed(2);
  const r=labels6(D.comp,h,D.bounds,d,k,D.sd_by_hl[h]);const s=stats(r.L);
  agree.textContent=(100*s.ag).toFixed(0)+"%";document.getElementById('sw').textContent=s.sw.toFixed(0);document.getElementById('ep').textContent=s.ep;document.getElementById('r2').textContent=s.r2.toFixed(3);
  const c=cv.getContext('2d'),W=cv.width,H=cv.height,n=b0-a0,px=W/n;c.clearRect(0,0,W,H);const top=16,ph=200,y=v=>top+ph-(v/100)*ph;
  D.bounds.forEach(bb=>{c.fillStyle="rgba(79,70,229,.08)";c.fillRect(0,y(bb+d),W,y(bb-d)-y(bb+d));c.strokeStyle="#999";c.lineWidth=1.2;c.setLineDash([6,4]);c.beginPath();c.moveTo(0,y(bb));c.lineTo(W,y(bb));c.stroke();c.setLineDash([]);});
  const line=(arr,col,lw,yf)=>{c.strokeStyle=col;c.lineWidth=lw;c.beginPath();let st=false;for(let i=a0;i<b0;i++){const v=arr[i];if(v==null){st=false;continue;}const X=(i-a0)*px,Y=yf(v);st?c.lineTo(X,Y):c.moveTo(X,Y);st=true;}c.stroke();};
  line(D.comp,"#bbb",1,y);line(r.sc,"#222",2,y);
  const dup=k*D.sd_by_hl[h],dtop=236,dh=90,ymax=3*dup+1e-9,yd=v=>dtop+dh/2-(v/ymax)*(dh/2);
  c.strokeStyle="#333";c.lineWidth=0.8;c.beginPath();c.moveTo(0,yd(0));c.lineTo(W,yd(0));c.stroke();c.strokeStyle="#999";c.setLineDash([6,4]);[dup,-dup].forEach(v=>{c.beginPath();c.moveTo(0,yd(v));c.lineTo(W,yd(v));c.stroke();});c.setLineDash([]);
  line(r.ds.map(v=>v==null?null:Math.max(-ymax,Math.min(ymax,v))),"#222",1.6,yd);
  for(let i=a0;i<b0;i++){c.fillStyle=COL[D.truth[i]]||"#999";c.fillRect((i-a0)*px,345,px+0.5,45);c.fillStyle=COL[r.L[i]]||"#999";c.fillRect((i-a0)*px,405,px+0.5,45);}
  c.font="16px Inter,system-ui,sans-serif";[["true",375],["estimated",435],["direction",dtop+dh/2+6]].forEach(([t,yy])=>{const w=c.measureText(t).width+14;c.fillStyle="rgba(255,255,255,.92)";c.fillRect(6,yy-19,w,26);c.fillStyle="#333";c.fillText(t,13,yy);});}
['hl','d','k'].forEach(id=>document.getElementById(id).addEventListener('input',run));run();
// ---- demo 10: boundaries + confusion
function runB(){const b1=+document.getElementById('b1').value,b2=+document.getElementById('b2').value;b1v.textContent=b1;b2v.textContent=b2;
  const r=labels6(D.comp,D.spec.halflife,[b1,Math.max(b2,b1+5)],D.spec.delta,D.spec.dir_k,D.sd_by_hl[D.spec.halflife]);const s=stats(r.L);
  b_agree.textContent=(100*s.ag).toFixed(0)+"%";b_sw.textContent=s.sw.toFixed(0);b_ep.textContent=s.ep;b_r2.textContent=s.r2.toFixed(3);
  const Lm=r.L.map((v,i)=>MASK[i]?v:null),Tm=D.truth.map((v,i)=>MASK[i]?v:null);
  document.getElementById('cm_live').innerHTML=cmTable(confusion(Tm,Lm,D.states),D.states,COL);}
['b1','b2'].forEach(id=>document.getElementById(id).addEventListener('input',runB));runB();
// ---- demo 6: partition editor
const DEFAULT={"low,up":"rising","mid,up":"rising","high,up":"stressed","low,flat":"carry","mid,flat":"agitated","high,flat":"stressed","low,down":"settling","mid,down":"settling","high,down":"normalising"};
function buildPart(){const useRules=document.getElementById('prevrules').checked;const P={};document.querySelectorAll('select[data-cell]').forEach(s=>P[s.dataset.cell]=s.value);
  const part={};Object.keys(P).forEach(k=>part[k]=P[k]);
  if(useRules){part["low,down"]={carry:"carry",_:P["low,down"]};part["mid,up"]={stressed:"stressed",normalising:"stressed",extreme:"stressed",_:P["mid,up"]};part["mid,down"]={stressed:"normalising",normalising:"normalising",extreme:"normalising",carry:"keep",_:P["mid,down"]};}
  ["extreme,up","extreme,flat","extreme,down"].forEach(k=>part[k]="extreme");return part;}
function runP(){const part=buildPart();const r=labels6(D.comp,D.spec.halflife,D.bounds,D.spec.delta,D.spec.dir_k,D.sd_by_hl[D.spec.halflife],part);const s=stats(r.L);
  p_agree.textContent=(100*s.ag).toFixed(0)+"%";p_sw.textContent=s.sw.toFixed(0);p_ep.textContent=isFinite(s.ep)?s.ep:"—";p_r2.textContent=s.r2.toFixed(3);
  ribbon(document.getElementById('cv_part'),[{name:"true",L:D.truth},{name:"edited",L:r.L}],COL,a0,b0);}
document.querySelectorAll('select[data-cell]').forEach(s=>s.addEventListener('change',runP));document.getElementById('prevrules').addEventListener('change',runP);
document.getElementById('resetpart').addEventListener('click',()=>{document.querySelectorAll('select[data-cell]').forEach(s=>s.value=DEFAULT[s.dataset.cell]);document.getElementById('prevrules').checked=true;runP();});
runP();
"""

toc = [("1", "What a state has to be"), ("2", "Features and the composite"), ("3", "Smoothing and direction"), ("4", "Level bands and the extreme band"), ("5", "Direction bands"), ("6", "The partition table · live editor"),
       ("7", "Live · the labeller"), ("8", "The finder I · two targets"), ("9", "The finder II · the candidate table"), ("10", "Live · boundary explorer"), ("11", "What estimation error looks like"), ("12", "Robustness across histories"), ("13", "On real data"), ("14", "Code map")]
out = page("states.html", "States and the state finder",
           "How a market day becomes one of six named states: the features and the composite, the two bands and their hysteresis, the partition table with its memory, the extreme band and its merge rule, the finder that sets every number from forward outcomes, and what the labels cost when they are wrong. Every figure is from synthetic data with a known answer; the live panels run the same labeller code in the browser.",
           toc, body, data, js)
print(out, agree)
