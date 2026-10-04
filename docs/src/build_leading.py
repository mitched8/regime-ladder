"""Deep dive 3: leading.html — Leading features and stored energy."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt, yaml
from common import world, energy_world, figure, save, table, page, COLOURS, BUILT, HYP, ROOT
import sys; sys.path.insert(0, str(ROOT))
from regime_ladder import synth, labels, leading, transitions, features, gates

W = world(); E = energy_world(3.0)
C = COLOURS; S6 = list(labels.STATES6)
G = gates.load_gates(str(ROOT / "configs/gates.yaml"))["phase4_leading"]
me, ste, este, compe, fwd5e = E["m"], E["st"], E["est"], E["comp"], E["fwd5"]
m0, st0, est0, comp0, fwd50 = W["m"], W["st"], W["est"], W["comp"], W["fwd5"]
x = np.arange(len(me)); seg = slice(1300, 2000)
figs = {}

def ribbon(ax, series, y, h, seg):
    cols = [C.get(v, "#999") for v in series.values[seg]]
    ax.bar(x[seg], [h] * len(cols), bottom=y, width=1, color=cols, lw=0)

# ---- F1 the gap
gz = leading.gap_z(me); gp = leading.gap_pct(me)
ls = np.log(me["spot"]); anchor = ls.rolling(120, min_periods=60).mean().shift(1)
fig, axs = plt.subplots(3, 1, figsize=(10, 5.6), sharex=True, gridspec_kw={"height_ratios": [1.4, 1, 0.45]})
axs[0].plot(x[seg], np.exp(ls.values[seg]), color="#222", lw=1, label="spot"); axs[0].plot(x[seg], np.exp(anchor.values[seg]), color="#4F46E5", lw=1.2, label="anchor: trailing 120-day mean (log), excluding today")
band = (me["atm_1m"] / 100 * np.sqrt(60 / 252)).values
axs[0].fill_between(x[seg], np.exp(anchor.values[seg] - band[seg]), np.exp(anchor.values[seg] + band[seg]), color="#4F46E5", alpha=0.08, label="± one implied-vol move over half the window")
axs[0].legend(frameon=False, fontsize=7.5); axs[0].set_ylabel("spot"); axs[0].set_title("The gap: how far spot has travelled from its anchor, in units of what implied vol allows")
axs[1].plot(x[seg], gz.values[seg], color="#222", lw=1, label="gap_z (signed)"); axs[1].plot(x[seg], gp.values[seg] / 50 - 1, color="#E8743B", lw=1, label="gap_pct (trailing percentile of |gap_z|, rescaled)"); axs[1].axhline(0, color="#333", lw=0.6); axs[1].legend(frameon=False, fontsize=7.5); axs[1].set_ylabel("gap")
ribbon(axs[2], ste, 0, 1, seg); axs[2].set_yticks([]); axs[2].grid(False); axs[2].set_xlabel("trading day")
figs["gap"] = save(fig, "ld_gap.png")

# ---- F2 complacency components
from regime_ladder.features import rolling_percentile
rv_iv = rolling_percentile(-(me["rv_1m"] / me["atm_1m"]), 756); vov = rolling_percentile(-me["atm_1m"].diff().rolling(21).std(), 756); slopep = rolling_percentile(-(me["atm_1m"] - me["atm_1y"]), 756)
compl = leading.complacency(me)
fig, axs = plt.subplots(2, 1, figsize=(10, 4.4), sharex=True, gridspec_kw={"height_ratios": [2, 0.45]})
axs[0].plot(x[seg], rv_iv.values[seg], color="#9aa0a6", lw=0.9, label="realised / implied LOW (pct)"); axs[0].plot(x[seg], vov.values[seg], color="#0EA5E9", lw=0.9, label="vol-of-vol LOW (pct)"); axs[0].plot(x[seg], slopep.values[seg], color="#8B5CF6", lw=0.9, label="front end cheap vs back (pct)")
axs[0].plot(x[seg], compl.values[seg], color="#222", lw=1.6, label="complacency = mean of the three"); axs[0].legend(frameon=False, fontsize=7.5, ncol=2); axs[0].set_ylabel("0–100"); axs[0].set_title("Complacency: how temporary the holder of the gap looks")
ribbon(axs[1], ste, 0, 1, seg); axs[1].set_yticks([]); axs[1].grid(False); axs[1].set_xlabel("trading day")
figs["compl"] = save(fig, "ld_compl.png")

# ---- F3 the product and the hidden pressure
se = leading.stored_energy(me)
fig, axs = plt.subplots(3, 1, figsize=(10, 5.6), sharex=True, gridspec_kw={"height_ratios": [1.4, 1.2, 0.45]})
axs[0].plot(x[seg], gp.values[seg], color="#4F46E5", lw=0.9, label="gap_pct"); axs[0].plot(x[seg], compl.values[seg], color="#9aa0a6", lw=0.9, label="complacency"); axs[0].plot(x[seg], se.values[seg], color="#E8743B", lw=1.5, label="stored energy = gap_pct × complacency / 100")
axs[0].legend(frameon=False, fontsize=7.5, ncol=3); axs[0].set_ylabel("0–100"); axs[0].set_title("The product: a stretched spot AND a complacent surface")
axs[1].plot(x[seg], me["pressure_true"].values[seg], color="#222", lw=1); axs[1].set_ylabel("hidden pressure"); axs[1].set_title("The generator's own pressure in the energy world — what the feature is trying to see", fontsize=9)
ribbon(axs[2], ste, 0, 1, seg); axs[2].set_yticks([]); axs[2].grid(False); axs[2].set_xlabel("trading day")
figs["product"] = save(fig, "ld_product.png")
corr_tab = pd.DataFrame({"feature": ["gap_pct", "complacency", "stored_energy", "min(gap, complacency)", "geometric mean"],
                         "corr with hidden pressure": [gp.corr(me["pressure_true"]), compl.corr(me["pressure_true"]), se.corr(me["pressure_true"]), np.minimum(gp, compl).corr(me["pressure_true"]), np.sqrt(gp * compl).corr(me["pressure_true"])],
                         "corr with pressure, days before escalation only": [None] * 5})
lvr = ste.map(transitions.RANK); up = (lvr.diff() > 0)
pre = up.shift(-1).fillna(False) | up.shift(-2).fillna(False) | up.shift(-3).fillna(False)
for i, s_ in enumerate([gp, compl, se, np.minimum(gp, compl), np.sqrt(gp * compl)]):
    corr_tab.iloc[i, 2] = s_[pre].corr(me["pressure_true"][pre])

# ---- F4 feedback proxies
Lall = leading.build(me)
fig, axs = plt.subplots(2, 1, figsize=(10, 4.4), sharex=True, gridspec_kw={"height_ratios": [2, 0.45]})
for col, colr, nm in [("jump_cluster_pct", "#E8743B", "jump clustering"), ("coupling_shift_pct", "#8B5CF6", "spot-vol coupling shift"), ("coherence_pct", "#0EA5E9", "cross-pair coherence")]:
    axs[0].plot(x[seg], Lall[col].values[seg], color=colr, lw=0.9, label=nm)
axs[0].legend(frameon=False, fontsize=7.5, ncol=3); axs[0].set_ylabel("0–100"); axs[0].set_title("Feedback: would a shock spread?")
ribbon(axs[1], ste, 0, 1, seg); axs[1].set_yticks([]); axs[1].grid(False); axs[1].set_xlabel("trading day")
figs["feedback"] = save(fig, "ld_feedback.png")

# ---- F5 event proximity
idx = me.index[seg]; ev = [idx[60], idx[61], idx[200], idx[420], idx[421], idx[422]]
ep = leading.event_proximity(idx, ev, horizon=10)
fig, ax = plt.subplots(figsize=(10, 1.8)); ax.plot(x[seg], ep.values, color="#E8743B", lw=1.3); ax.set_ylabel("event proximity"); ax.set_xlabel("trading day"); ax.set_title("A calendar covariate: 1 on the event day, ramping from 0 ten days before; known in advance, path = calendar")
figs["event"] = save(fig, "ld_event.png")

# ---- screens: energy world (true, estimated), null world (true, estimated)
cfgL = yaml.safe_load(open(ROOT / "configs/leading.yaml"))["leading"]
L0 = leading.build(m0); Le = Lall.copy(); Le["pressure_true"] = me["pressure_true"]
scr_e_true = leading.screen(Le, ste, compe, fwd5e, G); scr_e_est = leading.screen(Le, este, compe, fwd5e, G)
scr_0_true = leading.screen(L0, st0, comp0, fwd50, G); scr_0_est = leading.screen(L0, est0, comp0, fwd50, G)

# ---- F6 the outcome test by fold, energy world true labels
def fold_r2(feature, lab, comp, target, folds=4):
    df = pd.concat([feature.rename("f"), lab.rename("s"), comp.rename("c"), target.rename("y")], axis=1).dropna()
    ref = df["s"].value_counts().idxmax(); D = pd.get_dummies(df["s"]).astype(float).drop(columns=[ref]).values
    cz_ = ((df["c"] - comp.mean()) / comp.std()).values; base = np.column_stack([np.ones(len(df)), D, cz_, cz_ ** 2]); fz = ((df["f"] - df["f"].mean()) / df["f"].std()).values
    _, pb = leading._oos_r2(base, df["y"].values, folds, 0.4); _, pf = leading._oos_r2(np.column_stack([base, fz]), df["y"].values, folds, 0.4)
    return np.array(pf) - np.array(pb)
fig, ax = plt.subplots(figsize=(8, 3.2))
feats = ["pressure_true", "gap_pct", "stored_energy", "complacency", "jump_cluster_pct", "coherence_pct"]
for j, f in enumerate(feats):
    d = fold_r2(Le[f], ste, compe, fwd5e); ax.bar(np.arange(4) + (j - 2.5) * 0.13, d, 0.13, color=["#222", "#4F46E5", "#E8743B", "#9aa0a6", "#D4A017", "#0EA5E9"][j], label=f)
ax.axhline(0, color="#333", lw=0.8); ax.axhline(G["min_delta_r2"], color="#2EAE7A", lw=0.8, ls="--"); ax.text(3.5, G["min_delta_r2"] + 0.0005, "threshold (on the total)", fontsize=7, color="#2EAE7A", ha="right")
ax.set_xticks(range(4)); ax.set_xticklabels([f"fold {i + 1}" for i in range(4)]); ax.set_ylabel("ΔR² out of sample"); ax.set_title("The outcome test by fold: R² gain beyond state dummies + composite + composite² (energy world, true labels)"); ax.legend(frameon=False, fontsize=7, ncol=3)
figs["foldr2"] = save(fig, "ld_foldr2.png")

# ---- F7 lag check
rows = []
for lag in (0, 1, 2, 5, 10):
    for ln, lab in (("true labels", ste), ("estimated labels", este)):
        r = leading.incremental_value(Le["pressure_true"], lab, compe, fwd5e, lag=lag); rows.append(dict(lag=lag, labels=ln, delta_r2=r["delta_r2"], transition_gain=r["transition_gain"], retain=leading.retain(r, G)))
lagdf = pd.DataFrame(rows)
fig, axs = plt.subplots(1, 2, figsize=(10, 3))
for ln, col in (("true labels", "#222"), ("estimated labels", "#4F46E5")):
    g = lagdf[lagdf.labels == ln]; axs[0].plot(g.lag, g.delta_r2, "o-", color=col, label=ln); axs[1].plot(g.lag, g.transition_gain, "o-", color=col, label=ln)
axs[0].axhline(G["min_delta_r2"], color="#2EAE7A", ls="--", lw=0.8); axs[1].axhline(G["min_transition_gain"], color="#2EAE7A", ls="--", lw=0.8)
axs[0].set_xlabel("days the feature is delayed"); axs[0].set_ylabel("ΔR² OOS"); axs[0].set_title("Outcome test vs delay (true pressure)"); axs[0].legend(frameon=False, fontsize=8)
axs[1].set_xlabel("days the feature is delayed"); axs[1].set_ylabel("transition gain / obs"); axs[1].set_title("Transition test vs delay")
figs["lag"] = save(fig, "ld_lag.png")

# ---- F8 effect size vs power
rows = []
for eb in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
    Eb = energy_world(eb) if eb > 0 else dict(m=m0, st=st0, est=est0, comp=comp0, fwd5=fwd50)
    Lb = leading.build(Eb["m"], names=["gap_pct", "stored_energy"]); Lb["pressure_true"] = Eb["m"]["pressure_true"]
    for f in ("pressure_true", "gap_pct", "stored_energy"):
        for ln, lab in (("true", Eb["st"]), ("estimated", Eb["est"])):
            r = leading.incremental_value(Lb[f], lab, Eb["comp"], Eb["fwd5"])
            rows.append(dict(energy_beta=eb, feature=f, labels=ln, delta_r2=r.get("delta_r2", np.nan), folds_r2=r.get("folds_r2_up", 0), transition_gain=r.get("transition_gain", np.nan), folds_tr=r.get("folds_transition_up", 0), retain=leading.retain(r, G),
                             share_pressure=float((Eb["m"]["pressure_true"] > 0).mean())))
power = pd.DataFrame(rows)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
for f, col in (("pressure_true", "#222"), ("gap_pct", "#4F46E5"), ("stored_energy", "#E8743B")):
    for ln, ls_ in (("true", "-"), ("estimated", "--")):
        g = power[(power.feature == f) & (power.labels == ln)]; axs[0].plot(g.energy_beta, g.delta_r2, ls_, marker="o", ms=3.5, color=col, label=f"{f}, {ln} labels"); axs[1].plot(g.energy_beta, g.transition_gain, ls_, marker="o", ms=3.5, color=col)
axs[0].axhline(G["min_delta_r2"], color="#2EAE7A", ls=":", lw=1); axs[1].axhline(G["min_transition_gain"], color="#2EAE7A", ls=":", lw=1)
axs[0].set_xlabel("energy_beta (strength of the causal effect)"); axs[0].set_ylabel("ΔR² OOS"); axs[0].set_title("Outcome test vs effect size"); axs[0].legend(frameon=False, fontsize=6.5, ncol=2)
axs[1].set_xlabel("energy_beta"); axs[1].set_ylabel("transition gain / obs"); axs[1].set_title("Transition test vs effect size (dotted: thresholds)")
figs["power"] = save(fig, "ld_power.png")

# extreme kept vs merged, for the fragility note
r_kept = leading.incremental_value(Le["gap_pct"], ste, compe, fwd5e); r_merged = leading.incremental_value(Le["gap_pct"], ste.replace({"extreme": "stressed"}), compe, fwd5e)
n_extreme = int((ste == "extreme").sum())

# ---- F9 pressure index into the tilt: fan with and without
kept = list(scr_e_true.loc[scr_e_true["retain"] & (scr_e_true["feature"] != "pressure_true"), "feature"])
psi = leading.pressure(Lall, {k: 1.0 for k in kept} or {"gap_pct": 1.0})
cz = (compe - compe.mean()) / compe.std()
Pe = labels.transition_matrix(ste, names=labels.state_order(ste), prior_strength=10)
fit = transitions.fit_tilt(ste, psi, Pe, k=5, base=cz)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [1.6, 1]})
axs[0].plot(x[seg], psi.values[seg], color="#222", lw=1); axs[0].axhline(0, color="#333", lw=0.6); axs[0].set_ylabel("pressure ψ"); axs[0].set_xlabel("trading day"); axs[0].set_title(f"The pressure index from the retained features {kept or ['gap_pct']}: (mean − 50) / 25")
hi = [s for s in ("stressed", "extreme") if s in Pe.index]
for psi_now, ls_, lab_ in [(0.0, "-", "ψ = 0"), (1.0, "--", "ψ = +1, decaying"), (2.0, "-.", "ψ = +2, decaying")]:
    f = transitions.fan(Pe, pd.Series({"carry": 1.0}), psi_path=transitions.covariate_path(psi_now, 21, "decay", 10), beta=fit["beta"]); axs[1].plot(f.index, f[hi].sum(axis=1), ls_, color="#E8743B" if psi_now else "#555", lw=1.4, label=lab_)
axs[1].set_xlabel("days ahead"); axs[1].set_ylabel("P(stressed or extreme)"); axs[1].set_title(f"From carry, β = {fit['beta']:+.2f}, γ = {fit['gamma']:+.2f}"); axs[1].legend(frameon=False, fontsize=8)
figs["pressure"] = save(fig, "ld_pressure.png")

# ---- live data: energy world, true and estimated labels, per-fold P0 and gamma
S7 = S6 + ["extreme"]
def fold_setup(lab):
    n = len(lab); cuts = np.linspace(int(0.4 * n), n, 5).astype(int); out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        tr = lab.iloc[:a]; P_ = labels.transition_matrix(tr, names=S7, prior_strength=10).reindex(index=S7, columns=S7).fillna(0)
        for s in S7:
            if P_.loc[s].sum() == 0: P_.loc[s, s] = 1.0
        gamma = transitions.fit_tilt(tr, cz, P_, k=5)["beta"]
        out.append({"a": int(a), "b": int(b), "P": [[round(float(v), 5) for v in r] for r in P_.values], "gamma": round(float(gamma), 4)})
    return out
def ser(s): return [round(float(v), 3) if np.isfinite(v) else None for v in s.values]
data = {"states": S7, "colours": {s: C[s] for s in S7}, "ranks": [transitions.RANK[s] for s in S7],
        "feat": {c: ser(Lall[c]) for c in Lall.columns}, "pressure_true": ser(me["pressure_true"]), "comp_z": ser(cz), "fwd5": ser(fwd5e),
        "labels": {"true": list(ste.values), "estimated": list(este.values)}, "folds": {"true": fold_setup(ste), "estimated": fold_setup(este)},
        "thr": G}

fmt_scr = {"r2_base": lambda v: f"{v:.3f}", "delta_r2": lambda v: f"{v:+.4f}", "transition_gain": lambda v: f"{v:+.4f}", "beta_mean": lambda v: f"{v:+.2f}", "n": str, "n_transitions": lambda v: f"{int(v)}" if pd.notna(v) else "", "folds_r2_up": lambda v: f"{int(v)}/4" if pd.notna(v) else "", "folds_transition_up": lambda v: f"{int(v)}/4" if pd.notna(v) else "", "retain": lambda v: '<span class="ok">yes</span>' if v else '<span class="no">no</span>'}
cols = ["feature", "r2_base", "delta_r2", "folds_r2_up", "transition_gain", "folds_transition_up", "beta_mean", "retain"]
def scr_table(df): return table(df[cols], fmt_scr)
fmt_lag = {"lag": str, "delta_r2": lambda v: f"{v:+.4f}", "transition_gain": lambda v: f"{v:+.4f}", "retain": lambda v: '<span class="ok">yes</span>' if v else '<span class="no">no</span>'}
fmt_corr = {"corr with hidden pressure": lambda v: f"{v:.3f}", "corr with pressure, days before escalation only": lambda v: f"{v:.3f}"}
fmt_pow = {"energy_beta": lambda v: f"{v:.1f}", "delta_r2": lambda v: f"{v:+.4f}", "transition_gain": lambda v: f"{v:+.4f}", "folds_r2": lambda v: f"{int(v)}/4", "folds_tr": lambda v: f"{int(v)}/4", "share_pressure": lambda v: f"{v:.0%}", "retain": lambda v: '<span class="ok">yes</span>' if v else '<span class="no">no</span>'}

body = f"""
<!-- 1 -->
<h2><span class="n">1</span>The question, and the constraint {HYP}</h2>
<p>State features say where we are. Leading features say what might push us out — and, for a market-making book, the question is sharper than "will vol go up": option P&amp;L is realised <i>relative to implied</i>. A signal the surface already prices adds nothing to a position that is marked on that surface, unless it is mispriced. So a leading feature earns its place in one of two ways and no other: by forecasting <b>transitions the surface has not absorbed</b> (the state moves and the surface follows), or by forecasting <b>realised minus implied</b> directly. Everything on this page is organised around making that claim testable, feature by feature, against a baseline that already knows the state.</p>
<div class="box"><b>Two things this page is not.</b> It is not a signal library: the six generic candidates exist to exercise the machinery and to give the desk's own series something to be compared with. And it is not a claim that stored energy works: on synthetic data the machinery recovers a causal effect when one is planted and finds nothing when none is; whether the real surface leaves anything unpriced is what the screen on real data will say. The economics are a <span class="st hyp" style="margin:0">hypothesis</span>; the test is <span class="st built" style="margin:0">built</span>.</div>

<!-- 2 -->
<h2><span class="n">2</span>The anatomy of a vol event {HYP}</h2>
<p>The organising picture, kept as desk language and as a checklist of where a measurable proxy could come from. A <b>spark</b> (a catalyst: an event, a policy move, a rate gap, news, intervention) becomes a vol event only if it hits <b>fuel</b> — positioning that gets hurt (IMM positioning, the carry stock, barrier concentrations, client flow) — and an <b>amplifier</b> — dealer short gamma, a structured book that has to hedge the wrong way, thin liquidity. <b>Dampers</b> (dealer long gamma, vol supply, level-based flows, clean positioning) absorb it. What the realised path looks like — jump clustering, a shift in spot-vol correlation — feeds back into <b>premium</b>: hedgers buy, vol sellers withdraw, the surface reprices, and an implied overshoot can itself become fuel. A move need not fit the picture to matter; the picture says where to look.</p>
<figure>
<svg viewBox="0 0 980 230" width="100%" xmlns="http://www.w3.org/2000/svg">
<style>.t{{font-size:12px;font-weight:600;fill:#1a1d23}}.s{{font-size:9.6px;fill:#5f6670}}.b{{fill:#fff;stroke-width:1.4;rx:5}}</style>
<defs><marker id="ar2" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#6B7280"/></marker></defs>
<rect x="20" y="40" width="150" height="80" class="b" stroke="#E8743B"/><text x="30" y="60" class="t">A · Spark</text><text x="30" y="78" class="s">event calendar → event_proximity</text><text x="30" y="91" class="s">policy, rate gap, intervention:</text><text x="30" y="104" class="s">desk series</text>
<rect x="215" y="40" width="150" height="80" class="b" stroke="#E8743B"/><text x="225" y="60" class="t">B · Fuel</text><text x="225" y="78" class="s">gap to anchor → gap_pct</text><text x="225" y="91" class="s">IMM, carry stock, barriers,</text><text x="225" y="104" class="s">client flow: desk series</text>
<rect x="410" y="40" width="150" height="80" class="b" stroke="#8B5CF6" style="fill:#f3efff"/><text x="420" y="60" class="t">C · Amplifier</text><text x="420" y="78" class="s">coupling shift, coherence</text><text x="420" y="91" class="s">dealer gamma, structured</text><text x="420" y="104" class="s">book: desk series</text>
<rect x="605" y="40" width="150" height="80" rx="5" fill="#1a1d23"/><text x="615" y="60" class="t" style="fill:#fff">Realised vol</text><text x="615" y="78" class="s" style="fill:#d9dce1">jump_cluster_pct,</text><text x="615" y="91" class="s" style="fill:#d9dce1">coupling_shift_pct</text>
<rect x="800" y="40" width="160" height="80" class="b" stroke="#E8743B"/><text x="810" y="60" class="t">E · Premium</text><text x="810" y="78" class="s">complacency (rv/iv, vol-of-vol,</text><text x="810" y="91" class="s">front end) → the holder's weakness</text>
<rect x="215" y="150" width="150" height="60" class="b" stroke="#3B6FD4"/><text x="225" y="170" class="t">Stored energy</text><text x="225" y="187" class="s">gap_pct × complacency</text>
<rect x="410" y="150" width="150" height="60" class="b" stroke="#2EAE7A"/><text x="420" y="170" class="t">Dampers</text><text x="420" y="187" class="s">dealer long gamma, vol supply,</text><text x="420" y="200" class="s">level flows: desk series</text>
<g stroke="#9AA0A6" stroke-width="1.2" fill="none" marker-end="url(#ar2)"><path d="M170,80 L215,80"/><path d="M365,80 L410,80"/><path d="M560,80 L605,80"/><path d="M755,80 L800,80"/><path d="M290,150 L290,120"/><path d="M485,150 L485,120"/></g>
<path d="M880,120 C880,150 700,160 680,122" stroke="#9AA0A6" stroke-width="1" fill="none" stroke-dasharray="3 3" marker-end="url(#ar2)"/><text x="700" y="165" class="s">implied overshoot feeds back</text>
<text x="20" y="22" class="s" style="letter-spacing:1.2px;font-weight:600">EACH BOX NAMES THE GENERIC PROXY BUILT HERE AND THE DESK SERIES THAT WOULD DO THE JOB PROPERLY</text>
</svg>
<figcaption><b>Figure 1.</b> The anatomy with its proxies. The generic candidates are what can be built from a surface and a spot series alone; the series that would carry real edge — flow, dealer gamma, the barrier book, positioning — are the desk's, and they enter through the same contract (Section 8).</figcaption>
</figure>

<!-- 3 -->
<h2><span class="n">3</span>Stored energy I · the gap {BUILT}</h2>
<p>"A gap between a price and its anchor, held by something temporary." The gap half is the distance of spot from a trailing anchor, measured in the units that matter for an option book: what implied vol says a move of that size should cost.</p>
<div class="eq">anchor_t  =  mean( log spot , last 120 days, excluding today )
gap_z_t   =  ( log spot_t − anchor_t )  /  ( σ_impl,t · sqrt(60/252) )          σ_impl = ATM 1M / 100
gap_pct_t =  trailing percentile of | gap_z_t |  over 3 years</div>
<p>Why implied units and not pips: a 3% move in a 6-vol pair is a different object from a 3% move in a 12-vol pair, and the quantity that positions are hedged, margined and marked against is the implied one. Why a 120-day anchor: a quarter to half a year is the horizon over which carry positions and client hedges accumulate; it is a config parameter and a natural thing to vary. Why the percentile: it puts the feature on the same 0–100 scale as every other, and the retention test is scale-free anyway.</p>
{figure(figs['gap'], 2, "The gap in the energy world. Top: spot, its trailing anchor and the ± one-implied-move band around it. Middle: the signed gap in implied units, and its trailing percentile. A sustained one-way drift with vol staying low is what loads the gap; a spike in implied narrows it without spot moving, which is the point of the units.")}

<!-- 4 -->
<h2><span class="n">4</span>Stored energy II · complacency {BUILT}</h2>
<p>The "held by something temporary" half. The holder of a stretched spot — the carry stock, the barrier book, the client who sold the wings — looks weakest when the surface is quiet and well supplied: realised running far under implied (vol sellers are being paid and are adding), vol-of-vol low (nothing has been asked of the surface), and the front end cheap against the back (the market expects nothing soon). Each is a trailing percentile, oriented so that high = complacent, and complacency is their mean.</p>
<div class="eq">complacency_t  =  mean( pct( − RV_1M / IV_1M ) , pct( − std_21d(ΔIV) ) , pct( − (IV_1M − IV_1Y) ) )       each over 3 years</div>
{figure(figs['compl'], 3, "The three components and their mean. They disagree often — that is why the mean is used rather than any one — and the mean is high in long carry stretches and collapses on the first day the surface is asked a question.")}
<p>None of the three is a positioning measure; they are what positioning <i>looks like</i> from the surface. The desk series that would replace them — dealer gamma, the barrier book, flow — slot in exactly here, as one more percentile in the mean or as a candidate on their own.</p>

<!-- 5 -->
<h2><span class="n">5</span>Stored energy III · the product {BUILT}</h2>
<p>The hypothesis is a conjunction: a stretched spot is unremarkable if the surface is already pricing the risk, and a complacent surface is unremarkable if spot is at its anchor. The feature is therefore a <b>product</b>, not a sum — AND logic on a 0–100 scale — and it is high only when both are.</p>
<div class="eq">stored_energy_t  =  gap_pct_t × complacency_t / 100            in [0, 100]</div>
{figure(figs['product'], 4, "The product against its two parts (top) and against the generator's hidden pressure (middle), in the world where stored energy is causal. The generator's rule is |gap| counted in full when implied is low, zero otherwise, which is a sharper version of the same idea; the feature tracks it imperfectly and late, which is the realistic case.")}
{table(corr_tab, fmt_corr)}
<p class="dim">Correlation of each candidate with the hidden pressure, over all days and over the three days before each escalation. Other conjunctions (min, geometric mean) are close; the product is kept because it is the simplest and the screen, not this table, decides.</p>

<!-- 6 -->
<h2><span class="n">6</span>Feedback: would a shock spread? {BUILT}</h2>
<p>Three proxies for the amplifier side, each a trailing percentile:</p>
<ol>
<li><b>Jump clustering</b> — the share of the last 21 days with |surprise| &gt; 2 (surprise = return over the implied daily vol marked before it, the shock page's quantity). Jumps that cluster are a market that has started to move more than it is priced to.</li>
<li><b>Coupling shift</b> — |corr<sub>21d</sub> − corr<sub>126d</sub>| of spot returns with implied changes: the spot-vol relationship moving away from its recent norm, in either direction. A flip in coupling is often the first visible sign that the hedging flow has changed hands.</li>
<li><b>Cross-pair coherence</b> — the mean pairwise correlation of the other G10 pairs' returns over 21 days. When everything moves together, a shock in one place reaches all of them; when correlations are low, it dies locally.</li>
</ol>
{figure(figs['feedback'], 5, "The three feedback proxies over the same stretch. On the synthetic world none of them is causal — the generator's pressure is a gap-and-complacency rule — so these are what a feature that measures something real but irrelevant looks like going through the screen: high values, plausible stories, no retained value.")}

<!-- 7 -->
<h2><span class="n">7</span>The calendar: the one covariate known in advance {BUILT}</h2>
<p>Scheduled events — central bank meetings, major releases, elections — are the only leading information whose <i>future path is known</i>. <code>event_proximity</code> is 1 on the event day and ramps linearly from 0 ten days before; it enters the screen like any other candidate and the tilt with the <i>calendar</i> path policy, so a fan projected today shows a bump at the known date and nothing in between. The adapter supplies the dates per pair; nothing in the tracked repo knows what they are.</p>
{figure(figs['event'], 6, "Event proximity for three scheduled dates (one a two-day event). Point-in-time by construction.")}

<!-- 8 -->
<h2><span class="n">8</span>The contract: how a feature gets in {BUILT}</h2>
<p>Every leading feature is a function <code>f(market, asof=None, **params) → Series</code> that uses only rows up to <code>asof</code> and trailing normalisation. Registering it in <code>leading.LEADING</code> is all it takes for three things to happen without further code: the truncation test parametrises over the registry and fails the build if the feature peeks; <code>leading.build</code> includes it; and the screen scores it against the same baseline as every other candidate. The desk's own series come in from the untracked adapter exactly this way.</p>
<pre>def my_positioning(market, asof=None, window=756):
    m = market if asof is None else market.loc[:asof]
    return rolling_percentile(m["my_positioning_column"], window).rename("my_positioning_pct")

leading.LEADING["my_positioning_pct"] = my_positioning        # done: PIT test, build, screen, inspect</pre>

<!-- 9 -->
<h2><span class="n">9</span>The retention rule I · the outcome test {BUILT}</h2>
<p>Each candidate is tested <b>one at a time</b>. The baseline already knows the state: one dummy per named state, plus the continuous composite and its square, so that any value the feature has merely from tracking the state — or the labeller's own lag — is absorbed before the feature is scored. Both models are fitted by least squares on the training part of each of four time-ordered folds and scored on its test part; the statistic is the out-of-sample R² with the feature minus without.</p>
<div class="eq">baseline:   y_t  ~  Σ_s 1[state_t = s] · a_s  +  b₁ composite_t  +  b₂ composite_t²
candidate:  baseline  +  c · z(feature_t)
ΔR²  =  R²_OOS(candidate) − R²_OOS(baseline)       pooled over the test parts of 4 folds; also counted fold by fold</div>
<p>The outcome y is the archetype's forward 5-day P&amp;L where the backtester covers the window (config <code>target: forward_pnl</code>), or forward realised minus implied over 21 days on the longer history. The 5-day straddle earn is the default because it is the primary horizon of the ladder and the straddle is the cleanest vega object; realised-minus-implied is the alternative that needs no trades.</p>
{figure(figs['foldr2'], 7, "ΔR² by fold for six candidates in the energy world with true labels. The true pressure is positive in every fold; the gap feature in three; the feedback proxies hover around zero and change sign between folds. The threshold applies to the pooled total and to the count of positive folds (≥ 3 of 4).")}

<!-- 10 -->
<h2><span class="n">10</span>The retention rule II · the transition test {BUILT}</h2>
<p>The second test is the transitions page's: the standardised feature enters the tilt as the pressure, with the continuous composite as the baseline covariate γ (fitted first, held fixed), β fitted by maximum likelihood of the <b>5-step</b> transitions on each fold's training part, and the gain in out-of-sample log-likelihood per observation with β against β = 0. Both the k-step horizon and the baseline were forced on the design by the same observation: against estimated labels and without them, the <i>true</i> pressure came out with the wrong sign (transitions page, Section 7). The feature must clear both tests — a positive total and ≥ 3 positive folds on each — to be <b>retained</b>.</p>
<div class="eq">retain  ⇔  ΔR² ≥ 0.005  and  folds(ΔR² &gt; 0) ≥ 3  and  gain ≥ 0.002 nats/obs  and  folds(gain &gt; 0) ≥ 3        (configs/gates.yaml › phase4_leading)</div>

<!-- 11 -->
<h2><span class="n">11</span>The lag check {BUILT}</h2>
<p>A feature that is only available with a delay — a positioning report published weekly, a flow aggregate that lands next morning — must be scored at the delay it will actually have. <code>lag</code> shifts the feature back before the tests; the screen reports the undelayed and one-day-delayed values side by side, and the table below walks further out.</p>
{figure(figs['lag'], 8, "Both tests for the true pressure as it is delayed by 0–10 days, in the energy world. The value decays within a week against true labels and within a day or two against estimated labels; a feature that arrives late has to be much stronger to survive.")}
{table(lagdf, fmt_lag)}

<!-- 12 -->
<h2><span class="n">12</span>Positive and negative controls {BUILT}</h2>
<p>The synthetic world has a switch. With <code>energy_beta &gt; 0</code> the daily transition matrix is tilted toward escalation by a pressure the generator computes from its own path (|gap| counted in full when implied is low), so stored energy is <i>causal</i>; with it at zero the same world is a null. The screens below are the positive control (energy world, effect size 3) and the negative control (null world), each against true and estimated labels. The generator's own pressure is included as a candidate the model cannot see in practice — the ceiling of what the test can find.</p>
<h3>Energy world · true labels</h3>{scr_table(scr_e_true)}
<h3>Energy world · estimated labels</h3>{scr_table(scr_e_est)}
<h3>Null world · true labels</h3>{scr_table(scr_0_true)}
<h3>Null world · estimated labels</h3>{scr_table(scr_0_est)}
<p>Three readings. The machinery retains the planted effect and nothing in the null world. Against estimated labels the engineered proxies lose their value while the true pressure keeps it — label noise, not the test, is what they fail on. And the feedback proxies, which measure something real about the path but nothing causal in this world, are correctly rejected everywhere: the baseline is doing its job.</p>

<!-- 13 -->
<h2><span class="n">13</span>Effect size and power {BUILT}</h2>
<p>How strong does a causal effect have to be before the rule retains it on ten years of one pair? The generator's <code>energy_beta</code> scales the tilt; the share of days with any pressure at all is 7–14%, and escalations are rare events, so the tests are low-powered by construction.</p>
{figure(figs['power'], 9, "Both tests against effect size for the true pressure, the gap feature and the product, with true (solid) and estimated (dashed) labels. The true pressure clears both thresholds only from an effect size of about 3; the proxies only with true labels. At realistic effect sizes the rule will reject true signals. That is deliberate: a leading feature that enters the pressure index tilts every fan on the card, and the cost of a false one is borne on every trade.")}
{table(power[power.feature == 'pressure_true'][['energy_beta', 'labels', 'share_pressure', 'delta_r2', 'folds_r2', 'transition_gain', 'folds_tr', 'retain']], fmt_pow)}
<div class="box"><b>A fragility worth knowing.</b> The outcome test is sensitive to how the extreme days are handled. In the energy world the {n_extreme} extreme days carry the largest forward earn by far; with their own state dummy the baseline absorbs them and the gap feature's ΔR² is {r_kept['delta_r2']:+.4f} in {r_kept['folds_r2_up']} folds, but if extreme is merged into stressed before the test the same feature reads {r_merged['delta_r2']:+.4f} in {r_merged['folds_r2_up']} fold(s), because the variance of those days is then attributed to whatever the feature happens to be doing near them. The band exists for the ladder, but it also protects this test. On real data, run the screen with the band kept (the finder's <code>6x</code> labels) whenever it has episodes.</div>

<!-- 14 -->
<h2><span class="n">14</span>The pressure index and what it does downstream {BUILT}</h2>
<p>The retained features are averaged (equal weights, or weights from config) and centred: ψ = (mean − 50) / 25, so ψ ≈ 0 is an ordinary day and ψ ≈ +2 the top of the range. ψ enters the transition tilt with its fitted β and the declared path (decaying with a 10-day half-life by default — energy dissipates), and the card shows P(stressed or extreme) at 5 / 10 / 21 days under the constant matrix and under the tilt, side by side. Nothing else on the card changes: the ladder's numbers never depend on a leading feature.</p>
{figure(figs['pressure'], 10, f"The pressure index in the energy world (left) and the fan from carry under the fitted tilt at three pressure levels with the decaying path (right). β = {fit['beta']:+.2f} per standard deviation: the tilt is modest, which is what a weak, noisy, correctly-sized signal should produce.")}

<!-- 15 -->
<h2><span class="n">15</span>Live · build your own energy index {BUILT}</h2>
<div class="demo"><div class="hd">Live · choose components and weights, run both tests in the browser</div>
<p class="dim" style="margin:0 0 8px">Weight the six candidates (0 = out). The index is their weighted mean, standardised. The panel runs the outcome test (OLS with state dummies + composite + composite², four folds) and the transition test (5-step tilt likelihood with the composite baseline γ fixed per fold) exactly as the Python does, against true or estimated labels in the energy world, and reports the retention verdict. Try gap alone, the product alone, and a mix.</p>
<div class="ctl">
  {''.join(f'<div><label>{c}: <b id="w_{c}_v">{1 if c == "stored_energy" else 0}</b></label><input type="range" id="w_{c}" min="0" max="3" step="0.5" value="{1 if c == "stored_energy" else 0}"></div>' for c in Lall.columns)}
  <div><label>labels</label><select id="lab"><option value="true">true</option><option value="estimated">estimated</option></select></div>
</div>
<div class="stat"><div>corr with hidden pressure<b id="l_corr">—</b></div><div>ΔR² OOS<b id="l_dr2">—</b></div><div>folds up<b id="l_f1">—</b></div><div>transition gain / obs<b id="l_gain">—</b></div><div>folds up<b id="l_f2">—</b></div><div>β (mean over folds)<b id="l_beta">—</b></div><div>retained?<b id="l_ret">—</b></div></div>
<canvas id="cv_idx" width="1600" height="260"></canvas>
<p class="dim" style="margin-top:6px">Days 1300–2000: the index (black), the hidden pressure (orange, rescaled), and the state ribbon.</p>
</div>

<!-- 16 -->
<h2><span class="n">16</span>On real data {BUILT} <span class="dim">WU-16, WU-17</span></h2>
<ol>
<li><b>Run the screen on the whole history per pair</b>, with estimated labels — never true ones, there are none — and both targets if the backtester window is short. <code>leading_screen.csv</code> is the deliverable; the retain column is the only one that feeds anything.</li>
<li><b>Add the desk's series one at a time</b> through the contract of Section 8 and read their rows next to the generic ones. A proprietary series that fails the same test the generic ones fail is information about the series, not about the test.</li>
<li><b>Score at the real delay.</b> If a series arrives next morning, its row is the <code>lag = 1</code> one.</li>
<li><b>Expect to retain nothing at first.</b> Section 13 says why. A pair with no retained feature keeps the constant matrix and the card says so; Gate 4 failing does not touch Gates 2 or 3.</li>
<li><b>If something is retained,</b> declare its path (D19), build the pressure index from the retained set only, fit β with the baseline on the full history, and write the by-fold table next to the fan. The card then carries both the constant and the tilted P(stressed or extreme).</li>
<li><b>Re-run the screen when the labels are recalibrated.</b> The baseline changes with the labels; a feature's verdict is conditional on the labeller it was tested against.</li>
</ol>
<div class="box warn"><b>Gate 4.</b> At least one feature retained by both tests in ≥ 3 of 4 folds, and the pressure index built from the retained features improves the out-of-sample 5-step transition likelihood. Fail → the constant matrix stands; do not loosen <code>gates.yaml</code>.</div>

<!-- 17 -->
<h2><span class="n">17</span>Code map</h2>
<table><tr><th>function</th><th>does</th></tr>
<tr><td><code>leading.gap_z · gap_pct</code></td><td>the gap in implied units and its trailing percentile</td></tr>
<tr><td><code>leading.complacency</code></td><td>the mean of three holder-weakness percentiles</td></tr>
<tr><td><code>leading.stored_energy</code></td><td>the product</td></tr>
<tr><td><code>leading.jump_clustering · coupling_shift · cross_pair_coherence</code></td><td>the feedback proxies</td></tr>
<tr><td><code>leading.event_proximity(index, events, horizon)</code></td><td>the calendar covariate</td></tr>
<tr><td><code>leading.LEADING · leading.build(m, names, events, **params)</code></td><td>the registry and the builder</td></tr>
<tr><td><code>leading.incremental_value(feature, labels, state_score, target, lag, k)</code></td><td>both tests for one feature</td></tr>
<tr><td><code>leading.retain(res, cfg) · leading.screen(candidates, …)</code></td><td>the rule and the table</td></tr>
<tr><td><code>leading.pressure(L, weights)</code></td><td>the pressure index</td></tr>
<tr><td><code>transitions.fit_tilt · oos_gain · fan · covariate_path</code></td><td>what the index does downstream</td></tr>
<tr><td><code>gates.gate_phase4(screen, pressure_gain, cfg)</code></td><td>Gate 4</td></tr>
<tr><td><code>synth.simulate_market(..., energy_beta)</code></td><td>the causal world (and, at 0, the null)</td></tr>
<tr><td><code>python -m regime_ladder leading … · inspect</code></td><td>the CLI; <code>leading_features.csv</code>, <code>leading_screen.csv</code>, <code>tilt_fits.csv</code>, <code>pressure.csv</code></td></tr>
</table>
"""

js = r"""
const S=D.states, COL=D.colours, FE=Object.keys(D.feat), N=D.fwd5.length;
function zs(a){const v=a.filter(x=>x!=null);const mu=v.reduce((p,q)=>p+q,0)/v.length,sd=Math.sqrt(v.reduce((p,q)=>p+(q-mu)**2,0)/(v.length-1));return a.map(x=>x==null?null:(x-mu)/sd);}
function corr(a,b){let n=0,sa=0,sb=0,saa=0,sbb=0,sab=0;for(let i=0;i<a.length;i++){if(a[i]==null||b[i]==null)continue;n++;sa+=a[i];sb+=b[i];saa+=a[i]*a[i];sbb+=b[i]*b[i];sab+=a[i]*b[i];}const c=(sab-sa*sb/n)/Math.sqrt((saa-sa*sa/n)*(sbb-sb*sb/n));return c;}
// OLS via normal equations with tiny ridge for stability
function solve(A,b){const n=A.length;const M=A.map((r,i)=>[...r,b[i]]);for(let i=0;i<n;i++){let p=i;for(let r=i+1;r<n;r++)if(Math.abs(M[r][i])>Math.abs(M[p][i]))p=r;[M[i],M[p]]=[M[p],M[i]];const d=M[i][i]||1e-12;for(let r=0;r<n;r++){if(r===i)continue;const f=M[r][i]/d;for(let c=i;c<=n;c++)M[r][c]-=f*M[i][c];}}return M.map((r,i)=>r[n]/(r[i]||1e-12));}
function ols(X,y){const p=X[0].length;const XtX=Array.from({length:p},()=>new Array(p).fill(0)),Xty=new Array(p).fill(0);for(let i=0;i<X.length;i++){for(let a=0;a<p;a++){Xty[a]+=X[i][a]*y[i];for(let b=0;b<p;b++)XtX[a][b]+=X[i][a]*X[i][b];}}for(let a=0;a<p;a++)XtX[a][a]+=1e-8;return solve(XtX,Xty);}
function oosR2(X,y,folds){const n=y.length;const cuts=[];for(let f=0;f<=folds;f++)cuts.push(Math.round(0.4*n+(n-0.4*n)*f/folds));let sse=0,sst=0;const pf=[];
  for(let f=0;f<folds;f++){const a=cuts[f],b=cuts[f+1];const beta=ols(X.slice(0,a),y.slice(0,a));const my=y.slice(0,a).reduce((p,q)=>p+q,0)/a;let s1=0,s0=0;for(let i=a;i<b;i++){const pred=X[i].reduce((p,v,j)=>p+v*beta[j],0);s1+=(y[i]-pred)**2;s0+=(y[i]-my)**2;}sse+=s1;sst+=s0;pf.push(1-s1/s0);}return {r2:1-sse/sst,pf};}
function outcomeTest(fz,lab){const cnt={};for(let i=0;i<N;i++){if(fz[i]==null||D.comp_z[i]==null||D.fwd5[i]==null)continue;cnt[lab[i]]=(cnt[lab[i]]||0)+1;}const ref=Object.keys(cnt).sort((a,b)=>cnt[b]-cnt[a])[0];const SS=S.filter(s=>s!==ref);
  const rows=[],ys=[];for(let i=0;i<N;i++){if(fz[i]==null||D.comp_z[i]==null||D.fwd5[i]==null)continue;const d=SS.map(s=>lab[i]===s?1:0);rows.push([1,...d,D.comp_z[i],D.comp_z[i]**2,fz[i]]);ys.push(D.fwd5[i]);}
  const base=rows.map(r=>r.slice(0,-1));const r0=oosR2(base,ys,4),r1=oosR2(rows,ys,4);return {dr2:r1.r2-r0.r2,folds:r1.pf.filter((v,i)=>v>r0.pf[i]).length};}
// k-step transition likelihood with baseline gamma (fixed per fold)
const RK=D.ranks,K=S.length;
function loglik(idxs,si,eff,P0,k){let ll=0;for(const t of idxs){const T=tilt(P0,RK,eff[t]);const Tk=matpow(T,k);ll+=Math.log(Math.max(Tk[si[t]][si[t+k]],1e-300));}return ll;}
function transitionTest(fz,lab,folds){const si=lab.map(s=>S.indexOf(s));const k=5;let tot=0,ntot=0,pos=0,betas=[];
  for(const F of folds){const idxTr=[],idxTe=[];for(let t=0;t<N-k;t++){if(fz[t]==null||D.comp_z[t]==null||si[t]<0||si[t+k]<0)continue;if(t+k<F.a)idxTr.push(t);else if(t>=F.a-k&&t+k<F.b)idxTe.push(t);}
    const effFor=b=>D.comp_z.map((c,t)=>F.gamma*(c||0)+b*(fz[t]||0));
    let best=-Infinity,bb=0;for(let b=-3;b<=3.001;b+=0.25){const v=loglik(idxTr,si,effFor(b),F.P,k);if(v>best){best=v;bb=b;}}
    let lo=bb-0.25,hi=bb+0.25;for(let it=0;it<12;it++){const m1=lo+(hi-lo)/3,m2=hi-(hi-lo)/3;if(loglik(idxTr,si,effFor(m1),F.P,k)<loglik(idxTr,si,effFor(m2),F.P,k))lo=m1;else hi=m2;}bb=(lo+hi)/2;
    const g=(loglik(idxTe,si,effFor(bb),F.P,k)-loglik(idxTe,si,effFor(0),F.P,k));tot+=g;ntot+=idxTe.length;if(g>0)pos++;betas.push(bb);}
  return {gain:tot/ntot,folds:pos,beta:betas.reduce((p,q)=>p+q,0)/betas.length};}
function runIdx(){const w={};let sum=0;FE.forEach(c=>{w[c]=+document.getElementById('w_'+c).value;document.getElementById('w_'+c+'_v').textContent=w[c];sum+=w[c];});
  const lab=D.labels[document.getElementById('lab').value],folds=D.folds[document.getElementById('lab').value];
  if(sum===0){l_corr.textContent="—";return;}
  const idx=D.feat[FE[0]].map((_,i)=>{let s=0,ok=true;FE.forEach(c=>{if(w[c]===0)return;const v=D.feat[c][i];if(v==null)ok=false;else s+=w[c]*v;});return ok?s/sum:null;});
  const fz=zs(idx);l_corr.textContent=corr(idx,D.pressure_true).toFixed(3);
  const o=outcomeTest(fz,lab);l_dr2.textContent=(o.dr2>=0?"+":"")+o.dr2.toFixed(4);l_f1.textContent=o.folds+"/4";
  const tr=transitionTest(fz,lab,folds);l_gain.textContent=(tr.gain>=0?"+":"")+tr.gain.toFixed(4);l_f2.textContent=tr.folds+"/4";l_beta.textContent=(tr.beta>=0?"+":"")+tr.beta.toFixed(2);
  const ret=o.dr2>=D.thr.min_delta_r2&&o.folds>=D.thr.min_folds_up&&tr.gain>=D.thr.min_transition_gain&&tr.folds>=D.thr.min_folds_up;l_ret.innerHTML=ret?'<span class="ok">yes</span>':'<span class="no">no</span>';
  const a=1300,b=2000;const cv=document.getElementById('cv_idx');const idxS=idx.slice(a,b),pt=D.pressure_true.slice(a,b);const mx=Math.max(...idxS.filter(v=>v!=null));
  const r=lines(cv,[{name:"index (0–100)",y:idxS,color:"#222"},{name:"hidden pressure (×30)",y:pt.map(v=>v*30),color:"#E8743B",lw:1.2}],{pad:[16,10,40,52],ymin:0,ymax:Math.max(100,mx),xlabels:[[0,"1300"],[350,"1650"],[699,"2000"]]});
  const c=cv.getContext('2d');for(let i=a;i<b;i++){c.fillStyle=COL[lab[i]]||"#999";c.fillRect(r.X(i-a),cv.height-36,2.4,10);}}
FE.forEach(c=>document.getElementById('w_'+c).addEventListener('input',runIdx));document.getElementById('lab').addEventListener('input',runIdx);runIdx();
"""

toc = [("1", "The question and the constraint"), ("2", "The anatomy"), ("3", "Stored energy I · the gap"), ("4", "Stored energy II · complacency"), ("5", "Stored energy III · the product"), ("6", "Feedback proxies"),
       ("7", "The calendar covariate"), ("8", "The contract"), ("9", "Retention I · outcome test"), ("10", "Retention II · transition test"), ("11", "The lag check"), ("12", "Positive and negative controls"),
       ("13", "Effect size and power"), ("14", "The pressure index downstream"), ("15", "Live · build your own index"), ("16", "On real data"), ("17", "Code map")]
out = page("leading.html", "Leading features and stored energy",
           "What might push the market out of its state, measured so that it can be tested: the stored-energy hypothesis made into a feature (a stretched spot held by a complacent surface), the feedback proxies, the calendar, the contract any desk series must meet, the two-part retention rule with the state and the continuous score as its baseline, the synthetic worlds that act as positive and negative controls, and how little power ten years of one pair actually gives. The live panel runs both retention tests in the browser.",
           toc, body, data, js)
print(out, kept, fit["beta"])
