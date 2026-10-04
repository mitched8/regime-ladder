"""Deep dive 2: transitions.html — Transition probabilities."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from common import world, energy_world, figure, save, table, page, COLOURS, BUILT, HYP, ROOT
import sys; sys.path.insert(0, str(ROOT))
from regime_ladder import synth, labels, ladder, transitions, leading

W = world(); m, comp, st, est, spec, td = W["m"], W["comp"], W["st"], W["est"], W["spec"], W["td"]
C = COLOURS; S6 = list(labels.STATES6)
names = labels.state_order(est)
Cnt = labels.transition_matrix(est, names=names, prior_strength=0.0)
counts = pd.DataFrame(0.0, index=names, columns=names)
v = est.map({n: i for i, n in enumerate(names)}).values
for a, b in zip(v[:-1], v[1:]): counts.iloc[int(a), int(b)] += 1
P10 = labels.transition_matrix(est, names=names, prior_strength=10)
P_true = labels.transition_matrix(st, names=labels.state_order(st), prior_strength=0.0)
figs = {}

# ---- F1 matrices: counts, prior 0 / 10 / 50, and the true generator
fig, axs = plt.subplots(1, 4, figsize=(12, 3.3))
def heat(ax, M, title, fmt="{:.2f}", cmap="Blues"):
    im = ax.imshow(M.values, cmap=cmap, vmin=0, vmax=1, aspect="auto"); ax.grid(False)
    ax.set_xticks(range(len(M.columns))); ax.set_xticklabels([c[:5] for c in M.columns], rotation=0, fontsize=7); ax.set_yticks(range(len(M.index))); ax.set_yticklabels(M.index, fontsize=7)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]): ax.text(j, i, fmt.format(M.values[i, j]), ha="center", va="center", fontsize=6.3, color="#fff" if M.values[i, j] > 0.6 else "#111")
    ax.set_title(title, fontsize=9)
heat(axs[0], counts / counts.values.max(), "transition counts (10y, estimated labels)", fmt="{:.0f}") ;
for i in range(len(names)):
    for j in range(len(names)): axs[0].texts[i * len(names) + j].set_text(f"{int(counts.values[i, j])}")
heat(axs[1], Cnt, "P, no prior"); heat(axs[2], labels.transition_matrix(est, names=names, prior_strength=50), "P, sticky prior 50")
Ptrue_r = P_true.reindex(index=names, columns=names).fillna(0)
heat(axs[3], Ptrue_r, "P from the TRUE labels", cmap="Greens")
figs["mats"] = save(fig, "tr_mats.png")

# ---- F2 hazard curves: P(leave | survived d days) for estimated vs true labels
def hazard(lab, maxd=30):
    d = labels.durations(lab); out = {}
    for s, runs in d.items():
        runs = np.array(runs); h = []
        for k in range(1, maxd + 1):
            alive = (runs >= k).sum(); end = (runs == k).sum()
            h.append(end / alive if alive >= 15 else np.nan)
        out[s] = np.array(h)
    return out
hz_e, hz_t = hazard(est), hazard(st)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
for ax, hz, title in [(axs[0], hz_t, "true labels (a Markov chain): flat hazards"), (axs[1], hz_e, "estimated labels: hazards that rise and fall")]:
    for s in S6:
        if s in hz: ax.plot(range(1, 31), hz[s], "-", color=C[s], lw=1.4, label=s)
    ax.set_xlabel("days already in the state"); ax.set_title(title); ax.set_ylim(0, 0.5)
axs[0].set_ylabel("P(leave tomorrow | still here)"); axs[0].legend(frameon=False, fontsize=7.5, ncol=2)
figs["hazard"] = save(fig, "tr_hazard.png")

# ---- F3 durations + persistence split vs geometric survival
dur = transitions.implied_vs_observed_duration(P10, est)
sp = ladder.persistence_split(td, W["lab_est"], horizons=(5, 10, 20))
sp = sp[sp.archetype == "straddle_atm"]
fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
k6 = len(names)
axs[0].bar(np.arange(k6) - 0.18, dur["implied_days"], 0.36, color="#9aa0a6", label="1/(1−p_ii): geometric"); axs[0].bar(np.arange(k6) + 0.18, dur["observed_mean_days"], 0.36, color="#4F46E5", label="observed mean run")
axs[0].set_xticks(range(k6)); axs[0].set_xticklabels(names, rotation=20, fontsize=8); axs[0].set_ylabel("days"); axs[0].set_title("Expected duration: chain vs observed"); axs[0].legend(frameon=False, fontsize=8)
for s in names:
    g = sp[sp.regime == s].sort_values("h")
    if len(g): axs[1].plot(g.h, g.q_broke, "o-", color=C[s], ms=4, label=f"{s} observed"); axs[1].plot(g.h, 1 - P10.loc[s, s] ** g.h, ":", color=C[s], lw=1.2)
axs[1].set_xlabel("h"); axs[1].set_ylabel("P(entry state broke by h)"); axs[1].set_title("The ladder's break frequency q (solid) vs 1 − p_ii^h (dotted)"); axs[1].legend(frameon=False, fontsize=6.5, ncol=2); axs[1].set_xticks([5, 10, 20])
figs["dur"] = save(fig, "tr_dur.png")

# ---- F4 fans from each state + stationary distribution
fig, axs = plt.subplots(2, 3, figsize=(10.5, 5.2), sharex=True, sharey=True)
pi_stat = np.linalg.matrix_power(P10.values, 2000)[0]
for ax, s in zip(axs.ravel(), names[:6]):
    f = transitions.fan(P10, pd.Series({s: 1.0}), psi_path=np.zeros(63))
    ax.stackplot(f.index, f.T.values, colors=[C[c] for c in f.columns], alpha=0.9); ax.set_title(f"from {s}", fontsize=9)
    for i, c in enumerate(f.columns): ax.axhline(pi_stat[:i + 1].sum(), color="#fff", lw=0.5, alpha=0.6)
for ax in axs[1]: ax.set_xlabel("days ahead")
for ax in axs[:, 0]: ax.set_ylabel("P(state at h)")
fig.suptitle("Fans from each state under the constant matrix (thin white lines: the stationary distribution the fans converge to)", fontsize=10, fontweight="semibold")
figs["fans"] = save(fig, "tr_fans.png")

# ---- F5 the tilt: valid form vs the shortcut
ranks = transitions.ranks(names)
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.2))
row = P10.loc["carry"]
psis = np.linspace(0, 3, 61)
valid = np.array([transitions.tilt(P10, p, 1.0).loc["carry"].values for p in psis])
def shortcut(P, psi, lam=1.0):
    T = P.values.copy(); off = ~np.eye(len(P), dtype=bool); T[off] *= np.exp(lam * psi); np.fill_diagonal(T, 1 - (T * off).sum(1)); return T
short = np.array([shortcut(P10, p)[names.index("carry")] for p in psis])
for j, s in enumerate(names):
    axs[0].plot(psis, valid[:, j], color=C[s], lw=1.4, label=s); axs[1].plot(psis, short[:, j], color=C[s], lw=1.4)
axs[0].set_title("row 'carry' under the logit tilt"); axs[0].set_xlabel("β·ψ"); axs[0].set_ylabel("P(carry → ·)"); axs[0].legend(frameon=False, fontsize=7, ncol=2)
axs[1].set_title("the shortcut: scale off-diagonals, diagonal = remainder"); axs[1].set_xlabel("β·ψ"); axs[1].axhline(0, color="#333", lw=0.8); axs[1].text(1.6, -0.25, "p_ii < 0: not a probability", color="#9a3412", fontsize=8)
# k-step vs one-step from carry at psi=1
for k in (1, 5, 10):
    T = transitions.tilt(P10, 1.0, 1.0).values; Tk = np.linalg.matrix_power(T, k)
    axs[2].bar(np.arange(k6) + (k == 5) * 0.27 + (k == 10) * 0.54 - 0.27, Tk[names.index("carry")], 0.27, color={1: "#cbd5e1", 5: "#4F46E5", 10: "#1e1b4b"}[k], label=f"{k}-step")
axs[2].set_xticks(range(k6)); axs[2].set_xticklabels(names, rotation=20, fontsize=7.5); axs[2].set_title("from carry, β·ψ = 1: where the state is k days out"); axs[2].legend(frameon=False, fontsize=7.5)
figs["tilt"] = save(fig, "tr_tilt.png")

# ---- F6 the wrong-sign result: beta with and without the baseline, k = 1, 5, 10, true vs estimated labels (energy world)
E = energy_world(3.0); me, ste, este, compe = E["m"], E["st"], E["est"], E["comp"]
psi_true = (me["pressure_true"] - me["pressure_true"].mean()) / me["pressure_true"].std()
cz = (compe - compe.mean()) / compe.std()
Pe_t = labels.transition_matrix(ste, names=labels.state_order(ste), prior_strength=10); Pe_e = labels.transition_matrix(este, names=labels.state_order(este), prior_strength=10)
rows = []
for ln, lab, P_ in (("true labels", ste, Pe_t), ("estimated labels", este, Pe_e)):
    for k in (1, 5, 10):
        for bn, base in (("no baseline", None), ("with composite baseline γ", cz)):
            f = transitions.fit_tilt(lab, psi_true, P_, k=k, base=base)
            g = transitions.oos_gain(lab, psi_true, folds=4, k=k, base=base)
            rows.append(dict(labels=ln, k=k, baseline=bn, beta=f["beta"], gamma=f["gamma"], oos_gain=g.attrs["total_gain_per_transition"], folds_positive=g.attrs["folds_positive"]))
fits = pd.DataFrame(rows)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.3), sharey=False)
for ax, ln in zip(axs, ("true labels", "estimated labels")):
    g = fits[fits.labels == ln]
    for j, (bn, col) in enumerate([("no baseline", "#9aa0a6"), ("with composite baseline γ", "#4F46E5")]):
        gg = g[g.baseline == bn]; ax.bar(np.arange(3) + (j - 0.5) * 0.36, gg.beta, 0.36, color=col, label=bn)
        for i, (b, gain) in enumerate(zip(gg.beta, gg.oos_gain)): ax.text(i + (j - 0.5) * 0.36, b + (0.005 if b >= 0 else -0.02), f"gain {gain:+.4f}", ha="center", fontsize=6.5)
    ax.axhline(0, color="#333", lw=0.8); ax.set_xticks(range(3)); ax.set_xticklabels(["k = 1", "k = 5", "k = 10"]); ax.set_title(f"β for the TRUE pressure, {ln}"); ax.set_ylabel("fitted β")
axs[0].legend(frameon=False, fontsize=8)
figs["wrongsign"] = save(fig, "tr_wrongsign.png")

# ---- F7 OOS gain by fold (estimated labels, k=5, with baseline) for true pressure and gap_pct
Le = leading.build(me, names=["gap_pct", "stored_energy"])
gain_rows = []
for nm, series in (("true pressure", psi_true), ("gap_pct", (Le["gap_pct"] - Le["gap_pct"].mean()) / Le["gap_pct"].std())):
    g = transitions.oos_gain(este, series, folds=4, k=5, base=cz); g.insert(0, "feature", nm); gain_rows.append(g)
gains = pd.concat(gain_rows)

# ---- F8 covariate paths
fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
H = 63; beta_e = transitions.fit_tilt(ste, psi_true, Pe_t, k=5, base=cz)["beta"]
for pol, ls, lab_ in [("frozen", "--", "frozen at +2"), ("decay", ":", "+2 decaying, half-life 10d"), ("zero", "-", "zero (constant matrix)")]:
    path = transitions.covariate_path(2.0, H, pol, 10); axs[0].plot(range(1, H + 1), path, ls, color="#222", lw=1.3, label=lab_)
cal = np.zeros(H); cal[[9, 10, 11, 30, 31]] = 2.0; axs[0].plot(range(1, H + 1), cal, "-", color="#E8743B", lw=1.3, label="calendar: two known events")
axs[0].set_xlabel("days ahead"); axs[0].set_ylabel("ψ along the path"); axs[0].set_title("Declared future paths of a covariate"); axs[0].legend(frameon=False, fontsize=7.5)
hi = [s for s in ("stressed", "extreme") if s in Pe_t.index]
for pol, ls, lab_, path in [("zero", "-", "constant", np.zeros(H)), ("frozen", "--", "frozen +2", np.full(H, 2.0)), ("decay", ":", "decaying +2", transitions.covariate_path(2.0, H, "decay", 10)), ("calendar", "-", "calendar", cal)]:
    f = transitions.fan(Pe_t, pd.Series({"carry": 1.0}), psi_path=path, beta=beta_e); axs[1].plot(f.index, f[hi].sum(axis=1), ls, color="#E8743B" if pol != "zero" else "#555", lw=1.4, label=lab_)
axs[1].set_xlabel("days ahead"); axs[1].set_ylabel("P(stressed or extreme at h)"); axs[1].set_title(f"From carry under each path (β = {beta_e:+.2f}, energy world)"); axs[1].legend(frameon=False, fontsize=7.5)
figs["paths"] = save(fig, "tr_paths.png")

# ---- data for the live demo
Mfull = P10.reindex(index=S6, columns=S6).fillna(0.0); Cfull = counts.reindex(index=S6, columns=S6).fillna(0.0)
for s in S6:
    if Mfull.loc[s].sum() == 0: Mfull.loc[s, s] = 1.0
data = {"states": S6, "colours": {s: C[s] for s in S6}, "counts": [[float(x) for x in r] for r in Cfull.values], "ranks": [transitions.RANK[s] for s in S6], "beta": round(float(beta_e), 3),
        "pi_mix": [0.45, 0.15, 0.25, 0.05, 0.03, 0.07]}

fmt_fits = {"beta": lambda v: f"{v:+.3f}", "gamma": lambda v: f"{v:+.3f}", "oos_gain": lambda v: f"{v:+.4f}", "k": str, "folds_positive": lambda v: f"{int(v)}/4"}
fmt_gain = {"beta": lambda v: f"{v:+.3f}", "gamma": lambda v: f"{v:+.3f}", "oos_gain_per_transition": lambda v: f"{v:+.4f}", "fold": str, "n_test": str, "train_end": lambda v: str(v)[:10]}
dur_show = dur.reset_index()[["state", "implied_days", "observed_mean_days", "episodes"]]

body = f"""
<!-- 1 -->
<h2><span class="n">1</span>What the matrix is for, and what it is not {BUILT}</h2>
<p>A trade entered in one state lives through others. The horizon ladder measures that directly — transitions inside the window are in the outcomes it averages — so for a <b>fresh unit</b> no transition model is needed and none is used: the number on the card does not depend on anything in this page. The matrix earns its place for three other things.</p>
<ol>
<li><b>Explaining the ladder.</b> How long does carry usually last from here? What is the chance a unit entered in agitated sees stressed before expiry? The regime-dependence split on the card (persisted / broke) is the empirical version; the matrix is the model that should agree with it (Section 4).</li>
<li><b>The path model</b> (Phase 5): simulating state paths for an existing book of arbitrary age, where the ladder cannot reach.</li>
<li><b>Incorporating today's leading signals</b>, where today's propensity to move differs from the state's historical average. This is the only place in the design where a forward-looking input enters, and it enters as a tilt of this matrix, under a declared future path, and only from features that survived the retention rule (the leading-features page).</li>
</ol>
<div class="box"><b>The object.</b> P is a k × k matrix over the named states: P<sub>ij</sub> = P(state tomorrow = j | state today = i). Rows sum to one. Everything here is per trading day; multi-day questions are powers of P (constant) or products of tilted matrices (time-varying).</div>

<!-- 2 -->
<h2><span class="n">2</span>Estimation: counts and the sticky prior {BUILT}</h2>
<p>P is estimated from the ten-year <i>estimated</i> label history by counting transitions. Two things make the raw counts a poor estimate on their own: rare states have few exits (normalising here has ~{int(counts.loc['normalising'].sum()) if 'normalising' in counts.index else 0} days and a dozen exits), and a hysteretic labeller produces zero counts for transitions that are merely unobserved, not impossible. The <b>sticky prior</b> adds <code>prior_strength</code> pseudo-observations to each row, 95% on the diagonal and the rest spread over the off-diagonals — a Dirichlet prior that says "states persist, and anything can follow anything, rarely". With ten pseudo-observations against 100–800 real ones per row it changes the busy rows by nothing and stops the rare rows from asserting impossibilities.</p>
<div class="eq">P_ij  =  ( C_ij + κ · prior_ij ) / ( Σ_j C_ij + κ )          prior_ii = 0.95,  prior_ij = 0.05 / (k − 1),  κ = prior_strength</div>
{figure(figs['mats'], 1, "Left to right: the raw transition counts from the estimated labels; the matrix with no prior; with a prior of 50 (visibly pulled toward the diagonal on the rare rows); and the matrix of the true generating chain for comparison. The estimated matrix is stickier than the truth on every diagonal — a smoothed, hysteretic labeller makes states look more persistent than they are, and makes the short true episodes vanish into their neighbours.")}

<!-- 3 -->
<h2><span class="n">3</span>Is it a Markov chain? Durations and hazards {BUILT}</h2>
<p>A first-order chain implies a geometric holding time with mean 1/(1 − p<sub>ii</sub>) and a <i>flat hazard</i>: the chance of leaving tomorrow does not depend on how long you have been in the state. Neither is true of a hysteretic labeller. Its exit probability depends on the distance of the smoothed score from the band boundary and on how long the score has been drifting, so the hazard is low just after entry (the score has just crossed the boundary plus δ and must come all the way back) and rises with time in state. The check is cheap and is run before any geometric holding time is relied on.</p>
{figure(figs['hazard'], 2, "Empirical hazard — P(leave tomorrow | still in the state after d days) — from the true labels (left), which are a Markov chain by construction and show flat hazards at the generator's 1 − p<sub>ii</sub>, and from the estimated labels (right), where the hazard is near zero for the first few days of every state and climbs after. The chain is a usable approximation for horizons of a week or more; for the first days after a switch it understates persistence.")}
{figure(figs['dur'], 3, "Left: the geometric holding time each diagonal implies against the observed mean run of the estimated labels — close for the busy states, off for the rare ones. Right: the ladder's own regime-dependence split for the straddle — the observed share of entries whose state had broken by h (solid) — against the chain's 1 − p<sub>ii</sub><sup>h</sup> (dotted). Where the two agree, the matrix is explaining what the ladder measured; where they do not, the ladder is right and the matrix is the approximation.")}
{table(dur_show, {"implied_days": lambda v: f"{v:.1f}", "observed_mean_days": lambda v: f"{v:.1f}", "episodes": lambda v: f"{int(v)}"})}

<!-- 4 -->
<h2><span class="n">4</span>Multi-step: fans and the stationary distribution {BUILT}</h2>
<p>P<sup>h</sup> gives the state distribution h days ahead from any start; a start can be a point mass (today's label) or a probability vector π (today's label blended with the shock detectors). The <b>fan</b> is that distribution over h = 1…63; its row sums over the next 21 days are the <b>expected days in each state</b>; and every fan converges to the same <b>stationary distribution</b>, which is the unconditional share of time in each state. How fast it converges is the chain's memory — on the synthetic labels, half the distance to stationarity is covered in roughly {int(round(np.log(0.5) / np.log(sorted(np.abs(np.linalg.eigvals(P10.values)))[-2])))} days.</p>
{figure(figs['fans'], 4, "The fan from each of the six states under the constant matrix. The information in today's state is the gap between its fan and the stationary distribution (white lines); by two to three months the start no longer matters. This is also the honest statement of how far ahead a constant-matrix forecast can say anything.")}

<!-- 5 -->
<h2><span class="n">5</span>The tilt: letting today's pressure change tonight's odds {BUILT}</h2>
<p>Time variation enters through a single scalar <b>pressure</b> ψ<sub>t</sub> (the pressure index of the leading-features page, or any one covariate under test) and one sensitivity β, through a <b>row-normalised multinomial logit</b>:</p>
<div class="eq">P_ij(ψ)  =  P⁰_ij · exp( β ψ (rank_j − rank_i) )  /  Σ_k P⁰_ik · exp( β ψ (rank_k − rank_i) )

ranks:  carry 0 · settling 0.5 · rising 1 · agitated 1.5 · normalising 2 · stressed 3 · extreme 4</div>
<p>Three properties make this the only acceptable form. Every entry stays strictly inside (0, 1) for any ψ. Every row sums to one by construction. And the effect is <i>monotone and ordered</i>: a positive β·ψ moves mass from every state toward higher-ranked ones and away from lower-ranked ones, by an amount that grows with the rank distance, so one number tilts the whole matrix coherently instead of twenty-nine. The tempting shortcut — multiply the off-diagonals by a factor and set the diagonal to whatever is left — is not a probability model: for a large enough tilt the diagonal goes negative, and before that it moves mass symmetrically toward "leaving", with no sense of direction.</p>
{figure(figs['tilt'], 5, "Left: the carry row of the estimated matrix as β·ψ rises under the logit tilt — mass flows to rising, agitated and stressed in rank order, carry's own probability falls smoothly and never below zero. Centre: the shortcut on the same row, which drives the diagonal through zero. Right: from carry at β·ψ = 1, the distribution one, five and ten days out — the k-step view the fit uses.")}

<!-- 6 -->
<h2><span class="n">6</span>Fitting β: the k-step likelihood {BUILT}</h2>
<p>β is fitted by maximum likelihood on the observed transitions, with ψ aligned to the <i>from</i> date (today's pressure governs the moves from today). The naive version scores tonight's move: Σ<sub>t</sub> log P<sub>s<sub>t</sub> s<sub>t+1</sub></sub>(ψ<sub>t</sub>). Against labels the market actually produces that is the wrong question, because a smoothed labeller's one-step moves lag the market by days (states page, Figure 6): the day the pressure was high is rarely the night the label moves. The <b>k-step likelihood</b> asks where the state is k days out, under the tilted matrix held at today's ψ for k days:</p>
<div class="eq">ℓ(β)  =  Σ_t  log [ P(ψ_t ; β)^k ]_( s_t , s_(t+k) )            k = 5 by default; one-dimensional, solved by a grid and golden-section refinement</div>
<p>Overlapping k-step observations are not independent, so nothing here is a p-value; the likelihood is used to compare models on the same folds, which is all the retention rule needs. k = 5 is the primary horizon of the ladder and roughly the labeller's median lag, which is not a coincidence.</p>

<!-- 7 -->
<h2><span class="n">7</span>The baseline covariate, and the wrong sign {BUILT}</h2>
<p>This is the result that shaped the design. In the synthetic world where stored energy is made causal, the generator's <i>own</i> pressure — the quantity that literally tilts the true chain — was fitted against the <i>estimated</i> labels. At k = 1 it came out with a <b>negative</b> β: the true pressure appeared to make escalation <i>less</i> likely. The reason is a confound, not a bug. Pressure is highest when vol is low (that is what "a gap held by a complacent surface" means), which is exactly where the smoothed composite sits far below the first boundary and the labeller is least likely to move next. The estimated labels' own dynamics — distance to the boundary — were being attributed to the covariate.</p>
<p>The fix is to put the continuous state score into the tilt as a <b>baseline covariate</b> with its own coefficient γ, fitted first and held fixed, so that β measures what the covariate adds beyond what the labeller already knows about itself:</p>
<div class="eq">effective pressure_t  =  γ · z(composite_t)  +  β · z(covariate_t)          γ fitted on its own, then β given γ</div>
{figure(figs['wrongsign'], 6, "β for the true pressure in the energy world, by likelihood horizon k, with and without the composite baseline. Against true labels (left) every variant is positive, k = 5 strongest. Against estimated labels (right) the one-step fit without a baseline has the wrong sign; the baseline restores the sign at every k and the k = 5 fit has the largest out-of-sample gain. The numbers above the bars are the OOS gain per observation (nats) — the quantity the retention rule thresholds.")}
{table(fits[fits.labels == 'estimated labels'][['k', 'baseline', 'beta', 'gamma', 'oos_gain', 'folds_positive']], fmt_fits)}
<p>The same baseline sits in the outcome test of the retention rule, for the same reason. The general lesson, which applies to sub-states and tags as much as to leading features: <b>anything scored against a smoothed label must be given the continuous state first</b>, or it will be rewarded for rediscovering the label's own lag.</p>

<!-- 8 -->
<h2><span class="n">8</span>Out-of-sample gain by fold {BUILT}</h2>
<p>The retention test for the transition side: four time-ordered folds; on each training part estimate P⁰ (with the prior), γ and β; on the test part compute the k-step log-likelihood with β and with β = 0 (γ kept), and report the difference per observation. A covariate earns its place with a positive total and a positive gain in at least three folds. The table is for the energy world against estimated labels, k = 5, composite baseline.</p>
{table(gains[['feature', 'fold', 'train_end', 'beta', 'gamma', 'n_test', 'oos_gain_per_transition']], fmt_gain)}
<p class="dim">Gate 4 thresholds: gain ≥ 0.002 nats per observation in total and ≥ 3 folds positive, together with the outcome test. The true pressure clears both; the engineered gap feature does not clear the transition test against estimated labels on this history — a realistic outcome, and the reason real leading features are expected to fail more often than they pass.</p>

<!-- 9 -->
<h2><span class="n">9</span>Covariate paths: scenario, not forecast {BUILT}</h2>
<p>To project from today under a tilt you need ψ for every future day, and the honest answer is that you do not have it. So every covariate <b>declares a future path</b> and the fan is labelled with it: <i>frozen</i> at today's value (a sustained-pressure scenario), <i>decaying</i> to zero with a stated half-life (the default for the pressure index — energy dissipates), <i>calendar</i> (known in advance: event proximity is 1 on the event day and 0 otherwise), or <i>zero</i> (the constant matrix). A frozen path is not a forecast; it answers "if this persisted, what would the odds be".</p>
{figure(figs['paths'], 7, "Left: the four path policies for a covariate at +2 today. Right: P(stressed or extreme at h) from carry under each, with the β fitted on the energy world. The calendar path produces bumps at the known dates and nothing in between — which is exactly what an event covariate should do.")}

<!-- 10 -->
<h2><span class="n">10</span>Live · the matrix, the prior, the tilt and the paths {BUILT}</h2>
<div class="demo"><div class="hd">Live · edit the counts, set the prior, tilt and project</div>
<p class="dim" style="margin:0 0 8px">The cells are the transition <i>counts</i> from the synthetic labels; edit any count and the matrix re-estimates with the chosen prior (rows always renormalise). Pick a start, an effective pressure β·ψ and a path, and read the fan, the expected days in each state over the next 21, P(stressed or extreme) at 5 / 10 / 21 days and the stationary distribution.</p>
<div id="mat_edit"></div>
<div class="ctl">
  <div><label>sticky prior strength: <b id="prv">10</b></label><input type="range" id="pr" min="0" max="100" step="1" value="10"></div>
  <div><label>today's state</label><select id="s0"><option value="mix">mixture π</option>{''.join(f'<option value="{i}">{s}</option>' for i, s in enumerate(S6))}</select></div>
  <div><label>effective pressure β·ψ today: <b id="psv">0.00</b></label><input type="range" id="ps" min="-0.6" max="1.2" step="0.05" value="0"></div>
  <div><label>path</label><select id="pol"><option value="frozen">frozen</option><option value="decay" selected>decaying (half-life 10d)</option><option value="zero">zero (constant matrix)</option></select></div>
</div>
<div class="stat"><div>P(stressed or extreme) at 5d<b id="pc5">—</b></div><div>at 10d<b id="pc10">—</b></div><div>at 21d<b id="pc21">—</b></div><div>stationary<b id="stat" style="font-size:13px;font-weight:500">—</b></div></div>
<div class="stat"><div>expected days in each state, next 21<b id="occ" style="font-size:13px;font-weight:500">—</b></div><div>implied durations 1/(1−p<sub>ii</sub>)<b id="durs" style="font-size:13px;font-weight:500">—</b></div></div>
<canvas id="cv2" width="1600" height="400"></canvas>
<p class="dim" style="margin-top:6px">With β = {beta_e:+.2f} per standard deviation of the pressure index (the energy-world fit), a pressure of +2 is an effective β·ψ of about {2 * beta_e:+.2f} — small. The slider reaches further so the mechanics are visible; on real data the fitted β decides the scale.</p>
</div>

<!-- 11 -->
<h2><span class="n">11</span>Where the matrix goes next {HYP}</h2>
<p>Two uses are designed and not built. The <b>regime probability vector π</b> replaces today's point-mass label with a distribution: the label's own cell, with mass moved toward the escalation state by the shock score (the shock page's blend heuristic, a fraction of each state's mass proportional to the score) — the fan then starts from π. And the <b>path model</b> draws tomorrow's state from the row of the (possibly tilted) matrix, draws a surface move from historical days in that state, reprices the actual position and repeats; its first test is that, run with the constant matrix from historical entry dates, it reproduces the ladder within Monte Carlo error. Both wait on Gate 3.</p>

<!-- 12 -->
<h2><span class="n">12</span>On real data {BUILT} <span class="dim">WU-15, WU-17</span></h2>
<ol>
<li><b>Estimate on the full label history, per pair</b>, with the prior from config; write the counts next to the matrix (<code>transition_counts.csv</code>, <code>transition_matrix.csv</code>) so the prior's effect is visible row by row.</li>
<li><b>Run the duration and hazard checks before anything multi-step is quoted.</b> A state whose observed mean run is outside 0.6–1.5× the geometric one is flagged; its fan is shown with that caveat.</li>
<li><b>Cross-check against the ladder's persistence split</b> at h = 5, 10, 20 for each state (Figure 3, right). Disagreement is not an error in the ladder.</li>
<li><b>Fit β only with the baseline and only at k ≥ 5</b>; report the by-fold table, never the single number. A β that changes sign between k = 1 and k = 5 is the confound of Section 7 at work and is a reason to distrust the one-step fit, not the covariate.</li>
<li><b>Declare the path for every covariate in <code>configs/leading.yaml</code></b> (D19) and label every fan with it. The card shows P(stressed or extreme) at 5 / 10 / 21 days under the constant matrix and, if Gate 4 passed, under the tilt with the declared path.</li>
</ol>

<!-- 13 -->
<h2><span class="n">13</span>Code map</h2>
<table><tr><th>function</th><th>does</th></tr>
<tr><td><code>labels.transition_matrix(labels, names, prior_strength, stickiness)</code></td><td>counts + sticky prior → P</td></tr>
<tr><td><code>labels.durations · episodes</code></td><td>run lengths and counts per state</td></tr>
<tr><td><code>transitions.implied_vs_observed_duration(P, labels)</code></td><td>the geometric-vs-observed check</td></tr>
<tr><td><code>transitions.tilt(P, psi, beta)</code></td><td>the row-normalised logit tilt (ranks from <code>transitions.RANK</code>)</td></tr>
<tr><td><code>transitions.loglik(labels, psi, P, beta, k, base, gamma)</code></td><td>the k-step log-likelihood with the baseline covariate</td></tr>
<tr><td><code>transitions.fit_tilt(labels, psi, P, k, base)</code></td><td>γ then β by maximum likelihood</td></tr>
<tr><td><code>transitions.oos_gain(labels, psi, folds, k, base)</code></td><td>the by-fold out-of-sample gain; <code>.attrs</code> carry the total and folds positive</td></tr>
<tr><td><code>transitions.covariate_path · fan · expected_days · p_state_at</code></td><td>paths, the fan and its summaries</td></tr>
<tr><td><code>transitions.simulate_tilted_chain</code></td><td>ground truth for the tests and the synthetic energy world</td></tr>
<tr><td><code>python -m regime_ladder transitions --labels … [--pressure …]</code></td><td>the CLI; <code>inspect</code> writes the matrix, counts, durations, tilt fits and fans</td></tr>
</table>
"""

js = r"""
const S=D.states, COL=D.colours, K=S.length;
let CNT=D.counts.map(r=>r.slice());
function estimate(prior){return CNT.map((row,i)=>{const n=row.reduce((a,b)=>a+b,0);return row.map((c,j)=>(c+prior*(i===j?0.95:0.05/(K-1)))/(n+prior));});}
function renderMat(){let h='<table class="cm"><tr><th>from \\ to</th>'+S.map(s=>`<th style="color:${COL[s]}">${s}</th>`).join("")+'</tr>';
  CNT.forEach((row,i)=>{h+=`<tr><th style="text-align:left;color:${COL[S[i]]}">${S[i]}</th>`+row.map((v,j)=>`<td><input type="number" min="0" step="1" value="${v}" data-i="${i}" data-j="${j}" style="width:62px;font:inherit;font-size:12px;padding:2px;text-align:right;border:1px solid #ddd;border-radius:3px;background:${i===j?'#eef2ff':'#fff'}"></td>`).join("")+"</tr>";});
  document.getElementById('mat_edit').innerHTML=h+"</table>";document.querySelectorAll('#mat_edit input').forEach(inp=>inp.addEventListener('change',e=>{CNT[+e.target.dataset.i][+e.target.dataset.j]=Math.max(0,+e.target.value||0);run2();}));}
function stationary(P){let p=S.map(()=>1/K);for(let i=0;i<500;i++)p=mul(p,P);return p;}
function run2(){const prior=+pr.value;prv.textContent=prior;const P0=estimate(prior);const eff=+ps.value;psv.textContent=eff.toFixed(2);const pol=document.getElementById('pol').value;
  const s0=document.getElementById('s0').value;let p=(s0==="mix")?D.pi_mix.slice():S.map((_,i)=>i==+s0?1:0);
  const H=63,fan=[p.slice()];let occ=S.map(()=>0);
  for(let h=1;h<=H;h++){const e=pol==="frozen"?eff:(pol==="decay"?eff*Math.pow(0.5,h/10):0);const P=tilt(P0,D.ranks,e);p=mul(p,P);fan.push(p.slice());if(h<=21)occ=occ.map((o,i)=>o+p[i]);}
  const hi=h=>fan[h][3];pc5.textContent=(100*hi(5)).toFixed(0)+"%";pc10.textContent=(100*hi(10)).toFixed(0)+"%";pc21.textContent=(100*hi(21)).toFixed(0)+"%";
  document.getElementById('occ').textContent=S.map((s,i)=>s+" "+occ[i].toFixed(1)).join(" · ");
  const st=stationary(P0);document.getElementById('stat').textContent=S.map((s,i)=>s+" "+(100*st[i]).toFixed(0)+"%").join(" · ");
  document.getElementById('durs').textContent=S.map((s,i)=>s+" "+(1/(1-P0[i][i])).toFixed(1)+"d").join(" · ");
  const c=cv2.getContext('2d'),Wd=cv2.width,Hd=cv2.height,px=Wd/H,top=20,ph=Hd-60;c.clearRect(0,0,Wd,Hd);
  for(let h=0;h<H;h++){let y0=top;for(let i=0;i<K;i++){const hh=fan[h][i]*ph;c.fillStyle=COL[S[i]];c.fillRect(h*px,y0,px+0.6,hh);y0+=hh;}}
  c.fillStyle="#555";c.font="15px Inter,system-ui,sans-serif";for(let h=0;h<=H;h+=10)c.fillText(h+"d",h*px+2,Hd-14);
  c.fillStyle="rgba(255,255,255,.9)";c.fillRect(Wd-185,top+4,175,K*22+10);c.font="600 16px Inter,system-ui,sans-serif";let yl=top+22;S.forEach((s,i)=>{c.fillStyle=COL[s];c.fillRect(Wd-175,yl-12,13,13);c.fillStyle="#333";c.fillText(s,Wd-156,yl);yl+=22;});}
renderMat();['pr','ps','s0','pol'].forEach(id=>document.getElementById(id).addEventListener('input',run2));run2();
"""

toc = [("1", "What the matrix is for"), ("2", "Counts and the sticky prior"), ("3", "Durations and hazards"), ("4", "Fans and the stationary distribution"), ("5", "The tilt"), ("6", "Fitting β: k-step likelihood"),
       ("7", "The baseline covariate and the wrong sign"), ("8", "Out-of-sample gain by fold"), ("9", "Covariate paths"), ("10", "Live · matrix, prior, tilt, paths"), ("11", "Where the matrix goes next"), ("12", "On real data"), ("13", "Code map")]
out = page("transitions.html", "Transition probabilities",
           "The constant matrix and the time-varying one: how P is estimated from a hysteretic label history and regularised, whether it is a Markov chain at all, what its fans can and cannot say, the one valid way to let today's pressure tilt it, why the tilt has to be fitted on the k-step likelihood with the continuous state as a baseline, and how a future path for the covariate is declared rather than assumed. Every figure is from synthetic data with a known answer; the live panel runs the same tilt code in the browser.",
           toc, body, data, js)
print(out, beta_e)
