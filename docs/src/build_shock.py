"""Deep dive 5: shock.html — Shock detection."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt, yaml
from common import world, figure, save, table, page, COLOURS, BUILT, HYP, ROOT
import sys; sys.path.insert(0, str(ROOT))
from regime_ladder import synth, labels, shock, transitions

W = world(); m, st, est = W["m"], W["st"], W["est"]
C = COLOURS; S = list(synth.STATES)
lcfg = yaml.safe_load(open(ROOT / "configs/leading.yaml"))["shock"]
x = np.arange(len(m)); seg = slice(1500, 2200)
figs = {}
def ribbon(ax, series, y, h, seg):
    cols = [C.get(v, "#999") for v in series.values[seg]]
    ax.bar(x[seg], [h] * len(cols), bottom=y, width=1, color=cols, lw=0)

# ---- F1 the surprise: distribution and the empirical null
z = shock.surprise(m); x2 = z ** 2
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.1))
axs[0].hist(z.dropna(), bins=60, density=True, color="#9aa0a6", alpha=0.8); xx = np.linspace(-4, 4, 200); axs[0].plot(xx, np.exp(-xx ** 2 / 2) / np.sqrt(2 * np.pi), color="#222", lw=1.2)
axs[0].set_title(f"surprise z: sd {z.std():.2f}, kurtosis {z.kurt() + 3:.1f}"); axs[0].set_xlabel("z"); axs[0].set_yticks([])
mu0 = x2.rolling(252, min_periods=60).mean()
axs[1].plot(x, mu0.values, color="#222", lw=1); axs[1].axhline(1, color="#888", ls="--", lw=0.8); axs[1].set_title("trailing 252-day mean of z²: the empirical null"); axs[1].set_xlabel("trading day"); axs[1].set_ylabel("mean z²")
byst = pd.concat([x2.rename("x2"), st], axis=1).dropna().groupby("state_true")["x2"].mean().reindex(S)
axs[2].bar(range(len(S)), byst.values, color=[C[s] for s in S]); axs[2].set_xticks(range(len(S))); axs[2].set_xticklabels(S, rotation=25, fontsize=7.5); axs[2].axhline(1, color="#888", ls="--", lw=0.8); axs[2].set_title("mean z² by true state"); axs[2].set_ylabel("mean z²")
figs["surprise"] = save(fig, "sh_surprise.png")

# ---- F2 the Figure-7 path: 3σ vs CUSUM (capped)
mf = shock.figure7_path(seed=0); zf = shock.surprise(mf); x2f = zf ** 2
h_c = shock.calibrate_cusum(x2f.iloc[1:200].values, k=0.3, arl0=1000); cus = shock.cusum(x2f, 0.3, h_c, null_window=120)
alarms = list(np.where(cus["alarm"].values)[0]); n = len(mf)
fig, axs = plt.subplots(2, 1, figsize=(10, 4.0), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})
axs[0].bar(range(n), np.abs(zf.fillna(0)), color=["#333" if abs(v) > 3 else "#bbb" for v in zf.fillna(0)], width=1); axs[0].axhline(3, color="#888", ls=":", lw=0.9); axs[0].axvline(250, color="#222", ls="--", lw=0.8); axs[0].set_ylabel("|z|"); axs[0].set_title("Three isolated 3σ days, then realised 50% above implied with no day above 3σ")
axs[1].plot(cus["S"].values, color="#4F46E5", lw=1.2); axs[1].axhline(h_c, color="#E8743B", ls="--", lw=0.9)
for a in alarms: axs[1].axvline(a, color="#E8743B", lw=1.4)
axs[1].axvline(250, color="#222", ls="--", lw=0.8); axs[1].set_ylabel("CUSUM S"); axs[1].set_xlabel("trading day")
first_alarm = next((a - 250 for a in alarms if a >= 250), None)
axs[1].set_title(f"CUSUM on capped z² (k 0.3, h {h_c:.1f} for ARL₀ 1000): {sum(a < 250 for a in alarms)} false alarms, detects {first_alarm} days after the change", fontsize=9)
figs["fig7"] = save(fig, "sh_fig7.png")
# ---- F2b what the cap buys: one of the isolated days at 4.5σ, uncapped vs capped, each with its own calibrated h
z45 = zf.copy(); z45.iloc[141] = 4.5 * np.sign(z45.iloc[141] or 1); x245 = z45 ** 2
res_cap = {}
for nm, cap in (("uncapped", 1e9), ("capped at 9", 9.0)):
    hh = shock.calibrate_cusum(x245.iloc[1:200].values, k=0.3, arl0=1000, clip=cap); cc = shock.cusum(x245, 0.3, hh, null_window=120, clip=cap); al = np.where(cc["alarm"].values)[0]
    res_cap[nm] = dict(h=hh, S=cc["S"].values, alarms=list(al), false=int((al < 250).sum()), first=next((a - 250 for a in al if a >= 250), None))
fig, axs = plt.subplots(2, 1, figsize=(10, 3.8), sharex=True)
for ax, nm in zip(axs, ("uncapped", "capped at 9")):
    r = res_cap[nm]; ax.plot(r["S"], color="#4F46E5" if nm != "uncapped" else "#9aa0a6", lw=1.2); ax.axhline(r["h"], color="#E8743B", ls="--", lw=0.9)
    for a in r["alarms"]: ax.axvline(a, color="#E8743B", lw=1.4)
    ax.axvline(250, color="#222", ls="--", lw=0.8); ax.set_ylabel(f"S, {nm}"); ax.set_title(f"{nm}: h calibrated on its own null = {r['h']:.1f} · {r['false']} alarm(s) before the change · detects {('after ' + str(r['first']) + ' days') if r['first'] is not None else 'never within the path'}", fontsize=9)
axs[1].set_xlabel("trading day")
figs["cap"] = save(fig, "sh_cap.png")

# ---- F3 ARL calibration curves and the delay/false-alarm trade-off
null_x2 = x2f.iloc[1:200].values
fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
hs = np.linspace(2, 30, 15)
rng = np.random.default_rng(0)
def arl_curve(k, clip=9.0, n_paths=150, max_len=4000):
    xn = np.minimum(null_x2[~np.isnan(null_x2)], clip); mu = xn.mean(); paths = rng.choice(xn, size=(n_paths, max_len)) - mu - k; out = []
    for h in hs:
        s_ = np.zeros(n_paths); hit = np.full(n_paths, max_len); alive = np.ones(n_paths, dtype=bool)
        for i in range(max_len):
            s_ = np.maximum(0, s_ + paths[:, i]); cross = alive & (s_ > h); hit[cross] = i + 1; alive &= ~cross
            if not alive.any(): break
        out.append(hit.mean())
    return np.array(out)
for k, col in ((0.1, "#9aa0a6"), (0.3, "#4F46E5"), (0.6, "#E8743B")):
    axs[0].plot(hs, arl_curve(k), "o-", ms=3, color=col, label=f"k = {k}")
axs[0].set_yscale("log"); axs[0].axhline(1000, color="#333", ls="--", lw=0.8); axs[0].text(2.5, 1150, "ARL₀ = 1000 (one false alarm in ~4 years)", fontsize=8); axs[0].set_xlabel("threshold h"); axs[0].set_ylabel("average run length under the null (days)"); axs[0].set_title("ARL calibration: h for a chosen false-alarm budget"); axs[0].legend(frameon=False, fontsize=8)
rows = []
for arl0 in (100, 250, 500, 1000, 2000, 4000):
    for k in (0.1, 0.3, 0.6):
        hh = shock.calibrate_cusum(null_x2, k=k, arl0=arl0); delays, falses = [], []
        for seed in range(6):
            mm_ = shock.figure7_path(seed=seed); zz = shock.surprise(mm_); c_ = shock.cusum(zz ** 2, k, hh, null_window=120); al = np.where(c_["alarm"].values)[0]
            falses.append(int((al < 250).sum())); d = [a - 250 for a in al if a >= 250]; delays.append(d[0] if d else 60)
        rows.append(dict(arl0=arl0, k=k, h=hh, false_alarms_per_250d=np.mean(falses), detection_delay=np.mean(delays)))
trade = pd.DataFrame(rows)
for k, col in ((0.1, "#9aa0a6"), (0.3, "#4F46E5"), (0.6, "#E8743B")):
    g = trade[trade.k == k]; axs[1].plot(g.false_alarms_per_250d, g.detection_delay, "o-", ms=3.5, color=col, label=f"k = {k}")
    for _, r in g.iterrows(): axs[1].annotate(str(int(r.arl0)), (r.false_alarms_per_250d, r.detection_delay), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
axs[1].set_xlabel("false alarms in the 250 calm days (mean of 6 paths)"); axs[1].set_ylabel("days to detect the change"); axs[1].set_title("The trade: delay against false alarms, by ARL₀ (labels)"); axs[1].legend(frameon=False, fontsize=8)
figs["arl"] = save(fig, "sh_arl.png")

# ---- F4 BOCD: run-length posterior heatmaps on the clean change and on the Figure-7 path
rng2 = np.random.default_rng(1); xc = pd.Series(np.concatenate([rng2.normal(0, 1, 300), rng2.normal(0, 3, 60)]))
def rl_posterior(xv, hazard=1 / 100, max_run=400, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0):
    from scipy.special import gammaln
    T = len(xv); R = np.zeros(max_run + 1); R[0] = 1.0; mu = np.array([mu0]); kap = np.array([kappa0]); al = np.array([alpha0]); be = np.array([beta0]); post = np.zeros((T, max_run + 1))
    for t in range(T):
        v = xv[t]
        if np.isnan(v): post[t] = post[t - 1] if t else post[t]; continue
        df_ = 2 * al; scale2 = be * (kap + 1) / (al * kap); tt2 = (v - mu) ** 2 / scale2
        pred = np.exp(gammaln((df_ + 1) / 2) - gammaln(df_ / 2) - 0.5 * np.log(df_ * np.pi * scale2) - (df_ + 1) / 2 * np.log1p(tt2 / df_))
        n_ = len(mu); growth = R[:n_] * pred * (1 - hazard); cp = float((R[:n_] * pred * hazard).sum()); newR = np.zeros(max_run + 1); newR[0] = cp; newR[1:n_ + 1] = growth[:max_run]; newR /= max(newR.sum(), 1e-300); R = newR
        mu_n = (kap * mu + v) / (kap + 1); be_n = be + kap * (v - mu) ** 2 / (2 * (kap + 1))
        mu = np.concatenate([[mu0], mu_n])[:max_run + 1]; kap = np.concatenate([[kappa0], kap + 1])[:max_run + 1]; al = np.concatenate([[alpha0], al + 0.5])[:max_run + 1]; be = np.concatenate([[beta0], be_n])[:max_run + 1]
        post[t] = R
    return post
fig, axs = plt.subplots(2, 2, figsize=(10.5, 5.2), gridspec_kw={"height_ratios": [1, 1.6]})
for j, (xv, title, cp) in enumerate([(xc.values, "a clean variance change (sd 1 → 3 at day 300)", 300), (zf.values, "the Figure-7 path (sd 0.8 → 1.35 at day 250)", 250)]):
    post = rl_posterior(xv); b_ = shock.bocd(pd.Series(xv), hazard=1 / 100, m=10)
    axs[0, j].plot(xv, color="#555", lw=0.8); axs[0, j].axvline(cp, color="#222", ls="--", lw=0.8); axs[0, j].set_title(title, fontsize=9); axs[0, j].set_ylabel("x")
    axs[1, j].imshow(np.log10(post[:, :200].T + 1e-6), aspect="auto", origin="lower", cmap="Greys", vmin=-4, vmax=0); axs[1, j].plot(b_["map_run"].values, color="#E8743B", lw=1)
    ax2 = axs[1, j].twinx(); ax2.plot(b_["p_run_short"].values, color="#4F46E5", lw=1.2); ax2.set_ylim(0, 1.05); ax2.set_ylabel("P(run < 10)", color="#4F46E5", fontsize=8)
    axs[1, j].set_ylim(0, 200); axs[1, j].set_ylabel("run length"); axs[1, j].set_xlabel("day"); axs[1, j].axvline(cp, color="#222", ls="--", lw=0.8); axs[1, j].grid(False)
fig.suptitle("Bayesian online changepoint detection: the run-length posterior (grey), its mode (orange) and P(run < 10) (blue)", fontsize=10, fontweight="semibold")
figs["bocd"] = save(fig, "sh_bocd.png")

# ---- F5 HAR forecast
fwd = m["rv_1m"].shift(-21)
har = shock.har_series(m); har0 = shock.har_series(m, ridge=0.0)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [1.6, 1]})
axs[0].plot(x[seg], fwd.values[seg], color="#222", lw=1, label="realised vol over the NEXT 21 days (the target)"); axs[0].plot(x[seg], har.values[seg], color="#4F46E5", lw=1.2, label="HAR forecast (ridge to persistence)"); axs[0].plot(x[seg], m["rv_1m"].values[seg], color="#9aa0a6", lw=0.8, label="naive: today's realised"); axs[0].plot(x[seg], m["atm_1m"].values[seg], color="#E8743B", lw=0.8, ls="--", label="implied (ATM 1M)")
axs[0].legend(frameon=False, fontsize=7.5, ncol=2); axs[0].set_ylabel("vol"); axs[0].set_xlabel("trading day"); axs[0].set_title("The physical forecast against what was realised and what was implied")
both = pd.concat([har.rename("har"), har0.rename("har_unshrunk"), m["rv_1m"].rename("naive"), fwd.rename("fwd")], axis=1).iloc[504:].dropna()
cors = both.corr()["fwd"].drop("fwd")
axs[1].bar(range(3), cors.values, color=["#4F46E5", "#E8743B", "#9aa0a6"]); axs[1].set_xticks(range(3)); axs[1].set_xticklabels(["HAR, ridge", "HAR, unshrunk", "naive"], fontsize=8); axs[1].set_ylabel("corr with forward realised"); axs[1].set_title("Out of sample, after the first two years")
for i, v in enumerate(cors.values): axs[1].text(i, v + 0.005, f"{v:.2f}", ha="center", fontsize=8)
figs["har"] = save(fig, "sh_har.png")

# ---- F6 shock score and lead profile; reliability by decile
ss = shock.shock_score(m, cfg=lcfg); lp_t = shock.lead_profile(ss["score"], st); lp_e = shock.lead_profile(ss["score"], est)
fig, axs = plt.subplots(2, 1, figsize=(10, 4.2), sharex=True, gridspec_kw={"height_ratios": [2, 0.5]})
axs[0].plot(x[seg], ss["score"].values[seg], color="#222", lw=1, label="score"); axs[0].plot(x[seg], 100 * ss["cusum_pressure"].values[seg], color="#4F46E5", lw=0.8, alpha=0.8, label="CUSUM pressure ×100"); axs[0].plot(x[seg], 100 * ss["bocd_p_short"].values[seg], color="#E8743B", lw=0.8, alpha=0.8, label="BOCD P(run<10) ×100"); axs[0].plot(x[seg], 100 * ss["recent_abs"].values[seg], color="#9aa0a6", lw=0.8, alpha=0.8, label="recent |z| / 95th pct ×100")
axs[0].legend(frameon=False, fontsize=7.5, ncol=4); axs[0].set_ylabel("0–100"); axs[0].set_title("The shock score and its three components")
ribbon(axs[1], st, 0, 1, seg); axs[1].set_yticks([]); axs[1].grid(False); axs[1].set_xlabel("trading day")
figs["score"] = save(fig, "sh_score.png")
# reliability: realised escalation within 5 days by score decile
rank = st.map(transitions.RANK); esc5 = pd.Series([(rank.iloc[i + 1:i + 6] > rank.iloc[i]).any() if i + 6 < len(rank) else np.nan for i in range(len(rank))], index=rank.index)
dec = pd.qcut(ss["score"], 10, labels=False, duplicates="drop")
rel = pd.concat([dec.rename("decile"), esc5.rename("esc")], axis=1).dropna().groupby("decile")["esc"].agg(["mean", "size"])
rank_e = est.map(transitions.RANK); esc5e = pd.Series([(rank_e.iloc[i + 1:i + 6] > rank_e.iloc[i]).any() if i + 6 < len(rank_e) else np.nan for i in range(len(rank_e))], index=rank_e.index)
rele = pd.concat([dec.rename("decile"), esc5e.rename("esc")], axis=1).dropna().groupby("decile")["esc"].mean()
fig, ax = plt.subplots(figsize=(7, 3))
ax.bar(rel.index - 0.2, rel["mean"], 0.4, color="#222", label="true states"); ax.bar(rele.index + 0.2, rele.values, 0.4, color="#4F46E5", label="estimated states")
ax.axhline(esc5.mean(), color="#222", ls=":", lw=0.9); ax.axhline(esc5e.mean(), color="#4F46E5", ls=":", lw=0.9); ax.set_xlabel("shock-score decile (today)"); ax.set_ylabel("P(a higher state within 5 days)"); ax.set_title("Does the score carry information at the ladder's horizon? Escalation frequency by score decile (dotted: unconditional)"); ax.legend(frameon=False, fontsize=8)
figs["rel"] = save(fig, "sh_rel.png")

# ---- F7 blend into pi: fan from a blended start vs a point mass
P = labels.transition_matrix(est, names=labels.state_order(est), prior_strength=10)
today = est.iloc[-1]
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.0))
for ax, score in zip(axs, (0, 50, 100)):
    pi = shock.blend_pi(pd.Series({today: 1.0}, index=P.index).fillna(0.0), score, w_max=lcfg["w_max"])
    f = transitions.fan(P, pi, psi_path=np.zeros(42)); ax.stackplot(f.index, f.T.values, colors=[C[c] for c in f.columns], alpha=0.9); ax.set_title(f"start from {today}, shock score {score}: π = " + ", ".join(f"{k} {v:.2f}" for k, v in pi[pi > 0.005].items()), fontsize=7.5); ax.set_xlabel("days ahead")
axs[0].set_ylabel("P(state at h)")
figs["blend"] = save(fig, "sh_blend.png")

# ---- live data
rng3 = np.random.default_rng(0); base_norm = rng3.normal(0, 1, 400)
htab = {f"{k}|{arl0}": float(shock.calibrate_cusum(null_x2, k=k, arl0=arl0)) for k in (0.1, 0.2, 0.3, 0.5, 0.75) for arl0 in (250, 500, 1000, 2000, 4000)}
data = {"norm": [round(float(v), 4) for v in base_norm], "htab": htab, "spikes": [60, 140, 210], "clean": [round(float(v), 4) for v in xc.values]}

fmt_lp = {"all": lambda v: f"{v:.1f}", "before_up": lambda v: f"{v:.1f}", "before_down": lambda v: f"{v:.1f}", "before_up_n": str, "before_down_n": str}
lp = pd.concat([lp_t.assign(labels="true"), lp_e.assign(labels="estimated")])[["labels", "all", "before_up", "before_up_n", "before_down", "before_down_n"]]
fmt_trade = {"arl0": str, "k": lambda v: f"{v:.1f}", "h": lambda v: f"{v:.1f}", "false_alarms_per_250d": lambda v: f"{v:.2f}", "detection_delay": lambda v: f"{v:.0f}"}

body = f"""
<!-- 1 -->
<h2><span class="n">1</span>Why a fast path {BUILT}</h2>
<p>The labeller is smooth and hysteretic by design, and therefore late: median detection lags of a week are what it is (states page, Figure 6). Nothing in the ladder depends on catching the first day of a move — the entry-conditional numbers are what they are — but two things do. The <b>regime probability vector π</b> that a path model starts from should not sit at 100% carry on the day the market has plainly started to break; and a trader reading the card on that day deserves to be told that the detector disagrees with the label. Shock detectors are that fast path. They do not define the state, they do not change the ladder, and they are not forecasts; they move probability mass toward the next state, under a rule labelled heuristic until it is calibrated.</p>
<div class="box"><b>Two questions, two quantities.</b> <i>Did the market move more than the surface said it would?</i> — the economic surprise, the quantity option P&amp;L is realised against. <i>Where is realised variance going?</i> — a physical forecast with its own innovation, for comparing a forward-RV path with the traded surface. Mixing them produces a detector that is neither.</div>

<!-- 2 -->
<h2><span class="n">2</span>The economic surprise {BUILT}</h2>
<div class="eq">z_t  =  log( spot_t / spot_(t−1) )  /  ( σ_impl,(t−1) / 100 / √252 )          σ_impl marked at the close BEFORE the move</div>
<p>The denominator is yesterday's implied daily vol, not today's: today's implied has already reacted to the move, and standardising by it would hide exactly what the detector is for. If the surface were an unbiased forecast of realised, z would have unit variance; it does not in general — carry markets deliver less than they price, stress markets more — so the detectors below never assume a null of E[z²] = 1 and instead estimate it from a trailing window.</p>
{figure(figs['surprise'], 1, f"Left: the distribution of surprises over ten synthetic years against a standard normal — fat-tailed (kurtosis {z.kurt() + 3:.1f}). Centre: the trailing mean of z², the empirical null, which drifts with the regime mix. Right: mean z² by true state — far below one in carry, above one in stressed. A detector calibrated to a null of one would alarm permanently in one regime and never in another.")}

<!-- 3 -->
<h2><span class="n">3</span>The single-day rule, and why it is the wrong detector {BUILT}</h2>
<p>The obvious detector — flag any day with |z| &gt; 3 — answers the wrong question. A sustained rise in realised relative to implied, which is what a regime change looks like from the surface, rarely produces a single day above 3σ; and isolated 3σ days happen in calm markets without anything changing. The test case below is the one the design was built on: a calm path with three isolated 3σ days, then sixty days of realised running 50% above implied with no day above 3σ.</p>

<!-- 4 -->
<h2><span class="n">4</span>CUSUM on squared surprises {BUILT}</h2>
<div class="eq">S_t  =  max( 0 ,  S_(t−1) + min(z_t², cap) − μ₀,t − k )          alarm when S_t &gt; h ; S resets to 0 after an alarm
μ₀,t  =  trailing 252-day mean of min(z², cap), lagged one day     — the empirical null
k     =  allowance (0.3): the drift S tolerates before it starts to climb; sets the shift size detected fastest
cap   =  3σ² = 9: a single day contributes at most 9 − μ₀ − k, so no isolated day can cross h on its own
h     =  threshold, set for a chosen false-alarm budget (Section 5)</div>
<p>Three design choices carry the behaviour. The <b>empirical null</b> makes the detector regime-aware without being told the regime. The <b>cap</b> is the division of labour between the jump channel and the shift detector: a single day contributes at most 9 − μ₀ − k, so from a standing start no isolated day can cross h, and — the part that matters more in practice — an extreme day inside the calibration window does not inflate h and blind the detector to the sustained shifts it exists for. The <b>reset</b> after an alarm means a second alarm is new evidence, not the same episode counted twice.</p>
{figure(figs['fig7'], 2, f"The test case. Top: |z|; the three isolated days are the only ones above 3σ and all are before the change. Bottom: the CUSUM on capped z² — no false alarm in the calm 250 days, detection {first_alarm} days after the change (14–20 across seeds). The threshold h = {h_c:.1f} comes from the ARL calibration on the first 200 days.")}
{figure(figs['cap'], 3, f"What the cap buys. The same path with one of the isolated days raised to 4.5σ (z² = 20), each variant with h calibrated on its own null. Uncapped, the one big day in the calibration window pushes h to {res_cap['uncapped']['h']:.1f} and the sustained change is never detected; capped, h is {res_cap['capped at 9']['h']:.1f} and the change is caught after {res_cap['capped at 9']['first']} days. Both versions alarm once on the 4.5σ day itself, because it arrives on top of accumulated S — that alarm is the jump channel's and is labelled as such on the card.")}

<!-- 5 -->
<h2><span class="n">5</span>Setting h: average run length {BUILT}</h2>
<p>h is not chosen by looking at past alarms. It is set so that under the <i>null</i> — squared surprises resampled from a calm window — the average number of days until a false alarm (the ARL₀) equals a budget stated in config: 1000 days, about one false alarm in four years. The calibration resamples the null a few hundred times, runs the CUSUM on each path, and bisects h until the mean first-crossing time matches. k and the budget are the two dials; the figure shows what each buys.</p>
{figure(figs['arl'], 4, "Left: ARL under the null against h for three allowances — the budget line picks h. Right: the trade-off that the budget sets, measured on six draws of the test case: false alarms in the 250 calm days against days to detect the change, for budgets from 100 to 4000. A budget of 250 (one a year) halves the delay and buys a false alarm a year; 1000 is the default because a false regime alarm on a trading floor is expensive and the labeller will confirm a real one within a week or two anyway.")}
{table(trade, fmt_trade)}

<!-- 6 -->
<h2><span class="n">6</span>Bayesian online changepoint detection {BUILT}</h2>
<p>BOCD (Adams–MacKay) keeps a full posterior over the <b>run length</b> — how many days since the last changepoint — under an explicit observation model: here a Normal with unknown mean and variance, Normal–Inverse-Gamma prior, so that both a level shift and a variance shift in the surprises count. Each day the posterior is updated with the predictive density of today's value under every run-length hypothesis and a constant hazard (1/100: one changepoint per ~100 days a priori). The reported quantity is <b>P(run length &lt; 10)</b>: the probability that the regime of the surprise process changed within the last two weeks.</p>
{figure(figs['bocd'], 5, "The run-length posterior through time on a clean variance change (left) and on the test case (right). On the clean change the posterior collapses to a short run within a week and P(run &lt; 10) spikes to 0.9. On the test case's modest change (sd 0.8 → 1.35) the posterior drifts over weeks and the mode moves to a secondary hypothesis that started at the third isolated day. BOCD is honest about weak evidence; that is its value next to the CUSUM, not a weakness to tune away.")}

<!-- 7 -->
<h2><span class="n">7</span>The physical forecast {BUILT}</h2>
<p>Separately from the detectors, a model of where realised vol is going: a HAR regression of realised vol over the next 21 days on today's one-week, one-month and three-month realised vols, refitted every 63 days on a 756-day window using only rows whose forward target is already known (refit dates are fixed positions from the start of the history, so the series is point-in-time). It is <b>ridge-shrunk toward persistence</b> — the prior that next month's realised equals this month's — with weight 0.5 × n.</p>
<div class="eq">RV_(t→t+21)  ≈  a + b₁ RV_1W,t + b₂ RV_1M,t + b₃ RV_3M,t           minimise Σ residual² + 0.5 n · || b − (0, 0, 1, 0) ||²</div>
{figure(figs['har'], 6, f"Left: the forecast against what was realised over the following month, today's realised (the naive forecast) and implied. Right: correlation with forward realised out of sample after the first two years — the ridge version ({cors['har']:.2f}) is no worse than persistence ({cors['naive']:.2f}); the unshrunk HAR ({cors['har_unshrunk']:.2f}) confidently fits a mean reversion the regime-switching sample cannot forecast. The forecast's job is the gap between it and implied: a physical forecast of variance risk premium, compared with the surface, not a detector.")}

<!-- 8 -->
<h2><span class="n">8</span>The shock score and the lead profile {BUILT}</h2>
<p>For the card, the three fast signals are folded into one 0–100 number: the mean of CUSUM pressure (S / h, capped at 1), BOCD's P(run &lt; 10), and the EWMA of |z| against its trailing 95th percentile. It is a display quantity, not a model input; the components are what the path model will use. The first honesty check on it is the <b>lead profile</b>: is the score higher on the days before an escalation than on an average day? A detector that is merely coincident with the label shows nothing here.</p>
{figure(figs['score'], 7, "The score and its components over 700 days with the true state beneath. The CUSUM term fires in sustained runs; the BOCD term in short bursts after clean breaks; the recent-|z| term carries the day-to-day level.")}
{table(lp, fmt_lp)}
<p class="dim">Mean score on the five days before each move to a higher-ranked state, before each move down, and over all days — against true and estimated labels. The lead over the unconditional mean is modest and in the right direction.</p>
{figure(figs['rel'], 8, "A sharper version of the same check at the ladder's horizon: the frequency of a higher state within the next five days by today's score decile, against true and estimated states, with the unconditional rate dotted. The top deciles carry two to three times the base rate — information the label does not have yet — and the relationship is monotone, which is what a blend weight can be calibrated to.", maxw=640)}

<!-- 9 -->
<h2><span class="n">9</span>The blend into π {HYP}</h2>
<p>How the fast path reaches the state machinery: the regime probability vector starts as a point mass on today's label and the blend moves a fraction w of each state's mass to its <b>escalation state</b> — carry and settling to rising, rising, agitated and normalising to stressed, stressed to extreme — with w proportional to the score:</p>
<div class="eq">w  =  w_max · score / 100          π(escalation(s)) += w · π(s) ,  π(s) −= w · π(s)          w_max = 0.5 (config)</div>
{figure(figs['blend'], 9, f"The fan from {today} under the constant matrix when the start is a point mass (score 0), a half blend (score 50) and a full blend (score 100). The blend changes the first weeks of the fan and nothing after: the matrix forgets the start either way.")}
<div class="box warn"><b>Status: heuristic.</b> A statistical changepoint in squared returns is not automatically a move between named states, and nothing above says that w_max = 0.5 is the right scale. The calibration test that would promote the blend is the reliability check of Figure 8 made formal: over a holdout, the blended π's probability of escalation within five days against the realised frequency, by score bin, with a slope near one. Until that is run on real data the blend is shown on the card as a flag ("detector disagrees with the label: score 72") and does not enter any number. The final threshold is economic, not statistical: the ladder's response to recent shock size — EV at h = 5 conditional on the score decile — and the kink in it, is what decides what counts as a shock.</div>

<!-- 10 -->
<h2><span class="n">10</span>Live · the detectors on a path you control {BUILT}</h2>
<div class="demo"><div class="hd">Live · CUSUM allowance, budget and cap; BOCD hazard; inject your own change</div>
<p class="dim" style="margin:0 0 8px">A 400-day path of surprises: calm at sd 0.8 with three isolated 3.4σ days, then a change you place and size. The 3σ rule, the CUSUM (with h from the precomputed ARL table for the chosen k and budget) and BOCD run in the browser on the same path. Read false alarms before the change and days to detect after it.</p>
<div class="ctl">
  <div><label>change on day <b id="c_day_v">250</b></label><input type="range" id="c_day" min="150" max="350" step="5" value="250"></div>
  <div><label>sd after the change ×<b id="c_mult_v">1.7</b></label><input type="range" id="c_mult" min="1" max="3" step="0.1" value="1.7"></div>
  <div><label>CUSUM allowance k</label><select id="c_k"><option>0.1</option><option>0.2</option><option selected>0.3</option><option>0.5</option><option>0.75</option></select></div>
  <div><label>false-alarm budget ARL₀</label><select id="c_arl"><option>250</option><option>500</option><option selected>1000</option><option>2000</option><option>4000</option></select></div>
  <div><label><input type="checkbox" id="c_cap" checked> cap z² at 9</label></div>
  <div><label>BOCD hazard 1/<b id="c_hz_v">100</b></label><input type="range" id="c_hz" min="20" max="400" step="20" value="100"></div>
</div>
<div class="stat"><div>3σ alarms before / after<b id="r_3s">—</b></div><div>CUSUM h<b id="r_h">—</b></div><div>CUSUM false alarms<b id="r_cf">—</b></div><div>CUSUM detects after<b id="r_cd">—</b></div><div>BOCD P(run&lt;10) peak in 20 days after<b id="r_bp">—</b></div></div>
<canvas id="cv_sh" width="1600" height="520"></canvas>
<p class="dim" style="margin-top:6px">Top: |z| with the 3σ line (dark bars above it). Middle: the CUSUM statistic with h (orange) and alarms (orange bars). Bottom: BOCD P(run &lt; 10). Dashed: the change.</p>
</div>

<!-- 11 -->
<h2><span class="n">11</span>On real data {BUILT} <span class="dim">WU-14</span></h2>
<ol>
<li><b>Mark the surprise off the previous close's implied</b>, at the same cut as the spot. A cut-time mismatch between the surface and spot shows up as fat tails in z that are not in the market.</li>
<li><b>Calibrate h on the first year, once, and keep it.</b> The null window for μ₀ then rolls; h does not. Re-calibrating h on recent data makes the detector chase its own alarms.</li>
<li><b>Write the alarm dates next to the label-switch dates.</b> That table is the deliverable of WU-14: how many days the detector leads the label, per switch, and how many alarms had no switch within a month.</li>
<li><b>Run the lead profile and the decile reliability</b> against the estimated labels. If the top deciles do not carry at least the base rate, the score is noise on this pair and the card should not show it.</li>
<li><b>Keep the blend a flag until the calibration test is run.</b> D-level decision: the owner decides when, and whether, the blended π enters the path model.</li>
<li><b>Failure modes.</b> z with mean far from zero (a drift in spot the implied cannot know about — expected in trending EM, check the window); μ₀ near the cap (the null is so stressed that the cap binds — raise the cap or lengthen the null window); BOCD firing on holidays or stale marks (filter non-trading days before, not after).</li>
</ol>

<!-- 12 -->
<h2><span class="n">12</span>Code map</h2>
<table><tr><th>function</th><th>does</th></tr>
<tr><td><code>shock.surprise(market, asof)</code></td><td>z, return over the implied daily vol marked before it</td></tr>
<tr><td><code>shock.three_sigma(z, k)</code></td><td>the rule to beat</td></tr>
<tr><td><code>shock.cusum(x2, k, h, null_window, reset, clip)</code></td><td>the statistic, alarms and the empirical null</td></tr>
<tr><td><code>shock.calibrate_cusum(x2_null, k, arl0, …)</code></td><td>h by resampling the null and bisecting on the ARL</td></tr>
<tr><td><code>shock.cusum_series(market, asof, …)</code></td><td>CUSUM pressure S/h as a point-in-time series (h calibrated once on the first year)</td></tr>
<tr><td><code>shock.bocd(x, hazard, …, m) · bocd_series</code></td><td>the run-length posterior; P(run &lt; m) and the MAP run</td></tr>
<tr><td><code>shock.har_fit · har_series(market, …, ridge)</code></td><td>the physical forecast</td></tr>
<tr><td><code>shock.shock_score(market, cfg) · lead_profile · blend_pi</code></td><td>the score, its first check, and the heuristic blend</td></tr>
<tr><td><code>shock.figure7_path(seed)</code></td><td>the test case</td></tr>
<tr><td><code>python -m regime_ladder shock --market … [--labels …]</code></td><td>the CLI; <code>inspect</code> writes <code>shock.csv</code> and <code>shock_lead_profile.csv</code></td></tr>
</table>
"""

js = r"""
// ---- lgamma (Lanczos) and the student-t log pdf for BOCD
function lgamma(z){const g=7,c=[0.99999999999980993,676.5203681218851,-1259.1392167224028,771.32342877765313,-176.61502916214059,12.507343278686905,-0.13857109526572012,9.9843695780195716e-6,1.5056327351493116e-7];if(z<0.5)return Math.log(Math.PI/Math.sin(Math.PI*z))-lgamma(1-z);z-=1;let a=c[0];const t=z+g+0.5;for(let i=1;i<9;i++)a+=c[i]/(z+i);return 0.5*Math.log(2*Math.PI)+(z+0.5)*Math.log(t)-t+Math.log(a);}
function bocd(x,hazard,m){const T=x.length,maxRun=T;let R=new Float64Array(maxRun+1);R[0]=1;let mu=[0],kap=[1],al=[1],be=[1];const pshort=new Array(T).fill(0);
  for(let t=0;t<T;t++){const v=x[t],n=mu.length,newR=new Float64Array(maxRun+1);let cp=0;for(let i=0;i<n;i++){const df=2*al[i],sc2=be[i]*(kap[i]+1)/(al[i]*kap[i]),tt2=(v-mu[i])**2/sc2;const lp=lgamma((df+1)/2)-lgamma(df/2)-0.5*Math.log(df*Math.PI*sc2)-(df+1)/2*Math.log1p(tt2/df);const pr=R[i]*Math.exp(lp);cp+=pr*hazard;if(i+1<=maxRun)newR[i+1]=pr*(1-hazard);}
    newR[0]=cp;let s=0;for(let i=0;i<=n;i++)s+=newR[i];for(let i=0;i<=n;i++)newR[i]/=(s||1e-300);R=newR;
    const mu2=[0],kap2=[1],al2=[1],be2=[1];for(let i=0;i<n;i++){mu2.push((kap[i]*mu[i]+v)/(kap[i]+1));be2.push(be[i]+kap[i]*(v-mu[i])**2/(2*(kap[i]+1)));kap2.push(kap[i]+1);al2.push(al[i]+0.5);}mu=mu2;kap=kap2;al=al2;be=be2;
    let p=0;for(let i=0;i<m&&i<=n;i++)p+=R[i];pshort[t]=p;}return pshort;}
function cusum(x2,k,h,nullWin,cap){const n=x2.length,S=new Array(n).fill(0),alarm=[];let s=0;for(let t=0;t<n;t++){const v=Math.min(x2[t],cap);if(t<40){S[t]=0;continue;}let mu=0,c=0;for(let j=Math.max(0,t-nullWin);j<t;j++){mu+=Math.min(x2[j],cap);c++;}mu/=c;s=Math.max(0,s+v-mu-k);if(s>h){alarm.push(t);s=0;}S[t]=s;}return {S,alarm};}
function runSh(){const day=+c_day.value,mult=+c_mult.value,k=+document.getElementById('c_k').value,arl=+document.getElementById('c_arl').value,cap=document.getElementById('c_cap').checked?9:1e9,hz=+c_hz.value;c_day_v.textContent=day;c_mult_v.textContent=mult.toFixed(1);c_hz_v.textContent=hz;
  const z=D.norm.map((v,i)=>{let s=i<day?0.8*v:Math.max(-2.9,Math.min(2.9,0.8*mult*v));if(D.spikes.includes(i))s=3.4*Math.sign(v||1);return s;});const x2=z.map(v=>v*v);
  const h=D.htab[k+"|"+arl];r_h.textContent=h.toFixed(1);
  const n3b=z.slice(0,day).filter(v=>Math.abs(v)>3).length,n3a=z.slice(day).filter(v=>Math.abs(v)>3).length;r_3s.textContent=n3b+" / "+n3a;
  const c=cusum(x2,k,h,120,cap);const cf=c.alarm.filter(a=>a<day).length,cd=c.alarm.find(a=>a>=day);r_cf.textContent=cf;r_cd.textContent=cd===undefined?"not within the path":(cd-day)+" days";
  const p=bocd(z,1/hz,10);const pk=Math.max(...p.slice(day,day+20));r_bp.textContent=pk.toFixed(2);
  const cv=document.getElementById('cv_sh'),ctx=cv.getContext('2d'),W=cv.width,H=cv.height;ctx.clearRect(0,0,W,H);const n=z.length,px=(W-60)/n,X=i=>50+i*px;
  const panel=(y0,hh,ymax,draw)=>{ctx.strokeStyle="#e5e7eb";ctx.beginPath();ctx.moveTo(50,y0+hh);ctx.lineTo(W-10,y0+hh);ctx.stroke();draw(v=>y0+hh-(v/ymax)*hh);ctx.strokeStyle="#222";ctx.setLineDash([6,4]);ctx.beginPath();ctx.moveTo(X(day),y0);ctx.lineTo(X(day),y0+hh);ctx.stroke();ctx.setLineDash([]);};
  panel(10,150,4,Y=>{for(let i=0;i<n;i++){ctx.fillStyle=Math.abs(z[i])>3?"#333":"#bbb";ctx.fillRect(X(i),Y(Math.min(4,Math.abs(z[i]))),Math.max(1,px-0.5),Y(0)-Y(Math.min(4,Math.abs(z[i]))));}ctx.strokeStyle="#888";ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(50,Y(3));ctx.lineTo(W-10,Y(3));ctx.stroke();ctx.setLineDash([]);ctx.fillStyle="#555";ctx.font="13px Inter,system-ui";ctx.fillText("|z|",6,Y(2));});
  const smax=Math.max(h*1.3,...c.S);panel(180,160,smax,Y=>{ctx.strokeStyle="#4F46E5";ctx.lineWidth=1.6;ctx.beginPath();c.S.forEach((v,i)=>i?ctx.lineTo(X(i),Y(v)):ctx.moveTo(X(i),Y(v)));ctx.stroke();ctx.strokeStyle="#E8743B";ctx.setLineDash([6,4]);ctx.beginPath();ctx.moveTo(50,Y(h));ctx.lineTo(W-10,Y(h));ctx.stroke();ctx.setLineDash([]);c.alarm.forEach(a=>{ctx.fillStyle="#E8743B";ctx.fillRect(X(a)-1,Y(smax),3,Y(0)-Y(smax));});ctx.fillStyle="#555";ctx.font="13px Inter,system-ui";ctx.fillText("CUSUM S",6,Y(smax/2));});
  panel(360,140,1,Y=>{ctx.strokeStyle="#E8743B";ctx.lineWidth=1.6;ctx.beginPath();p.forEach((v,i)=>i?ctx.lineTo(X(i),Y(v)):ctx.moveTo(X(i),Y(v)));ctx.stroke();ctx.fillStyle="#555";ctx.font="13px Inter,system-ui";ctx.fillText("BOCD",6,Y(0.5));});
  ctx.fillStyle="#666";ctx.font="13px Inter,system-ui";for(let d=0;d<=400;d+=50)ctx.fillText(d,X(d)-8,H-4);}
['c_day','c_mult','c_k','c_arl','c_cap','c_hz'].forEach(id=>document.getElementById(id).addEventListener('input',runSh));runSh();
"""

toc = [("1", "Why a fast path"), ("2", "The economic surprise"), ("3", "The single-day rule"), ("4", "CUSUM on squared surprises"), ("5", "Setting h: average run length"), ("6", "Bayesian online changepoint detection"),
       ("7", "The physical forecast"), ("8", "The shock score and the lead profile"), ("9", "The blend into π"), ("10", "Live · the detectors"), ("11", "On real data"), ("12", "Code map")]
out = page("shock.html", "Shock detection",
           "The fast path next to a deliberately slow labeller: the economic surprise and why it is marked off yesterday's implied, why a single-day rule is the wrong detector, the CUSUM with its empirical null, its cap and its false-alarm budget, Bayesian online changepoint detection and what it is honest about, the physical forecast and what it is for, the shock score and its first honesty checks, and the blend into the state probability vector that stays a heuristic until it is calibrated. The live panel runs all three detectors in the browser on a path you shape.",
           toc, body, data, js)
print(out, h_c, first_alarm)
