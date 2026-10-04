"""Deep dive 4: profiles.html — Profiles, sub-states and tags."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt, yaml
from common import world, figure, save, table, page, COLOURS, BUILT, HYP, ROOT
import sys; sys.path.insert(0, str(ROOT))
from regime_ladder import synth, labels, profile, discover, tags, ladder, schema

W = world(); m, st, est, td, lab_true = W["m"], W["st"], W["est"], W["td"], W["lab_true"]
C = COLOURS; S = list(synth.STATES); S6 = list(labels.STATES6)
pcfg = yaml.safe_load(open(ROOT / "configs/profile.yaml")); tcfg = yaml.safe_load(open(ROOT / "configs/default.yaml"))["tags"]
ch = profile.characteristics(m)
prof_t = profile.state_profile(ch, st.rename("regime")); prof_e = profile.state_profile(ch, est.rename("regime"))
x = np.arange(len(m)); seg = slice(1500, 2200)
figs = {}
PS = [s for s in S if s != "extreme"]

def ribbon(ax, series, y, h, seg):
    cols = [C.get(v, "#999") for v in series.values[seg]]
    ax.bar(x[seg], [h] * len(cols), bottom=y, width=1, color=cols, lw=0)

# ---- F1 the characteristics vector over time (a few)
fig, axs = plt.subplots(5, 1, figsize=(10, 6.4), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 1, 0.45]})
for ax, col, nm in zip(axs[:4], ["spotvol_corr", "vrp", "dollar_share", "corr_spot_eq"], ["spot-vol correlation (20d)", "VRP = implied − realised", "dollar-factor share of G10 variance", "corr(spot, equities) 20d"]):
    ax.plot(x[seg], ch[col].values[seg], color="#222", lw=1); ax.set_ylabel(nm, fontsize=7.5)
ribbon(axs[4], st, 0, 1, seg); axs[4].set_yticks([]); axs[4].grid(False); axs[4].set_xlabel("trading day")
axs[0].set_title("Four of the characteristics over 700 days, with the true state below")
figs["chars"] = save(fig, "pr_chars.png")

# ---- F2 profile heatmaps: true vs estimated labels
def heat(ax, prof, states, title):
    Dp = prof[[f"{s}_d" for s in states]].copy(); Dp.columns = states
    top = Dp.abs().max(axis=1).sort_values(ascending=False).index[:16]
    im = ax.imshow(Dp.loc[top].values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto"); ax.grid(False)
    ax.set_xticks(range(len(states))); ax.set_xticklabels(states, fontsize=7.5, rotation=20); ax.set_yticks(range(len(top))); ax.set_yticklabels(top, fontsize=7.5)
    for i in range(len(top)):
        for j in range(len(states)): ax.text(j, i, f"{Dp.loc[top[i], states[j]]:+.1f}", ha="center", va="center", fontsize=6.3, color="#111")
    ax.set_title(title, fontsize=9.5); return im
fig, axs = plt.subplots(1, 2, figsize=(11, 5.4))
heat(axs[0], prof_t, PS, "true labels"); im = heat(axs[1], prof_e, S6, "estimated labels")
plt.colorbar(im, ax=axs, fraction=0.02, pad=0.02, label="standardised difference vs all days (σ)")
fig.suptitle("State profiles: each characteristic's mean inside the state against all days, in standard deviations", fontsize=10, fontweight="semibold")
p = str(save(fig, "pr_heat.png")); figs["heat"] = p

# ---- F3 today's placement
place = profile.today_placement(ch, est); today = est.iloc[-1]
top3 = profile.distinguishing(prof_e, today, 4).index
fig, axs = plt.subplots(1, 4, figsize=(10.5, 2.6))
for ax, c in zip(axs, top3):
    vals = ch.loc[est[est == today].index, c].dropna(); ax.hist(vals, bins=30, color=C[today], alpha=0.6)
    ax.axvline(ch[c].iloc[-1], color="#222", lw=1.6); ax.set_title(f"{c}\ntoday at the {place.loc[c, 'pct_within_state']:.0%} pct of {today}", fontsize=8.5); ax.set_yticks([])
figs["place"] = save(fig, "pr_place.png")

# ---- discovery per state (true labels), stability by k
disc = {s: discover.discover(ch, st.rename("regime"), s, kmax=4) for s in PS}
fig, ax = plt.subplots(figsize=(9, 3))
for i, s in enumerate(PS):
    ks = sorted(disc[s]["stability"]); ax.plot(ks, [disc[s]["stability"][k] for k in ks], "o-", color=C[s], label=f"{s} (k={disc[s]['k']})", ms=4)
ax.axhline(0.7, color="#333", ls="--", lw=0.9); ax.text(3.6, 0.71, "stability threshold", fontsize=8); ax.set_xticks([2, 3, 4]); ax.set_xlabel("k clusters"); ax.set_ylabel("episode-split stability"); ax.set_title("Stability by k inside each true state: only rising and agitated clear the bar, both at k = 2"); ax.legend(frameon=False, fontsize=7.5, ncol=3)
figs["stab"] = save(fig, "pr_stab.png")

# ---- F4 discovery scatters (rising, agitated) with hidden types
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.6))
res = disc["rising"]; sub = res["labels"]; hid = m.loc[sub.index, "subtype_true"]
for ax, colour_by, title in [(axs[0], sub, "rising: discovered"), (axs[1], hid, "rising: hidden type")]:
    for v, col in zip(sorted(colour_by.unique()), ["#2EAE7A", "#E8743B"]):
        idx = colour_by.index[colour_by == v]; ax.scatter(ch.loc[idx, "corr_spot_eq"], ch.loc[idx, "dollar_share"], s=9, alpha=0.6, color=col, label=(res["names"].get(v, v) if colour_by is sub else v)[:40])
    ax.set_xlabel("corr(spot, equities)"); ax.set_ylabel("dollar-factor share"); ax.set_title(title, fontsize=9.5); ax.legend(frameon=False, fontsize=6.5)
ct = pd.crosstab(hid, sub).values; agree_r = max(np.trace(ct), np.trace(ct[:, ::-1])) / ct.sum()
res_a = disc["agitated"]; sub_a = res_a["labels"]; hid_a = m.loc[sub_a.index, "subtype_true"]
for v, col in zip(sorted(sub_a.unique()), ["#D4A017", "#4F46E5"]):
    idx = sub_a.index[sub_a == v]; axs[2].scatter(ch.loc[idx, "spotvol_corr"], ch.loc[idx, "volofvol"], s=9, alpha=0.6, color=col, label=res_a["names"].get(v, v)[:40])
cta = pd.crosstab(hid_a, sub_a).values; agree_a = max(np.trace(cta), np.trace(cta[:, ::-1])) / cta.sum() if cta.shape == (2, 2) else np.nan
axs[2].set_xlabel("spot-vol correlation"); axs[2].set_ylabel("vol-of-vol"); axs[2].set_title(f"agitated: discovered, {agree_a:.0%} agreement with the hidden sign", fontsize=9.5); axs[2].legend(frameon=False, fontsize=6.5)
fig.suptitle(f"Discovery recovers what was planted: rising k=2 ({agree_r:.0%} agreement with the hidden type), agitated k=2", fontsize=10, fontweight="semibold")
figs["disc"] = save(fig, "pr_disc.png")

# ---- F5 impurity detector: discovery inside ESTIMATED carry, composition by true state
res_c = discover.discover(ch, est, "carry", kmax=4)
comp_tab = None
if res_c["k"] > 1:
    comp_tab = pd.crosstab(res_c["labels"], st.loc[res_c["labels"].index]).reindex(columns=S, fill_value=0)
# also agitated estimated
res_ae = discover.discover(ch, est, "agitated", kmax=4)
imp_rows = []
for s_ in S6:
    r_ = discover.discover(ch, est, s_, kmax=4)
    if r_["k"] > 1:
        ctab = pd.crosstab(r_["labels"], st.loc[r_["labels"].index])
        for sub_name, row in ctab.iterrows():
            share = row / row.sum(); imp_rows.append(dict(estimated_state=s_, sub_state=sub_name, days=int(row.sum()), top_true_state=share.idxmax(), share=float(share.max()), name=r_["names"].get(sub_name, "")[:60]))
    else:
        imp_rows.append(dict(estimated_state=s_, sub_state="—", days=int((est == s_).sum()), top_true_state="", share=np.nan, name=f"k = 1 (best stability {max(r_['stability'].values()) if r_['stability'] else float('nan'):.2f})"))
impur = pd.DataFrame(imp_rows)

# ---- tags
T = tags.build(m, tcfg, "EURUSD"); tcs = T["corr_sign"]; tpin = T["pinned"]
split_lab, tinfo = tags.split_labels_frame(lab_true, tcs, min_episodes=5)
cells = pd.DataFrame([{**{"cell": k}, **v} for k, v in tinfo["EURUSD"].items()]).sort_values("cell")
corr21 = np.log(m["spot"]).diff().rolling(21).corr(m["atm_1m"].diff())
fig, axs = plt.subplots(3, 1, figsize=(10, 5.4), sharex=True, gridspec_kw={"height_ratios": [1.6, 0.6, 0.6]})
axs[0].plot(x[seg], corr21.values[seg], color="#222", lw=1); axs[0].axhspan(-0.2, 0.2, color="#9aa0a6", alpha=0.15); axs[0].axhline(0.2, color="#888", ls="--", lw=0.7); axs[0].axhline(-0.2, color="#888", ls="--", lw=0.7)
axs[0].set_ylabel("21d corr(Δspot, ΔATM)"); axs[0].set_title("The corr-sign tag: a dead band ±0.2 and hysteresis 0.05 on a 21-day correlation")
tcol = {"neg": "#1d4ed8", "pos": "#dc2626", "flat": "#d1d5db"}; hcol = {"corr_neg": "#1d4ed8", "corr_pos": "#dc2626"}
axs[1].bar(x[seg], 1, width=1, color=[tcol[v] for v in tcs.values[seg]], lw=0); axs[1].set_yticks([]); axs[1].grid(False); axs[1].set_title("tag: neg / flat / pos", fontsize=8, loc="left")
axs[2].bar(x[seg], 1, width=1, color=[hcol.get(v, "#f3f4f6") for v in m["subtype_true"].values[seg]], lw=0); axs[2].set_yticks([]); axs[2].grid(False); axs[2].set_title("hidden correlation regime inside agitated (grey = not agitated)", fontsize=8, loc="left"); axs[2].set_xlabel("trading day")
figs["tagcorr"] = save(fig, "pr_tagcorr.png")
# pinned + intervention illustration
rng = np.random.default_rng(2)
idx2 = pd.bdate_range("2024-01-01", periods=160)
spot_demo = np.concatenate([1 + 0.0004 * np.sin(np.arange(70)), (1 + 0.0004 * np.sin(69)) * np.exp(np.cumsum(rng.normal(0.0012, 0.006, 90)))])
md = pd.DataFrame({"spot": spot_demo, "atm_1m": 8.0}, index=idx2); rr_ = pd.Series(np.log(md["spot"]).diff(), index=idx2); md["rv_1m"] = (rr_.rolling(21).std() * np.sqrt(252) * 100).bfill()
tp = tags.pinned_tag(md); ti = tags.intervention_risk_tag(md, direction=1, move_window=21, move_threshold=0.05)
fig, axs = plt.subplots(2, 1, figsize=(10, 3.6), sharex=True, gridspec_kw={"height_ratios": [2, 0.8]})
axs[0].plot(range(160), md["spot"].values, color="#222", lw=1.2); axs[0].set_ylabel("spot"); axs[0].set_title("A constructed path: 70 pinned days, then a fast one-way move")
axs[1].bar(range(160), 1, width=1, color=["#0EA5E9" if v == "pinned" else "#f3f4f6" for v in tp.values], lw=0); axs[1].bar(range(160), 0.5, width=1, color=["#B91C1C" if v == "risk" else "none" for v in ti.values], lw=0)
axs[1].set_yticks([0.25, 0.75]); axs[1].set_yticklabels(["intervention risk", "pinned"], fontsize=7.5); axs[1].grid(False); axs[1].set_xlabel("day")
figs["tagpin"] = save(fig, "pr_tagpin.png")
# split ladder
cum_s = ladder.attach_entry_labels(ladder.cumulative(td, horizons=(1, 3, 5, 10, 20)), split_lab); lad_s = ladder.ladder(cum_s, split_lab, ci="hac")
fig, ax = plt.subplots(figsize=(6, 3.2))
for reg, col in [("agitated", C["agitated"]), ("agitated·neg", "#1d4ed8"), ("agitated·pos", "#dc2626")]:
    g = lad_s[(lad_s.archetype == "rr_25d") & (lad_s.regime == reg)].sort_values("h")
    if len(g): ax.fill_between(g.h, g.ci_lo, g.ci_hi, color=col, alpha=0.12, lw=0); ax.plot(g.h, g["mean"], "-o", color=col, ms=3.5, label=f"{reg} (ep {int(g.episodes.iloc[0])})")
ax.axhline(0, color="#333", lw=0.8); ax.set_xticks([1, 3, 5, 10, 20]); ax.set_xlabel("h"); ax.set_ylabel("rr_25d cumulative EV"); ax.set_title("The risk reversal in agitated, split by the corr-sign tag"); ax.legend(frameon=False, fontsize=7.5)
figs["split"] = save(fig, "pr_split.png")

# ---- live data A: per-state characteristics for the scatter demo (true labels)
cols_live = ["spotvol_corr", "spotvol_beta", "vrp", "term_slope", "rr_level", "fly_level", "volofvol", "jump_freq", "dollar_share", "avg_pair_corr", "corr_spot_eq", "corr_vol_eq", "corr_spot_rate2y", "corr_spot_dxy", "corr_vol_vix"]
live = {}
for s in PS:
    r_ = disc[s]; idx = r_["labels"].index
    live[s] = {"k": r_["k"], "stab": {str(k): round(float(v), 3) for k, v in r_["stability"].items()}, "names": r_["names"],
               "rows": [[round(float(ch.loc[d, c]), 4) if np.isfinite(ch.loc[d, c]) else None for c in cols_live] + [r_["labels"].loc[d], m.loc[d, "subtype_true"]] for d in idx]}
# ---- live data B: tag inputs
rng_ = (m["spot"].rolling(21).max() - m["spot"].rolling(21).min()) / m["spot"]; implied_range = 2 * m["atm_1m"] / 100 * np.sqrt(21 / 252)
tag_in = {"corr21": [round(float(v), 3) if np.isfinite(v) else None for v in corr21.values], "rv_iv": [round(float(v), 3) if np.isfinite(v) else None for v in (m["rv_1m"] / m["atm_1m"]).values],
          "range_ratio": [round(float(v), 3) if np.isfinite(v) else None for v in (rng_ / implied_range).values], "move21": [round(float(v), 4) if np.isfinite(v) else None for v in (m["spot"] / m["spot"].shift(21) - 1).values],
          "truth": list(st.values), "hidden": list(m["subtype_true"].values)}
data = {"states": PS, "colours": {s: C[s] for s in S}, "cols": cols_live, "live": live, "tag": tag_in}

fmt_cells = {"episodes": lambda v: f"{int(v)}", "days": lambda v: f"{int(v)}", "kept": lambda v: '<span class="ok">kept</span>' if v else '<span class="no">collapsed</span>'}
fmt_imp = {"days": lambda v: f"{int(v)}", "share": lambda v: f"{v:.0%}" if pd.notna(v) else ""}
desc_items = "".join(f"<li><b>{s}</b> — {profile.describe(prof_t, s).split(' · ', 1)[1]}</li>" for s in PS)

body = f"""
<!-- 1 -->
<h2><span class="n">1</span>Where a state is, and what it looks like {BUILT}</h2>
<p>A named state is a region of a two-dimensional score. That says <i>where</i> the market is on a stress scale and which way it is moving; it says nothing about <i>what kind</i> of rising this is — dollar-led or idiosyncratic, with equities or against them, with vol bid on rallies or on sell-offs. The trader's first question about a label is that one, and it decides which component the label is good or bad for. Three tools answer it, in increasing order of how much they are allowed to change the number on the card:</p>
<ol>
<li><b>The state profile</b> describes each state in terms of a daily <i>characteristics vector</i> — always shown, never conditions anything.</li>
<li><b>Sub-states</b> are clusters discovered inside a state on that vector without being told what to look for — descriptive until, separately, conditioning on them beats the parent out of sample.</li>
<li><b>Tags</b> are qualifiers the desk declares in advance (correlation sign, event window, pinned, intervention risk) — they split a state's ladder wherever the split has enough episodes.</li>
</ol>

<!-- 2 -->
<h2><span class="n">2</span>The characteristics vector {BUILT}</h2>
<p>Daily, trailing, point-in-time, in three groups. Every entry is a plain function of the market frame over a 20-day window (config), so it passes the same truncation test as a feature and can be computed for any day of the history.</p>
<table><tr><th>group</th><th>characteristic</th><th>definition (20-day window)</th></tr>
<tr><td rowspan="8">own pair</td><td><code>spotvol_beta</code>, <code>spotvol_corr</code></td><td>β and ρ of ΔATM on spot returns — the sign and strength of the spot-vol relationship</td></tr>
<tr><td><code>vrp</code></td><td>implied − realised (1M): how much the surface is paying over what is being delivered</td></tr>
<tr><td><code>term_slope</code></td><td>ATM 1M − ATM 1Y: inverted in stress, steep in carry</td></tr>
<tr><td><code>rr_level</code>, <code>fly_level</code></td><td>the 25d risk reversal and fly: skew and wings</td></tr>
<tr><td><code>volofvol</code></td><td>std of ΔATM: how much the surface itself is moving</td></tr>
<tr><td><code>jump_freq</code></td><td>share of days with |return| &gt; 2.5 trailing σ</td></tr>
<tr><td rowspan="2">cross-pair</td><td><code>dollar_share</code></td><td>share of the G10 panel's variance in its first principal component: is this move USD-led or idiosyncratic?</td></tr>
<tr><td><code>avg_pair_corr</code></td><td>mean pairwise correlation of the panel</td></tr>
<tr><td>cross-asset</td><td><code>corr_spot_*</code>, <code>corr_vol_*</code></td><td>correlation of the pair's spot return, and separately of its vol change, with each of: two equity indices, front- and long-end yields (diff), oil, gold, the dollar index, a credit spread, equity vol, rates vol</td></tr></table>
{figure(figs['chars'], 1, "Four of the characteristics over 700 days. Spot-vol correlation flips sign between agitated episodes (the hidden sub-type); VRP is thin in rising and fat in normalising; the dollar-factor share is high in the dollar-led kind of rising and low in the idiosyncratic kind. The list is a config entry; add a series and nothing else changes.")}

<!-- 3 -->
<h2><span class="n">3</span>The state profile {BUILT}</h2>
<p>For each characteristic and state: the mean inside the state against the mean over all days, in units of the all-days standard deviation — the <i>standardised difference</i> d. Ranked by |d|, the top three are what distinguishes the state; <code>describe</code> turns them into a phrase for the card.</p>
<div class="eq">d_(c,s)  =  ( mean of c on days in state s  −  mean of c on all days )  /  sd of c on all days</div>
{figure(figs['heat'], 2, "The profile from true labels (left) and from the labeller's (right). Stressed: steep skew, rich wings, inverted term structure, jumps. Carry: the reverse. Normalising keeps the wings and adds vol-of-vol. Agitated sits between and has the widest spread of spot-vol correlation of any state. The estimated-label profile is a blurred copy — each state's days include some of its neighbours' — but the ranking of what distinguishes each state survives.")}
<ul>{desc_items}</ul>
<p class="dim">The card's phrase for each true state, from the top three distinguishing characteristics.</p>
<h3>Today's placement</h3>
<p>The profile says what the state usually looks like; the card also needs to say whether <i>today</i> is a typical day of it. <code>today_placement</code> gives, for each characteristic, today's value and its percentile within the state's own history — a day at the 99th percentile of spot-equity correlation inside agitated is a different agitated from the median one, and the trader should see that before the number.</p>
{figure(figs['place'], 3, f"Today's placement inside {today}: the distribution of each of the four most distinguishing characteristics over the state's history, with today marked.")}

<!-- 4 -->
<h2><span class="n">4</span>Sub-state discovery {BUILT}</h2>
<p>Inside one named state, the days are clustered on the standardised characteristics vector with k-means (k-means++ seeding, six restarts, k = 2…4). A sub-state is allowed to exist only if three conditions hold at once:</p>
<ol>
<li><b>Preferred by BIC.</b> A spherical-cluster BIC proxy must prefer k &gt; 1 — a weak condition on its own; it almost always does.</li>
<li><b>Enough episodes.</b> Every cluster must appear in at least five contiguous runs. A cluster that is one long stretch of one year is a period, not a sub-state.</li>
<li><b>Stable across episodes.</b> The clustering fitted on a random half of the state's <i>episodes</i> and the clustering fitted on the other half must assign the same cluster to at least 70% of days, averaged over five random splits. Splitting by episode rather than by day matters: days inside one episode are near-duplicates of each other, so a day-level split would flatter any clustering; and a single first-half / second-half split (the first draft) is one noisy draw — on one seed it sat at 0.67 for a split that is real.</li>
</ol>
<p>Fail any one and k = 1: the named state stands alone. When a sub-state does exist it is named automatically from the two characteristics that most distinguish it from the rest of the state — "rising-2 · corr_spot_eq ↑ · corr_spot_vix ↓" — so that the card's phrase says what the cluster <i>is</i>, not that it is cluster 2.</p>
{figure(figs['stab'], 4, "Episode-split stability by k inside each true state. Rising and agitated clear the 0.7 bar at k = 2 and nowhere else; carry, settling, stressed and normalising never do. The information criterion would have split every one of them.")}
{figure(figs['disc'], 5, f"What discovery found. Inside rising (left two panels) the two clusters recover the hidden dollar-led / idiosyncratic type at {agree_r:.0%} agreement, separated on exactly the characteristics a trader would name — correlation with equities and the dollar-factor share. Inside agitated (right) it finds the hidden correlation-sign flip at {agree_a:.0%} agreement, without having been told that correlation sign was the thing to look for — which is the point: the tag of Section 6 was declared because the desk knows this flip exists; discovery is for the ones it does not.")}
<div class="box"><b>Sub-states and the number on the card.</b> A sub-state is descriptive — shown in the profile panel with its name and stability — until, separately, the ladder conditioned on it beats the parent state's ladder out of sample (benchmark layer 2b in the validation plan). Only then does it enter the number. The default is that it does not.</div>

<!-- 5 -->
<h2><span class="n">5</span>Discovery as a label-impurity detector {BUILT}</h2>
<p>Run inside the <i>estimated</i> states, discovery finds something else as well: the days the labeller got wrong. A cluster inside estimated carry whose characteristics look like settling is, in truth, mostly settling. The table runs discovery inside each estimated state and, where it splits, shows which true state each cluster is mostly made of.</p>
{table(impur, fmt_imp)}
<p>This is a diagnostic, not a correction: the labels are not relabelled by their clusters. But on real data, where there is no true state to compare with, a stable cluster inside a state whose profile reads like a neighbour is the clearest evidence available that the labeller is late or the boundary is misplaced, and it points to which one.</p>

<!-- 6 -->
<h2><span class="n">6</span>Tags {BUILT}</h2>
<p>Two agitated markets with the same score can be opposite trades for a skew position. Six states cannot carry that distinction and more states would starve the cells; discovery may or may not find it. A <b>tag</b> is a qualifier the desk declares in advance, computed point-in-time, that never changes the state but splits a state's ladder into <i>state·tag</i> cells wherever the split has enough evidence.</p>
<table><tr><th>tag</th><th>values</th><th>rule</th><th>config</th></tr>
<tr><td><code>corr_sign</code></td><td>neg · flat · pos</td><td>21-day correlation of spot returns with ΔATM, cut at ±band with hysteresis δ; starts flat</td><td>window 21, band 0.2, δ 0.05</td></tr>
<tr><td><code>event_window</code></td><td>pre · post · none</td><td>the N business days up to and including a scheduled event, and M after; events from a calendar the desk supplies — known in advance, so point-in-time by construction</td><td>before 3, after 1, events per pair</td></tr>
<tr><td><code>pinned</code></td><td>pinned · none</td><td>realised / implied &lt; r <b>and</b> the 21-day high-low range &lt; q × the range implied vol would suggest (2σ√(21/252)) — the market paying for movement it is not getting; long-gamma P&amp;L dies here first</td><td>rv_ratio 0.6, range_ratio 0.6</td></tr>
<tr><td><code>intervention_risk</code></td><td>risk · none</td><td>pair-specific: spot has moved more than m in the leaning-against direction over w days, or sits beyond a level, or an official-comment / rate-check flag from the adapter is set</td><td>per pair: direction, move 5% / 21d, level, flag column</td></tr></table>
{figure(figs['tagcorr'], 6, "The corr-sign tag against the hidden correlation regime that generated the agitated episodes. The dead band keeps weak correlations 'flat' (neutral, never splits); the hysteresis stops a correlation hovering near ±0.2 from flickering; the 21-day window is the lag. The tag labels the corr_neg days 'neg' and the corr_pos days 'pos' by better than two to one.")}
{figure(figs['tagpin'], 7, "Pinned and intervention-risk on a constructed path: seventy days of a market that barely moves while implied stays at 8 (pinned fires once realised has fallen far enough under implied and the range is narrow), then a fast one-way move that trips the intervention-risk threshold (+5% over 21 days, direction +1 — the BoJ-style configuration for a pair whose authority leans against a rising spot).")}
<h3>The split rule</h3>
<p>The regime series becomes <i>state·tag</i> where the tag is non-neutral, and stays the plain state where it is neutral (<i>flat</i>, <i>none</i>). Each tagged cell is then counted: a cell with fewer than five episodes collapses back to its parent state. So a tag adds cells only for the condition it names and only where the condition recurs; it can never multiply the cells until nothing is estimable.</p>
{table(cells[['cell', 'episodes', 'days', 'kept']], fmt_cells)}
<p class="dim">Cells from the corr-sign tag over the ten synthetic years with true labels. Both agitated cells have the episodes to stand.</p>
{figure(figs['split'], 8, "The risk reversal in agitated, split by the tag. In this world the flip moves the skew legs' earn by a fixed amount, so the difference is modest and widest at the short horizons; on real data it is the question the tag exists to ask, and the answer is allowed to be 'no difference' — the cells collapse and the card shows the parent.", maxw=620)}

<!-- 7 -->
<h2><span class="n">7</span>Tags, sub-states and leading features are three different things {BUILT}</h2>
<table><tr><th></th><th>tag</th><th>sub-state</th><th>leading feature</th></tr>
<tr><td>who defines it</td><td>the desk, in advance</td><td>the data, by clustering</td><td>the desk or the generic set, then the retention test</td></tr>
<tr><td>what it describes</td><td>today's market, a declared condition</td><td>today's market, a recurring pattern</td><td>what might happen next</td></tr>
<tr><td>how it is validated</td><td>episodes per cell (≥ 5)</td><td>BIC + episodes + episode-split stability; then layer 2b</td><td>OOS transition gain and OOS ΔR² beyond state + continuous score</td></tr>
<tr><td>what it changes</td><td>splits the ladder into cells</td><td>nothing, until 2b beats 2</td><td>the transition tilt and the fan; never the ladder</td></tr>
<tr><td>baseline it must beat</td><td>none — it is a conditioning choice</td><td>the parent state's ladder</td><td>the named state and the continuous score</td></tr></table>

<!-- 8 -->
<h2><span class="n">8</span>Live · inside a state {BUILT}</h2>
<div class="demo"><div class="hd">Live · pick a state and two characteristics</div>
<p class="dim" style="margin:0 0 8px">Each point is a day of the chosen (true) state. Colour by the discovered sub-state, by the hidden generating type (where there is one), or by nothing. The panel reports k, the stability by k and the auto-generated names from the Python run; the axes are yours to choose, which is the way to see <i>what</i> separates two sub-states.</p>
<div class="ctl">
  <div><label>state</label><select id="ps_state">{''.join(f'<option value="{s}"{" selected" if s == "rising" else ""}>{s}</option>' for s in PS)}</select></div>
  <div><label>x</label><select id="ps_x">{''.join(f'<option value="{i}"{" selected" if c == "corr_spot_eq" else ""}>{c}</option>' for i, c in enumerate(cols_live))}</select></div>
  <div><label>y</label><select id="ps_y">{''.join(f'<option value="{i}"{" selected" if c == "dollar_share" else ""}>{c}</option>' for i, c in enumerate(cols_live))}</select></div>
  <div><label>colour by</label><select id="ps_col"><option value="sub">discovered sub-state</option><option value="hid">hidden type</option><option value="none">nothing</option></select></div>
</div>
<div class="stat"><div>days<b id="ps_n">—</b></div><div>k found<b id="ps_k">—</b></div><div>stability by k<b id="ps_stab" style="font-size:13px;font-weight:500">—</b></div></div>
<div id="ps_names" class="dim" style="margin-bottom:6px"></div>
<canvas id="cv_ps" width="1600" height="620"></canvas>
</div>

<!-- 9 -->
<h2><span class="n">9</span>Live · tag thresholds and the cell table {BUILT}</h2>
<div class="demo"><div class="hd">Live · set the corr-sign band and hysteresis, the pinned ratios, the intervention threshold</div>
<p class="dim" style="margin:0 0 8px">Recomputes the three market tags on the synthetic history (true labels) and the episode count per state·tag cell. Cells with fewer than the minimum episodes collapse. Compare the corr-sign ribbon with the hidden regime beneath it.</p>
<div class="ctl">
  <div><label>corr-sign band ±<b id="t_band_v">0.20</b></label><input type="range" id="t_band" min="0" max="0.6" step="0.05" value="0.2"></div>
  <div><label>corr-sign hysteresis <b id="t_d_v">0.05</b></label><input type="range" id="t_d" min="0" max="0.2" step="0.01" value="0.05"></div>
  <div><label>pinned: realised/implied &lt; <b id="t_rv_v">0.60</b></label><input type="range" id="t_rv" min="0.3" max="1" step="0.05" value="0.6"></div>
  <div><label>pinned: range ratio &lt; <b id="t_rg_v">0.60</b></label><input type="range" id="t_rg" min="0.2" max="1.2" step="0.05" value="0.6"></div>
  <div><label>intervention: 21d move &gt; <b id="t_mv_v">5.0</b>%</label><input type="range" id="t_mv" min="1" max="10" step="0.5" value="5"></div>
  <div><label>minimum episodes per cell: <b id="t_min_v">5</b></label><input type="range" id="t_min" min="1" max="20" step="1" value="5"></div>
</div>
<canvas id="cv_tag" width="1600" height="150"></canvas>
<p class="dim" style="margin:4px 0 8px">Days 1500–2200: true state · corr-sign tag · hidden correlation regime · pinned · intervention risk.</p>
<div id="t_cells"></div>
</div>

<!-- 10 -->
<h2><span class="n">10</span>On real data {BUILT} <span class="dim">WU-06b, WU-08d, WU-11</span></h2>
<ol>
<li><b>Build the characteristics on the real frame first</b> (WU-06b) and look at the profile heatmap from the estimated labels before anything else: if stressed does not read as steep skew and inverted term structure, the labels are not describing the market the desk sees, whatever Gate 2 says.</li>
<li><b>Expect discovery to find little, and treat what it finds as a question.</b> On real data the candidates inside rising are the ones traders already name — dollar-led risk-off, rates-led, an idiosyncratic pair event, a carry unwind, commodity- or EM-led — and discovery should be made to show which of them are stable. Run it inside the estimated states as well, as the impurity detector of Section 5.</li>
<li><b>Declare tags before looking at ladders.</b> The thresholds in <code>configs/default.yaml › tags</code> and the per-pair intervention configs (D17) are set from desk knowledge, not tuned to outcomes; the cell table says which cells have the episodes to stand.</li>
<li><b>A sub-state or a tag enters the number only through layer 2b</b> — conditioning on it beats the parent out of sample at h ≤ 10 — and only with the continuous score in the baseline, for the same reason that applies to leading features: otherwise it is rewarded for finding the labeller's errors.</li>
<li><b>Failure modes.</b> A cross-asset series with a different cut time (correlations biased toward zero — fix the adapter); stability that is high for one k and one seed only (raise <code>n_splits</code>); a tag whose cells all collapse (the condition is rarer than thought, or the window is wrong); a tag that never reads neutral (the dead band is too narrow).</li>
</ol>

<!-- 11 -->
<h2><span class="n">11</span>Code map</h2>
<table><tr><th>function</th><th>does</th></tr>
<tr><td><code>profile.characteristics(market, cfg, asof)</code></td><td>the daily vector; <code>configs/profile.yaml</code> names the series</td></tr>
<tr><td><code>profile.state_profile · distinguishing · describe · today_placement</code></td><td>the profile, its ranking, the phrase, today's percentiles</td></tr>
<tr><td><code>discover.discover(chars, labels, state, kmax, min_episodes, stability, n_splits)</code></td><td>the three-condition discovery; returns k, labels, BIC, stability, names</td></tr>
<tr><td><code>discover.assign(chars_row, model)</code></td><td>today's sub-state under a fitted model</td></tr>
<tr><td><code>tags.corr_sign_tag · event_window_tag · pinned_tag · intervention_risk_tag</code></td><td>the four tags, each <code>f(market, asof)</code></td></tr>
<tr><td><code>tags.build(market, cfg, pair) · tag_split · split_labels_frame</code></td><td>all configured tags; the episode-gated split that feeds <code>ladder.ladder</code> unchanged</td></tr>
<tr><td><code>python -m regime_ladder inspect</code></td><td><code>characteristics.csv</code>, <code>state_profile.csv</code>, <code>today_placement.csv</code>, <code>discovery.json</code>, <code>substates_*.csv</code>, <code>tags.csv</code>, <code>tag_cells.json</code></td></tr>
</table>
"""

js = r"""
const COL=D.colours;
// ---- demo 8: scatter inside a state
const SUBCOL=["#2EAE7A","#E8743B","#4F46E5","#D4A017"],HIDCOL={usd_led:"#E8743B",idiosyncratic:"#2EAE7A",corr_neg:"#1d4ed8",corr_pos:"#dc2626",none:"#9aa0a6"};
function runPS(){const s=document.getElementById('ps_state').value,xi=+document.getElementById('ps_x').value,yi=+document.getElementById('ps_y').value,by=document.getElementById('ps_col').value;
  const L=D.live[s];ps_n.textContent=L.rows.length;ps_k.textContent=L.k;ps_stab.textContent=Object.entries(L.stab).map(([k,v])=>"k="+k+": "+v.toFixed(2)).join(" · ");
  ps_names.textContent=L.k>1?Object.values(L.names).join("   |   "):"no stable sub-state: the named state stands alone";
  const cv=document.getElementById('cv_ps'),c=cv.getContext('2d'),W=cv.width,H=cv.height;c.clearRect(0,0,W,H);
  const xs=L.rows.map(r=>r[xi]).filter(v=>v!=null),ys=L.rows.map(r=>r[yi]).filter(v=>v!=null);const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
  const pl=70,pb=50,pt=20,pr=20,X=v=>pl+(W-pl-pr)*(v-x0)/((x1-x0)||1),Y=v=>pt+(H-pt-pb)*(1-(v-y0)/((y1-y0)||1));
  c.strokeStyle="#e5e7eb";for(let t=0;t<=4;t++){const gx=pl+(W-pl-pr)*t/4,gy=pt+(H-pt-pb)*t/4;c.beginPath();c.moveTo(gx,pt);c.lineTo(gx,H-pb);c.stroke();c.beginPath();c.moveTo(pl,gy);c.lineTo(W-pr,gy);c.stroke();}
  c.fillStyle="#666";c.font="14px Inter,system-ui";c.textAlign="center";for(let t=0;t<=4;t++){c.fillText((x0+(x1-x0)*t/4).toFixed(2),pl+(W-pl-pr)*t/4,H-pb+20);}c.textAlign="right";for(let t=0;t<=4;t++){c.fillText((y1-(y1-y0)*t/4).toFixed(2),pl-8,pt+(H-pt-pb)*t/4+5);}
  c.textAlign="center";c.font="600 15px Inter,system-ui";c.fillStyle="#333";c.fillText(D.cols[xi],(pl+W-pr)/2,H-10);c.save();c.translate(16,(pt+H-pb)/2);c.rotate(-Math.PI/2);c.fillText(D.cols[yi],0,0);c.restore();
  const subs=[...new Set(L.rows.map(r=>r[D.cols.length]))].sort(),hids=[...new Set(L.rows.map(r=>r[D.cols.length+1]))].sort();
  L.rows.forEach(r=>{if(r[xi]==null||r[yi]==null)return;let col=COL[s];if(by==="sub")col=SUBCOL[subs.indexOf(r[D.cols.length])%4];if(by==="hid")col=HIDCOL[r[D.cols.length+1]]||"#999";c.globalAlpha=0.65;c.fillStyle=col;c.beginPath();c.arc(X(r[xi]),Y(r[yi]),4.2,0,2*Math.PI);c.fill();});c.globalAlpha=1;
  let lx=pl+10,ly=pt+16;c.font="600 14px Inter,system-ui";c.textAlign="left";const leg=by==="sub"?subs.map((v,i)=>[v,SUBCOL[i%4]]):(by==="hid"?hids.map(v=>[v,HIDCOL[v]||"#999"]):[]);leg.forEach(([t,col])=>{c.fillStyle=col;c.fillRect(lx,ly-10,12,12);c.fillStyle="#333";c.fillText(t,lx+16,ly);ly+=20;});}
['ps_state','ps_x','ps_y','ps_col'].forEach(id=>document.getElementById(id).addEventListener('input',runPS));runPS();
// ---- demo 9: tags
function orderedStart1(x,lo,hi,d){let st=1;const out=[];for(const v of x){if(v!=null){while(st<2&&v>[lo,hi][st]+d)st++;while(st>0&&v<[lo,hi][st-1]-d)st--;}out.push(["neg","flat","pos"][st]);}return out;}
function runs(seq,val){let n=0;for(let i=0;i<seq.length;i++)if(seq[i]===val&&(i===0||seq[i-1]!==val))n++;return n;}
function runT(){const band=+t_band.value,d=+t_d.value,rv=+t_rv.value,rg=+t_rg.value,mv=+t_mv.value/100,mn=+t_min.value;t_band_v.textContent=band.toFixed(2);t_d_v.textContent=d.toFixed(2);t_rv_v.textContent=rv.toFixed(2);t_rg_v.textContent=rg.toFixed(2);t_mv_v.textContent=(100*mv).toFixed(1);t_min_v.textContent=mn;
  const T=D.tag;const cs=orderedStart1(T.corr21,-band,band,d);const pin=T.rv_iv.map((v,i)=>(v!=null&&T.range_ratio[i]!=null&&v<rv&&T.range_ratio[i]<rg)?"pinned":"none");const iv=T.move21.map(v=>(v!=null&&v>mv)?"risk":"none");
  const a=1500,b=2200,cv=document.getElementById('cv_tag');const tcol={neg:"#1d4ed8",pos:"#dc2626",flat:"#d1d5db"},hcol={corr_neg:"#1d4ed8",corr_pos:"#dc2626"};
  ribbon(cv,[{name:"true state",L:T.truth},{name:"corr-sign tag",L:cs},{name:"hidden regime",L:T.hidden},{name:"pinned",L:pin},{name:"intervention risk",L:iv}],Object.assign({},COL,tcol,hcol,{none:"#f3f4f6",usd_led:"#f3f4f6",idiosyncratic:"#f3f4f6",pinned:"#0EA5E9",risk:"#B91C1C"}),a,b);
  let h='<table><tr><th>tag</th><th>cell</th><th class=num>episodes</th><th class=num>days</th><th>verdict</th></tr>';
  [["corr_sign",cs,["neg","pos"]],["pinned",pin,["pinned"]],["intervention_risk",iv,["risk"]]].forEach(([nm,seq,vals])=>{D.states.forEach(s=>{vals.forEach(v=>{const cell=T.truth.map((t,i)=>(t===s&&seq[i]===v)?s+"·"+v:t);const ep=runs(cell,s+"·"+v),days=cell.filter(c=>c===s+"·"+v).length;if(days===0)return;h+=`<tr><td>${nm}</td><td style="color:${COL[s]}">${s}·${v}</td><td class=num>${ep}</td><td class=num>${days}</td><td>${ep>=mn?'<span class="ok">kept</span>':'<span class="no">collapsed</span>'}</td></tr>`;});});});
  document.getElementById('t_cells').innerHTML=h+"</table>";}
['t_band','t_d','t_rv','t_rg','t_mv','t_min'].forEach(id=>document.getElementById(id).addEventListener('input',runT));runT();
"""

toc = [("1", "Where a state is vs what it looks like"), ("2", "The characteristics vector"), ("3", "The state profile and today's placement"), ("4", "Sub-state discovery"), ("5", "Discovery as an impurity detector"),
       ("6", "Tags"), ("7", "Tags vs sub-states vs leading features"), ("8", "Live · inside a state"), ("9", "Live · tag thresholds"), ("10", "On real data"), ("11", "Code map")]
out = page("profiles.html", "Profiles, sub-states and tags",
           "What kind of rising is this? The characteristics vector that describes a market day in the trader's terms, the profile that says what each state looks like and where today sits in it, the discovery procedure that finds stable sub-states inside a state without being told what to look for (and doubles as a detector of label error), and the tags the desk declares in advance — correlation sign, event window, pinned, intervention risk — that split a state's ladder wherever the split has the episodes to stand.",
           toc, body, data, js)
print(out, agree_r, agree_a, res_c["k"])
