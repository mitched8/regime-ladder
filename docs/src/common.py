"""Shared toolkit for the deep-dive pages (docs/*.html).

Each page is one generator script next to this file. They share: the visual language of the
framework page, figure and data embedding (self-contained HTML, nothing external), a cached
synthetic world so all pages describe the same history, and a small JavaScript library (the
labeller, the tilt, canvas plotting) so the live demos are the same code on every page.

Regenerating on real data: point `world()` at a real market frame and trade-day table and every
figure and demo on every page is rebuilt from it.
"""
from __future__ import annotations

import base64, json, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
FIG = HERE / "_fig"; FIG.mkdir(exist_ok=True)
CACHE = HERE / "_cache"; CACHE.mkdir(exist_ok=True)
DOCS = HERE.parent

for f in font_manager.findSystemFonts(fontpaths=["/usr/share/fonts/opentype/inter"]):
    try: font_manager.fontManager.addfont(f)
    except Exception: pass
plt.rcParams.update({"font.family": "Inter", "font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "semibold",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 170})

import sys; sys.path.insert(0, str(ROOT))
from regime_ladder import report  # noqa: E402
COLOURS = report.COLOURS

# ----------------------------------------------------------------------------- the shared world
def world(force: bool = False) -> dict:
    """Ten years of synthetic market (seed 0), trades on the last five, features, composite, finder, labels."""
    p = CACHE / "world.pkl"
    if p.exists() and not force:
        return pickle.load(open(p, "rb"))
    from regime_ladder import synth, features, labels, schema
    m = synth.simulate_market(2520, seed=0); m5 = m.iloc[-1260:]
    td = schema.coerce(synth.simulate_trades(m5, seed=1))
    F = features.build(m, names=["vol_pct", "term_slope_pct", "rr_stress_pct"]); comp = labels.composite(F)
    st = m["state_true"]
    fwd5 = pd.Series([sum(synth.earn("straddle_atm", st.values[i + a], 21 - a) for a in range(1, 6)) if i + 6 < len(m) else np.nan for i in range(len(m))], index=m.index)
    lvl = m["atm_1m"].shift(-5)
    cal = labels.calibrate_states(comp, fwd5, level_target=lvl)
    cal_rv = labels.calibrate_states(comp, m["rv_1m"].shift(-21), level_target=lvl)
    cal_pnl_only = labels.calibrate_states(comp, fwd5)  # bounds on P&L: the thing that failed
    spec = cal["specs"]["6"]; est = labels.apply_spec(comp, spec)
    w = dict(m=m, m5=m5, td=td, F=F, comp=comp, st=st, fwd5=fwd5, cal=cal, cal_rv=cal_rv, cal_pnl_only=cal_pnl_only, spec=spec, est=est,
             lab_est=schema.labels_frame("EURUSD", est.index, est.values), lab_true=schema.labels_frame("EURUSD", m.index, st.values))
    pickle.dump(w, open(p, "wb"))
    return w


def energy_world(beta: float = 3.0, force: bool = False) -> dict:
    p = CACHE / f"energy_{beta}.pkl"
    if p.exists() and not force:
        return pickle.load(open(p, "rb"))
    from regime_ladder import synth, features, labels
    m = synth.simulate_market(2520, seed=0, energy_beta=beta)
    F = features.build(m, names=["vol_pct", "term_slope_pct", "rr_stress_pct"]); comp = labels.composite(F)
    st = m["state_true"]
    fwd5 = pd.Series([sum(synth.earn("straddle_atm", st.values[i + a], 21 - a) for a in range(1, 6)) if i + 6 < len(m) else np.nan for i in range(len(m))], index=m.index)
    cal = labels.calibrate_states(comp, fwd5, level_target=m["atm_1m"].shift(-5)); est = labels.apply_spec(comp, cal["specs"]["6"])
    w = dict(m=m, F=F, comp=comp, st=st, fwd5=fwd5, cal=cal, est=est)
    pickle.dump(w, open(p, "wb"))
    return w


# ----------------------------------------------------------------------------- html pieces
def b64(path) -> str:
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def save(fig, name: str) -> str:
    fig.tight_layout(); p = FIG / name; fig.savefig(p); plt.close(fig); return str(p)


def figure(path, n, caption, syn=True, maxw=None) -> str:
    style = f' style="max-width:{maxw}px;margin:0 auto"' if maxw else ""
    tag = '<span class="syn">synthetic</span>' if syn else ""
    return f'<figure><img src="{b64(path)}" alt=""{style}><figcaption><b>Figure {n}.</b>{tag} {caption}</figcaption></figure>'


def table(df: pd.DataFrame, fmt=None, index=False, cls="") -> str:
    fmt = fmt or {}
    cols = ([df.index.name or ""] if index else []) + list(df.columns)
    head = "".join(f"<th{' class=num' if pd.api.types.is_numeric_dtype(df[c]) else ''}>{c}</th>" if c in df.columns else f"<th>{c}</th>" for c in cols)
    rows = []
    for idx, r in df.iterrows():
        cells = [f"<td>{idx}</td>"] if index else []
        for c in df.columns:
            v = r[c]
            f = fmt.get(c)
            s = f(v) if f else (f"{v:+.3f}" if isinstance(v, float) else str(v))
            cells.append(f"<td{' class=num' if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool) else ''}>{s}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table class="{cls}"><tr>{head}</tr>{"".join(rows)}</table>'


BUILT = '<span class="st built">built · validated on synthetic</span>'
DESIGNED = '<span class="st designed">designed · not yet built</span>'
HYP = '<span class="st hyp">hypothesis</span>'

CSS = r"""
:root{--ink:#1a1d23;--muted:#5f6670;--rule:#d9dce1;--bg:#f6f7f9;--ind:#4F46E5;--carry:#3B6FD4;--rising:#2EAE7A;--agit:#D4A017;--crisis:#E8743B;--extreme:#B91C1C;--norm:#8B5CF6;--settle:#0EA5E9}
*{box-sizing:border-box} html,body{margin:0;background:#fff;color:var(--ink);font-family:Inter,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;font-size:15px;line-height:1.55}
main{max-width:900px;margin:0 auto;padding:40px 20px 80px}
h1{font-size:34px;line-height:1.1;letter-spacing:-.4px;margin:0 0 8px} h2{font-size:22px;margin:52px 0 10px;letter-spacing:-.2px;padding-top:14px;border-top:1px solid var(--rule)} h2 span.n{color:var(--ind);margin-right:8px} h3{font-size:16.5px;margin:22px 0 6px}
p{margin:0 0 12px} .kicker{font-size:12px;letter-spacing:1.6px;text-transform:uppercase;color:var(--ind);font-weight:600} .kicker a{color:inherit;text-decoration:none}
.sub{font-size:17px;color:var(--muted);max-width:740px}
.syn{display:inline-block;font-size:10px;letter-spacing:.6px;text-transform:uppercase;padding:2px 7px;border-radius:3px;background:#eef0f3;color:#4b5563;font-weight:600;vertical-align:middle;margin-left:6px}
.st{display:inline-block;font-size:10px;letter-spacing:.6px;text-transform:uppercase;padding:2px 7px;border-radius:3px;font-weight:600;vertical-align:middle;margin-left:8px}
.st.built{background:#e6f7ef;color:#166534} .st.designed{background:#ece9ff;color:#4c1d95} .st.hyp{background:#fdece2;color:#9a3412}
.box{border-left:3px solid var(--ind);background:var(--bg);padding:12px 16px;margin:14px 0 16px;border-radius:0 4px 4px 0} .box.warn{border-left-color:var(--crisis)} .box.ok{border-left-color:#2EAE7A} .box b{font-weight:600} .box p:last-child{margin:0}
pre{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;background:var(--bg);padding:12px 14px;border-radius:4px;overflow:auto;line-height:1.45} code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;background:var(--bg);padding:1px 4px;border-radius:3px}
.eq{font-family:ui-monospace,Menlo,monospace;font-size:13.5px;background:var(--bg);padding:10px 14px;border-radius:4px;margin:10px 0 14px;overflow:auto;white-space:pre}
figure{margin:14px 0 22px} figure img{width:100%;display:block;border:1px solid var(--rule);border-radius:4px} figcaption{font-size:13px;color:var(--muted);margin-top:6px} figcaption b{color:var(--ink)}
ol{padding-left:22px} li{margin-bottom:6px} ol li::marker{color:var(--ind);font-weight:600} ul{padding-left:20px}
table{width:100%;border-collapse:collapse;font-size:13.5px;margin:8px 0 14px} th{text-align:left;font-size:10.5px;letter-spacing:1px;text-transform:uppercase;color:var(--muted);padding:6px 8px;border-bottom:1px solid var(--ink);vertical-align:bottom} td{padding:6px 8px;border-bottom:1px solid var(--rule);vertical-align:top} td.num,th.num{text-align:right;font-variant-numeric:tabular-nums} .dim{color:var(--muted);font-size:12px}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin:12px 0 6px} .cell{border:1px solid var(--rule);border-radius:6px;padding:10px;font-size:13px;text-align:center} .cell b{display:block;font-size:14px} .cell small{color:var(--muted)} .cell select{font:inherit;font-size:13px;padding:3px;margin-top:4px;width:100%}
.ctl{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:18px;margin:10px 0} .ctl label{font-size:13px;color:var(--muted);display:block} .ctl input[type=range]{width:100%} .ctl select{font:inherit;padding:4px;width:100%} .ctl input[type=number]{font:inherit;padding:3px;width:70px}
.stat{display:flex;gap:28px;font-size:14px;margin:6px 0 10px;flex-wrap:wrap} .stat b{font-size:20px;display:block}
canvas{width:100%;height:auto;border:1px solid var(--rule);border-radius:4px;display:block}
.ok{color:#166534;font-weight:600} .no{color:#9a3412;font-weight:600}
.toc{columns:2;font-size:14px;margin:18px 0 6px} .toc div{break-inside:avoid;padding:3px 0} .toc span{color:var(--ind);font-weight:600;margin-right:8px}
.cm{border-collapse:collapse;font-size:12px;width:auto} .cm td,.cm th{padding:4px 8px;text-align:right;border:1px solid #eee} .cm th{font-size:10px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:18px} @media(max-width:700px){.two{grid-template-columns:1fr}}
.demo{border:1.5px solid var(--ind);border-radius:8px;padding:14px 18px;margin:16px 0 22px;background:#fff} .demo .hd{font-size:11px;letter-spacing:1px;text-transform:uppercase;color:var(--ind);font-weight:600;margin-bottom:6px}
svg text{font-family:Inter,system-ui,sans-serif}
footer{margin-top:50px;font-size:13px;color:var(--muted);border-top:1px solid var(--rule);padding-top:12px}
.nav{font-size:13px;color:var(--muted);margin:0 0 18px} .nav a{color:var(--ind);text-decoration:none;margin-right:12px}
@media(max-width:640px){h1{font-size:28px} .toc{columns:1} .grid{grid-template-columns:1fr}}
"""

PAGES = [("framework.html", "Framework"), ("states.html", "States"), ("transitions.html", "Transitions"), ("leading.html", "Leading features"),
         ("profiles.html", "Profiles & tags"), ("shock.html", "Shocks"), ("ladder.html", "Ladder")]


def nav(current: str) -> str:
    return '<div class="nav">' + " ".join(f'<a href="{f}">{t}</a>' if f != current else f'<b>{t}</b>' for f, t in PAGES) + '</div>'


def page(filename: str, title: str, subtitle: str, toc: list[tuple[str, str]], body: str, data: dict | None = None, js: str = "") -> str:
    toc_html = '<div class="toc">' + "".join(f"<div><span>{n}</span>{t}</div>" for n, t in toc) + "</div>"
    data_js = f"const D = {json.dumps(data, default=lambda v: float(v) if isinstance(v, (np.floating, np.integer)) else (None if v is None else str(v)))};\n" if data else ""
    html = f"""<!doctype html>
<html lang="en-GB"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Regime Ladder · {title}</title>
<style>{CSS}</style></head><body><main>
{nav(filename)}
<div class="kicker"><a href="framework.html">Regime Ladder · research scaffold</a> · deep dive</div>
<h1>{title}</h1>
<p class="sub">{subtitle}</p>
{toc_html}
{body}
<footer>Regime Ladder research scaffold · all figures synthetic until replaced on real data · the overview is <a href="framework.html">framework.html</a>; code and plan: the <code>regime-ladder</code> repository · generated by <code>docs/src/{Path(filename).stem and 'build_' + Path(filename).stem + '.py'}</code>.</footer>
</main>
<script>
{data_js}{JS_COMMON}
{js}
</script></body></html>"""
    out = DOCS / filename
    out.write_text(html)
    return str(out)


# ----------------------------------------------------------------------------- the shared JavaScript
JS_COMMON = r"""
// ---- the labeller, exactly as regime_ladder/labels.py
function ewma(x,h){const a=1-Math.exp(-Math.LN2/h);let s=null;return x.map(v=>{if(v==null)return null;s=(s==null)?v:a*v+(1-a)*s;return s;});}
function slope(sc,w){return sc.map((v,i)=>(v==null||i<w||sc[i-w]==null)?null:(v-sc[i-w])/w);}
function levelWay(x,bounds,d){let st=0;return x.map(v=>{if(v!=null){while(st<bounds.length&&v>bounds[st]+d)st++;while(st>0&&v<bounds[st-1]-d)st--;}return st;});}
function dirWay(x,lo,hi,d){let st=1;return x.map(v=>{if(v!=null){if(st==0&&v>lo+d)st=1;if(st==1){if(v>hi+d)st=2;else if(v<lo-d)st=0;}else if(st==2&&v<hi-d)st=1;}return st;});}
const LV=["low","mid","high","extreme"],DR=["down","flat","up"];
const PART6={"low,flat":"carry","low,up":"rising","low,down":{carry:"carry",_:"settling"},"mid,flat":"agitated","mid,up":{stressed:"stressed",normalising:"stressed",extreme:"stressed",_:"rising"},"mid,down":{stressed:"normalising",normalising:"normalising",extreme:"normalising",carry:"keep",_:"settling"},"high,up":"stressed","high,flat":"stressed","high,down":"normalising","extreme,up":"extreme","extreme,flat":"extreme","extreme,down":"extreme"};
const PART3={};["low","mid","high","extreme"].forEach(l=>DR.forEach(d=>PART3[l+","+d]={low:"carry",mid:"transition",high:"crisis",extreme:"crisis"}[l]));
function applyPartition(lv,dr,part,start){let prev=start||"carry";return lv.map((l,i)=>{let s=part[LV[l]+","+DR[dr[i]]];if(s===undefined)s="keep";if(typeof s==="object")s=s[prev]??s._;if(s!=="keep")prev=s;return prev;});}
function labels6(comp,hl,bounds,d,k,sd,part){const sc=ewma(comp,hl),ds=slope(sc,5),dup=k*sd;return {sc,ds,L:applyPartition(levelWay(sc,bounds,d),dirWay(ds,-dup,dup,0.2*dup),part||PART6)};}
function switches(L){let s=0;for(let i=1;i<L.length;i++)if(L[i]!==L[i-1])s++;return s;}
function episodes(L,states){const e={};(states||[...new Set(L)]).forEach(s=>e[s]=0);for(let i=0;i<L.length;i++){if(i==0||L[i]!==L[i-1])e[L[i]]=(e[L[i]]||0)+1;}return e;}
function agreement(L,truth,mask){let a=0,n=0;for(let i=0;i<L.length;i++){if(mask&&!mask[i])continue;n++;if(L[i]===truth[i])a++;}return a/n;}
function confusion(truth,L,states){const idx={};states.forEach((s,i)=>idx[s]=i);const C=states.map(()=>states.map(()=>0));for(let i=0;i<L.length;i++){if(idx[truth[i]]===undefined||idx[L[i]]===undefined)continue;C[idx[truth[i]]][idx[L[i]]]++;}return C;}
// out-of-sample R^2 of y on state dummies, time-ordered folds (labels.oos_separation)
function oosR2(L,y,folds){const ok=[];for(let i=0;i<L.length;i++)if(y[i]!=null&&L[i]!=null)ok.push(i);const n=ok.length;let sse=0,sst=0;const cuts=[];for(let f=0;f<=folds;f++)cuts.push(Math.round(0.4*n+(n-0.4*n)*f/folds));
  for(let f=0;f<folds;f++){const a=cuts[f],b=cuts[f+1];const sum={},cnt={};let tot=0;for(let j=0;j<a;j++){const i=ok[j];sum[L[i]]=(sum[L[i]]||0)+y[i];cnt[L[i]]=(cnt[L[i]]||0)+1;tot+=y[i];}const mAll=tot/a;
    for(let j=a;j<b;j++){const i=ok[j];const pred=cnt[L[i]]?sum[L[i]]/cnt[L[i]]:mAll;sse+=(y[i]-pred)**2;sst+=(y[i]-mAll)**2;}}
  return 1-sse/sst;}
// ---- the tilt, exactly as regime_ladder/transitions.py
function tilt(M,ranks,eff){return M.map((row,i)=>{const t=row.map((p,j)=>p*Math.exp(eff*(ranks[j]-ranks[i])));const z=t.reduce((a,b)=>a+b,0);return t.map(v=>v/z);});}
function mul(p,P){return P[0].map((_,j)=>p.reduce((a,pi,i)=>a+pi*P[i][j],0));}
function matpow(P,k){let R=P;for(let i=1;i<k;i++)R=R.map(row=>mul(row,P));return R;}
// ---- canvas helpers
function cmTable(C,states,colours){let h='<table class="cm"><tr><th></th>'+states.map(s=>`<th style="color:${colours[s]}">${s}</th>`).join("")+'<th>n</th></tr>';
  C.forEach((row,i)=>{const n=row.reduce((a,b)=>a+b,0)||1;h+=`<tr><th style="text-align:left;color:${colours[states[i]]}">${states[i]}</th>`+row.map((v,j)=>{const f=v/n;const bg=i===j?`rgba(46,174,122,${0.15+0.6*f})`:`rgba(232,116,59,${0.6*f})`;return `<td style="background:${bg}">${v}<br><span style="font-size:10px;color:#555">${(100*f).toFixed(0)}%</span></td>`;}).join("")+`<td>${n}</td></tr>`;});
  return h+"</table>";}
function lines(cv,series,opts){const c=cv.getContext('2d'),W=cv.width,H=cv.height,o=Object.assign({pad:[16,10,34,52],ymin:null,ymax:null,xlabels:null,zero:true},opts||{});const [pt,pr,pb,pl]=o.pad;c.clearRect(0,0,W,H);
  let ymin=o.ymin,ymax=o.ymax;if(ymin==null||ymax==null){let lo=Infinity,hi=-Infinity;series.forEach(s=>s.y.forEach(v=>{if(v==null)return;lo=Math.min(lo,v);hi=Math.max(hi,v);}));if(ymin==null)ymin=lo;if(ymax==null)ymax=hi;if(ymin===ymax){ymin-=1;ymax+=1;}}
  const n=Math.max(...series.map(s=>s.y.length)),X=i=>pl+(W-pl-pr)*i/(n-1),Y=v=>pt+(H-pt-pb)*(1-(v-ymin)/(ymax-ymin));
  c.strokeStyle="#e5e7eb";c.lineWidth=1;for(let t=0;t<=4;t++){const v=ymin+(ymax-ymin)*t/4,y=Y(v);c.beginPath();c.moveTo(pl,y);c.lineTo(W-pr,y);c.stroke();c.fillStyle="#666";c.font="13px Inter,system-ui";c.textAlign="right";c.fillText(v.toFixed(Math.abs(ymax-ymin)>20?0:2),pl-6,y+4);}
  if(o.zero&&ymin<0&&ymax>0){c.strokeStyle="#333";c.beginPath();c.moveTo(pl,Y(0));c.lineTo(W-pr,Y(0));c.stroke();}
  series.forEach(s=>{if(s.fill){c.fillStyle=s.fill;c.beginPath();let started=false;s.y.forEach((v,i)=>{if(v==null)return;started?c.lineTo(X(i),Y(v)):c.moveTo(X(i),Y(v));started=true;});for(let i=s.y.length-1;i>=0;i--){if(s.lo&&s.lo[i]!=null)c.lineTo(X(i),Y(s.lo[i]));}c.closePath();c.fill();}
    c.strokeStyle=s.color||"#222";c.lineWidth=s.lw||1.6;if(s.dash)c.setLineDash(s.dash);c.beginPath();let st=false;s.y.forEach((v,i)=>{if(v==null){st=false;return;}st?c.lineTo(X(i),Y(v)):c.moveTo(X(i),Y(v));st=true;});c.stroke();c.setLineDash([]);});
  c.fillStyle="#666";c.font="13px Inter,system-ui";c.textAlign="center";const xl=o.xlabels||[];xl.forEach(([i,t])=>c.fillText(t,X(i),H-pb+18));
  if(o.legend!==false){let x=pl+8;c.textAlign="left";c.font="600 13px Inter,system-ui";series.forEach(s=>{if(!s.name)return;c.fillStyle=s.color||"#222";c.fillRect(x,pt+2,12,4);c.fillStyle="#333";c.fillText(s.name,x+16,pt+8);x+=c.measureText(s.name).width+34;});}
  return {X,Y};}
function ribbon(cv,rows,colours,x0,x1){const c=cv.getContext('2d'),W=cv.width,H=cv.height,n=x1-x0,px=W/n,rh=H/rows.length;c.clearRect(0,0,W,H);rows.forEach((r,k)=>{for(let i=x0;i<x1;i++){c.fillStyle=colours[r.L[i]]||"#ddd";c.fillRect((i-x0)*px,k*rh+2,px+0.5,rh-4);}c.font="600 14px Inter,system-ui";const w=c.measureText(r.name).width+12;c.fillStyle="rgba(255,255,255,.9)";c.fillRect(6,k*rh+6,w,20);c.fillStyle="#333";c.fillText(r.name,12,k*rh+21);});}
"""
