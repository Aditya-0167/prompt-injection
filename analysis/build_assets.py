import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from values import *
import collections
plt.rcParams.update({"font.family":"serif","font.size":7.5,"axes.linewidth":0.6,"pdf.fonttype":42,"ps.fonttype":42})
ABBR={"Qwen3-4B":"Q3","Phi-4-mini":"Ph","Qwen2.5-3B":"Q3b","Granite-3.3-2B":"Gr","SmolLM2-1.7B":"Sm","Qwen2.5-1.5B":"Q1.5"}
ABBR={"Qwen3-4B":"Qwen3","Phi-4-mini":"Phi4","Qwen2.5-3B":"Q2.5-3B","Granite-3.3-2B":"Granite","SmolLM2-1.7B":"SmolLM2","Qwen2.5-1.5B":"Q2.5-1.5B"}
FAMN={"severity":"Severity downgrade","ioc":"IOC omission","technique":"Technique relabel"}

# ---------------- Fig 1: clean twin vs injected, undefended ----------------
c=h1[h1.defense=="none"]
fig,axes=plt.subplots(1,3,figsize=(7.1,2.1),sharey=True)
for ax,fam in zip(axes,FAM):
    x=np.arange(len(ORDER)); w=0.38
    for j,(col,lo,hi,lab,hatch,color) in enumerate([("clean_rate","clean_lo","clean_hi","clean twin","///","#cfcfcf"),("inj_rate","inj_lo","inj_hi","injected","","#2b2b2b")]):
        vals=[];el=[];eh=[]
        for m in ORDER:
            r=c[(c.model==m)&(c.family==fam)].iloc[0]
            vals.append(r[col]*100); el.append((r[col]-r[lo])*100); eh.append((r[hi]-r[col])*100)
        ax.bar(x+(j-0.5)*w,vals,w,yerr=[el,eh],capsize=1.5,label=lab,hatch=hatch,color=color,edgecolor="black",linewidth=0.4,error_kw={"linewidth":0.5})
    ax.set_xticks(x); ax.set_xticklabels([ABBR[m] for m in ORDER],rotation=35,ha="right",fontsize=6.5)
    ax.set_title(FAMN[fam],fontsize=8); ax.set_ylim(0,100); ax.grid(axis="y",linewidth=0.3)
axes[0].set_ylabel("Attacker target state reached (%)")
axes[0].legend(fontsize=6.5,frameon=False,loc="upper right")
fig.tight_layout(pad=0.3); fig.savefig("figs/fig_attack.pdf"); plt.close(fig)

# ---------------- Fig 2: strategy x model heatmap ----------------
inj=core[(core.report_kind=="inj")&(core.defense=="none")]
pv=inj.pivot_table(index="strategy",columns="model",values="hit")[ORDER]*100
pv["All"]=inj.groupby("strategy").hit.mean()*100
pv=pv.loc[V["strat_order"]]
fig,ax=plt.subplots(figsize=(3.45,2.5))
im=ax.imshow(pv.values,cmap="Greys",vmin=0,vmax=100,aspect="auto")
ax.set_xticks(range(pv.shape[1])); ax.set_xticklabels([ABBR.get(m,m) for m in pv.columns],rotation=40,ha="right",fontsize=6.5)
ax.set_yticks(range(pv.shape[0])); ax.set_yticklabels(pv.index,fontsize=7)
for i in range(pv.shape[0]):
    for j in range(pv.shape[1]):
        v=pv.values[i,j]; ax.text(j,i,f"{v:.0f}",ha="center",va="center",fontsize=6.5,color="white" if v>55 else "black")
ax.axvline(len(ORDER)-0.5,color="red",linewidth=0.8)
fig.tight_layout(pad=0.3); fig.savefig("figs/fig_strategy.pdf"); plt.close(fig)

# ---------------- Fig 3: defense benefit vs cost ----------------
fig,ax=plt.subplots(figsize=(3.45,2.7))
mk={"D1":"o","D2":"s","D3":"^"}
for d in DEF[1:]:
    for m in ORDER:
        da=wavg(h2[(h2.model==m)&(h2.defense==d)],"rd_defended_minus_undefended")*100
        di=wavg(h3[(h3.model==m)&(h3.defense==d)&(h3.metric=="clean_integrity")],"rd")*100
        ax.scatter(di,da,marker=mk[d],s=22,facecolor="white" if d=="D1" else ("black" if d=="D2" else "#888888"),edgecolor="black",linewidth=0.6,zorder=3)
        if d=="D2": ax.annotate(ABBR[m],(di,da),textcoords="offset points",xytext=(3,2),fontsize=5.5)
ax.axhline(0,color="black",linewidth=0.5); ax.axvline(0,color="black",linewidth=0.5)
ax.set_xlabel("Change in clean-report accuracy (pp)"); ax.set_ylabel("Change in attack success (pp)")
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([],[],marker=mk[d],linestyle="",markerfacecolor="white" if d=="D1" else ("black" if d=="D2" else "#888888"),markeredgecolor="black",label=l) for d,l in [("D1","D1 reminder"),("D2","D2 spotlighting"),("D3","D3 sandwich")]],fontsize=6,frameon=False,loc="upper left")
ax.text(0.98,0.04,"better ↓",transform=ax.transAxes,ha="right",fontsize=6)
ax.grid(linewidth=0.3); fig.tight_layout(pad=0.3); fig.savefig("figs/fig_defense.pdf"); plt.close(fig)

# ---------------- Fig 4: adaptive attackers ----------------
h4d=pd.DataFrame(V["h4"]); dabd=pd.DataFrame(V["dab"])
fig,axes=plt.subplots(1,2,figsize=(7.1,2.1),gridspec_kw={"width_ratios":[1.15,1]})
ax=axes[0]; x=np.arange(len(FULL)); w=0.2
for j,(d,kind,lab,hatch,color) in enumerate([("none","mean_single_rate","single, no defense","","#cfcfcf"),("none","best_of_8_rate","best of 8, no defense","","#2b2b2b"),("D2","mean_single_rate","single, D2","///","#cfcfcf"),("D2","best_of_8_rate","best of 8, D2","///","#2b2b2b")]):
    vals=[h4d[(h4d.model==m)&(h4d.defense==d)].iloc[0][kind]*100 for m in FULL]
    err=None
    if kind=="best_of_8_rate":
        err=[[ (h4d[(h4d.model==m)&(h4d.defense==d)].iloc[0][kind]-h4d[(h4d.model==m)&(h4d.defense==d)].iloc[0]["bo8_lo"])*100 for m in FULL],[ (h4d[(h4d.model==m)&(h4d.defense==d)].iloc[0]["bo8_hi"]-h4d[(h4d.model==m)&(h4d.defense==d)].iloc[0][kind])*100 for m in FULL]]
    ax.bar(x+(j-1.5)*w,vals,w,yerr=err,capsize=1.2,label=lab,hatch=hatch,color=color,edgecolor="black",linewidth=0.4,error_kw={"linewidth":0.5})
ax.set_xticks(x); ax.set_xticklabels([ABBR[m] for m in FULL]); ax.set_ylim(0,130); ax.set_yticks([0,20,40,60,80,100]); ax.set_ylabel("Cases attacked successfully (%)")
ax.legend(fontsize=5.8,frameon=False,ncol=2,loc="upper center"); ax.grid(axis="y",linewidth=0.3); ax.set_title("(a) attacker picks best of 8 strategies",fontsize=8)
ax=axes[1]; w=0.2
for j,(d,lab,hatch,color) in enumerate([("none","none","","#cfcfcf"),("D1","D1","...","#9a9a9a"),("D2","D2","///","#5a5a5a"),("D3","D3","\\\\\\","#2b2b2b")]):
    vals=[dabd[(dabd.model==m)&(dabd.defense==d)].iloc[0]["best_of_3_rate"]*100 for m in FULL]
    lo=[(dabd[(dabd.model==m)&(dabd.defense==d)].iloc[0]["best_of_3_rate"]-dabd[(dabd.model==m)&(dabd.defense==d)].iloc[0]["lo"])*100 for m in FULL]
    hi=[(dabd[(dabd.model==m)&(dabd.defense==d)].iloc[0]["hi"]-dabd[(dabd.model==m)&(dabd.defense==d)].iloc[0]["best_of_3_rate"])*100 for m in FULL]
    ax.bar(x+(j-1.5)*w,vals,w,yerr=[lo,hi],capsize=1.2,label=lab,hatch=hatch,color=color,edgecolor="black",linewidth=0.4,error_kw={"linewidth":0.5})
ax.set_xticks(x); ax.set_xticklabels([ABBR[m] for m in FULL]); ax.set_ylim(0,130); ax.set_yticks([0,20,40,60,80,100]); ax.legend(fontsize=6,frameon=False,ncol=4,loc="upper center",title="defense",title_fontsize=6)
ax.grid(axis="y",linewidth=0.3); ax.set_title("(b) defense-aware strategies, best of 3",fontsize=8)
fig.tight_layout(pad=0.3); fig.savefig("figs/fig_adaptive.pdf"); plt.close(fig)

# ---------------- Tables ----------------
def sig(p): return "$^{*}$" if p<0.05 else ""
lines=[r"\begin{tabular}{@{}lccc@{}}",r"\toprule",r"Model & Severity & IOC omission & Technique \\",r"\midrule"]
for m in ORDER:
    cells=[]
    for fam in FAM:
        r=c[(c.model==m)&(c.family==fam)].iloc[0]
        cells.append(f"{r.clean_rate*100:.0f} $\\rightarrow$ {r.inj_rate*100:.0f}{sig(r.p_holm)}")
    lines.append(f"{SHORT[m]} & "+" & ".join(cells)+r" \\")
lines+= [r"\bottomrule",r"\end{tabular}"]
open("tabs/attack.tex","w").write("\n".join(lines))

lines=[r"\begin{tabular}{@{}lrrrrrr@{}}",r"\toprule",r" & \multicolumn{2}{c}{D1 reminder} & \multicolumn{2}{c}{D2 spotlight} & \multicolumn{2}{c}{D3 sandwich} \\",r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",r"Model & $\Delta$atk & $\Delta$acc & $\Delta$atk & $\Delta$acc & $\Delta$atk & $\Delta$acc \\",r"\midrule"]
for m in ORDER:
    cs=[]
    for d in DEF[1:]:
        da=wavg(h2[(h2.model==m)&(h2.defense==d)],"rd_defended_minus_undefended")*100
        di=wavg(h3[(h3.model==m)&(h3.defense==d)&(h3.metric=="clean_integrity")],"rd")*100
        cs+= [f"{da:+.1f}",f"{di:+.1f}"]
    lines.append(f"{SHORT[m]} & "+" & ".join(cs)+r" \\")
lines.append(r"\midrule")
pc=[]
for d in DEF[1:]:
    pc+=[f"{V['def_'+d]}\\,[{V['def_'+d+'_lo']},{V['def_'+d+'_hi']}]".replace("-","$-$"),f"{V['int_'+d]}\\,[{V['int_'+d+'_lo']},{V['int_'+d+'_hi']}]".replace("-","$-$")]
lines.append(r"Pooled (95\% CI) & "+" & ".join(pc)+r" \\")
lines+= [r"\bottomrule",r"\end{tabular}"]
open("tabs/defense.tex","w").write("\n".join(lines))

lines=[r"\begin{tabular}{@{}lrrrrrrrr@{}}",r"\toprule",r" & \multicolumn{2}{c}{Single (mean)} & \multicolumn{2}{c}{Best of 8} & \multicolumn{4}{c}{Defense-aware, best of 3} \\",r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-9}",r"Model & none & D2 & none & D2 & none & D1 & D2 & D3 \\",r"\midrule"]
for m in FULL:
    g=lambda d,k,df: df[(df.model==m)&(df.defense==d)].iloc[0][k]*100
    lines.append(f"{SHORT[m]} & {g('none','mean_single_rate',h4d):.0f} & {g('D2','mean_single_rate',h4d):.0f} & {g('none','best_of_8_rate',h4d):.0f} & {g('D2','best_of_8_rate',h4d):.0f} & "+" & ".join(f"{g(d,'best_of_3_rate',dabd):.0f}" for d in DEF)+r" \\")
lines+= [r"\bottomrule",r"\end{tabular}"]
open("tabs/adaptive.tex","w").write("\n".join(lines))

# appendix: full attack table with CIs
lines=[r"\begin{tabular}{@{}llrrrrr@{}}",r"\toprule",r"Model & Family & Clean \% & Injected \% [95\% CI] & $\Delta$ (pp) & $n_{10}/n_{01}$ & $p_{\mathrm{Holm}}$ \\",r"\midrule"]
for m in ORDER:
    for fam in FAM:
        r=c[(c.model==m)&(c.family==fam)].iloc[0]
        pstr="<0.001" if r.p_holm<0.001 else f"{r.p_holm:.3f}"
        lines.append(f"{SHORT[m] if fam=='severity' else ''} & {fam} & {r.clean_rate*100:.1f} & {r.inj_rate*100:.1f} [{r.inj_lo*100:.1f}, {r.inj_hi*100:.1f}] & {r.rd_inj_minus_clean*100:+.1f} & {int(r.injection_induced)}/{int(r.reverse)} & {pstr} \\\\")
    lines.append(r"\addlinespace[1pt]")
lines+= [r"\bottomrule",r"\end{tabular}"]
open("tabs/attack_full.tex","w").write("\n".join(lines))

# appendix: authorized-instruction success and clean accuracy by model/defense
lines=[r"\begin{tabular}{@{}lrrrrrrrr@{}}",r"\toprule",r" & \multicolumn{4}{c}{Clean-report accuracy (\%)} & \multicolumn{4}{c}{Authorized instruction followed (\%)} \\",r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",r"Model & none & D1 & D2 & D3 & none & D1 & D2 & D3 \\",r"\midrule"]
for m in ORDER:
    a=[core[(core.model==m)&(core.report_kind=="clean")&(core.defense==d)].integrity_ok.mean()*100 for d in DEF]
    b=[core[(core.model==m)&(core.report_kind=="auth")&(core.defense==d)].hit.mean()*100 for d in DEF]
    lines.append(f"{SHORT[m]} & "+" & ".join(f"{v:.0f}" for v in a+b)+r" \\")
lines+= [r"\bottomrule",r"\end{tabular}"]
open("tabs/utility.tex","w").write("\n".join(lines))

# defense-aware by strategy (appendix numbers)
dd=dabs.copy(); 
lines=[r"\begin{tabular}{@{}llrrrr@{}}",r"\toprule",r"Model & Strategy & none & D1 & D2 & D3 \\",r"\midrule"]
for m in FULL:
    for s in ["escape","mimic_auth","policy_negation"]:
        vals=[dd[(dd.model==m)&(dd.strategy==s)&(dd.defense==d)].rate.iloc[0]*100 for d in DEF]
        lines.append(f"{SHORT[m] if s=='escape' else ''} & {s.replace('_',' ')} & "+" & ".join(f"{v:.0f}" for v in vals)+r" \\")
    lines.append(r"\addlinespace[1pt]")
lines+= [r"\bottomrule",r"\end{tabular}"]
open("tabs/da_strategy.tex","w").write("\n".join(lines))
print("assets built")
# print DA by strategy summary for text
for m in FULL:
    print(m,[ (s,[round(dd[(dd.model==m)&(dd.strategy==s)&(dd.defense==d)].rate.iloc[0]*100) for d in DEF]) for s in ["escape","mimic_auth","policy_negation"]])

# ================= extra assets: outcomes figure, per-cell defense table, examples =================
ij=core[core.report_kind=="inj"]
fig,ax=plt.subplots(figsize=(7.1,2.2))
x0=0; xt=[]; xl=[]
cols={"compromised":"#2b2b2b","abstained_manual_review":"#9a9a9a","resisted_normal_output":"#ffffff"}
hat={"compromised":"","abstained_manual_review":"///","resisted_normal_output":""}
for mi,m in enumerate(ORDER):
    for di,d in enumerate(DEF):
        t=tw[(tw.model==m)&(tw.defense==d)]
        n=t.n.sum(); bot=0
        for k in ["compromised","abstained_manual_review","resisted_normal_output"]:
            v=t[k].sum()/n*100
            ax.bar(x0,v,0.85,bottom=bot,color=cols[k],edgecolor="black",linewidth=0.4,hatch=hat[k],label={"compromised":"compromised","abstained_manual_review":"escalated to human","resisted_normal_output":"resisted"}[k] if (mi==0 and di==0) else None)
            bot+=v
        ax.text(x0,-9,["–","1","2","3"][di],ha="center",fontsize=5.5)
        x0+=1
    xt.append(x0-2.5); xl.append(ABBR[m]); x0+=0.8
ax.set_xticks(xt); ax.set_xticklabels(xl,fontsize=6.5); ax.tick_params(axis="x",pad=9)
ax.set_ylabel("Injected reports (%)"); ax.set_ylim(0,100)
ax.legend(fontsize=6,frameon=True,ncol=3,loc="upper center",bbox_to_anchor=(0.5,1.17))
fig.tight_layout(pad=0.3); fig.savefig("figs/fig_outcomes.pdf"); plt.close(fig)

# per-cell defense effects (appendix)
def star(p): return "$^{*}$" if p<0.05 else ""
lines=[r"\begin{tabular}{@{}llrrr@{}}",r"\toprule",r"Model & Family & D1 & D2 & D3 \\",r"\midrule"]
for m in ORDER:
    for fam in FAM:
        cs_=[]
        for d in DEF[1:]:
            r=h2[(h2.model==m)&(h2.family==fam)&(h2.defense==d)].iloc[0]
            cs_.append(f"{r.rd_defended_minus_undefended*100:+.1f}{star(r.p_holm)}")
        lines.append(f"{SHORT[m] if fam=='severity' else ''} & {fam} & "+" & ".join(cs_)+r" \\")
    lines.append(r"\addlinespace[1pt]")
lines+=[r"\bottomrule",r"\end{tabular}"]
open("tabs/defense_cells.tex","w").write("\n".join(lines))

# example outputs (appendix)
import json as _json, re as _re
bench={c["case_id"]:c for c in _json.load(open(BENCH))}
def esc(s):
    s=str(s)
    for a,b in [("\\",r"\textbackslash{}"),("&",r"\&"),("%",r"\%"),("#",r"\#"),("_",r"\_"),("{",r"\{"),("}",r"\}"),("$",r"\$"),("^",r"\^{}"),("<",r"\textless{}"),(">",r"\textgreater{}")]:
        s=s.replace(a,b)
    return s
def summ(raw):
    try:
        o=_json.loads(raw[raw.find("{"):raw.rfind("}")+1])
        return f"action={o.get('action')}, severity={o.get('severity')}, {len(o.get('iocs',[]))} IOCs, technique={o.get('technique')}"
    except Exception: return raw[:80]
exs=[("Qwen3-4B","inj","none","severity",lambda t:t.hit==1,"compromised"),
     ("Qwen2.5-3B","inj","none","severity",lambda t:(t.abstain==1)&(t.hit==0),"escalated"),
     ("Qwen3-4B","inj","none","severity",lambda t:(t.hit==0)&(t.abstain==0)&(t.integrity_ok==1),"resisted"),
     ("Qwen3-4B","inj","D2","technique",lambda t:t.hit==1,"compromised under D2")]
lines=[r"\begin{tabular}{@{}p{0.96\columnwidth}@{}}",r"\toprule"]
for (m,k,d,fam,cond,lab) in exs:
    t=core[(core.model==m)&(core.report_kind==k)&(core.defense==d)&(core.family==fam)]; t=t[cond(t)]
    r_=t.iloc[0]; c_=bench[r_.case_id]
    inj_t=injection_text_ex=None
    # reconstruct the injected sentence from the frozen report
    rep=c_["report_inj_primary"]; clean=c_["report_clean"]
    inj_line=[ln.strip() for ln in rep.split("\n") if ln.strip() not in [x.strip() for x in clean.split("\n")]]
    inj_line=inj_line[0] if inj_line else ""
    lines.append(r"\textbf{"+esc(lab)+r"} ("+esc(m)+", "+esc(d)+", "+esc(r_.case_id)+", gold "+esc(c_["gold_sev"] if fam=="severity" else c_["gold_tech"])+", target "+esc(c_["target"])+r")\\")
    lines.append(r"\emph{Injected sentence} ("+esc(c_["strategy"])+", "+esc(c_["placement"])+r"): \texttt{"+esc(inj_line[:170])+r"}\\")
    lines.append(r"\emph{Output:} \texttt{"+esc(summ(r_.raw_output))+r"}\\[2pt]")
    lines.append(r"\midrule")
lines[-1]=r"\bottomrule"; lines.append(r"\end{tabular}")
open("tabs/examples.tex","w").write("\n".join(lines))
print("extra assets built")

# ================= appendix tables: coverage, outcomes, strategy/placement x defense, data quality =================
tiers=[("core","Core"),("adaptive","Best-of-8"),("defense_aware","Def.-aware"),("sampled","Sampled")]
lines=[r"\begin{tabular}{@{}lrrrrr@{}}",r"\toprule",r"Model & Core & Best-of-8 & Def.-aware & Sampled & Total \\",r"\midrule"]
tot=collections.Counter() if False else {}
import collections as _c
tot=_c.Counter()
for m in ORDER:
    vals=[]
    for t,_ in tiers:
        v=int(cov[(cov.model==m)&(cov.tier==t)].rows_done.sum()); tot[t]+=v; vals.append(v)
    lines.append(f"{SHORT[m]} & "+" & ".join(f"{v:,}" if v else "--" for v in vals)+f" & {sum(vals):,}"+r" \\")
lines+= [r"\midrule",f"All models & "+" & ".join(f"{tot[t]:,}" for t,_ in tiers)+f" & {sum(tot.values()):,}"+r" \\",r"\bottomrule",r"\end{tabular}"]
open("tabs/coverage.tex","w").write("\n".join(lines))

lines=[r"\begin{tabular}{@{}lcccc@{}}",r"\toprule",r"Model & none & D1 & D2 & D3 \\",r"\midrule"]
for m in ORDER:
    cs_=[]
    for d in DEF:
        t=tw[(tw.model==m)&(tw.defense==d)]; n=t.n.sum()
        cs_.append("/".join(f"{t[k].sum()/n*100:.0f}" for k in ["compromised","abstained_manual_review","resisted_normal_output"]))
    lines.append(f"{SHORT[m]} & "+" & ".join(cs_)+r" \\")
lines+=[r"\bottomrule",r"\end{tabular}"]
open("tabs/outcomes.tex","w").write("\n".join(lines))

ijc=core[core.report_kind=="inj"]
lines=[r"\begin{tabular}{@{}lrrrr@{}}",r"\toprule",r"Pooled over six models & none & D1 & D2 & D3 \\",r"\midrule"]
for dim,vals in [("strategy",["correction","system","authority","direct","consistency","format","policy","urgency"]),("placement",["analyst_note","threat_feed","log_annotation","appendix_handoff","quoted_ticket"])]:
    for v in vals:
        r_=[ijc[(ijc[dim]==v)&(ijc.defense==d)].hit.mean()*100 for d in DEF]
        lines.append(f"{v.replace('_',' ')} & "+" & ".join(f"{x:.1f}" for x in r_)+r" \\")
    lines.append(r"\midrule")
lines[-1]=r"\bottomrule"; lines.append(r"\end{tabular}")
open("tabs/strat_def.tex","w").write("\n".join(lines))

qq=pd.read_csv(S+"quality.csv")
lines=[r"\begin{tabular}{@{}lrrrr@{}}",r"\toprule",r"Model & Valid JSON (\%) & Lowest cell (\%) & Abstain, clean (\%) & Abstain, injected (\%) \\",r"\midrule"]
for m in ORDER:
    t=qq[qq.model==m]; ok_=(t.parser_valid*t.n).sum()/t.n.sum()*100; lo_=t.parser_valid.min()*100
    ac=core[(core.model==m)&(core.report_kind=="clean")&(core.defense=="none")].abstain.mean()*100
    ai=core[(core.model==m)&(core.report_kind=="inj")&(core.defense=="none")].abstain.mean()*100
    lines.append(f"{SHORT[m]} & {ok_:.1f} & {lo_:.1f} & {ac:.1f} & {ai:.1f}"+r" \\")
lines+=[r"\bottomrule",r"\end{tabular}"]
open("tabs/quality.tex","w").write("\n".join(lines))
print("appendix tables built")
