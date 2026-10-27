import re, collections, zipfile, json
from values import *
def S(x):  # latex-safe number text
    return str(x).replace("-","$-$") if str(x).startswith("-") else str(x)
def f0(x): return f"{x*100:.0f}"
def f1(x): return f"{x*100:.1f}"
T=dict()
for k in ["n_cases","rows_total","rows_core","clean_overall_none","inj_overall_none","atk_none","atk_none_lo","atk_none_hi","def_D1","def_D1_lo","def_D1_hi","def_D3","def_D3_lo","def_D3_hi","clean_ioc_max","clean_ioc_max_model","h1_sig","rho","rho_p","h2_sig_better_D1","h2_sig_worse_D1","h2_sig_better_D2","int_q15_none","int_q15_D2","abst_clean_D2_q15","int_q3_none","int_q3_D2","abst_clean_D2_q3","tw_abs_none","tw_abs_D1","tw_comp_none","tw_res_none","tw_comp_D2","tw_abs_D2","tw_res_D2","parse_all","parse_min","regen_n","regen_text_diff"]:
    T[k]=V[k]
T["def_D2_abs"]=V["def_D2"].lstrip("-"); T["int_D2_abs"]=V["int_D2"].lstrip("-")
T["def_D2_lo_abs"]=V["def_D2_hi"].lstrip("-"); T["def_D2_hi_abs"]=V["def_D2_lo"].lstrip("-")
T["int_D2_lo_abs"]=V["int_D2_hi"].lstrip("-"); T["int_D2_hi_abs"]=V["int_D2_lo"].lstrip("-")
T["def_D2_ok"]=S(V["def_D2_ok"]); T["int_D2_ok_abs"]=V["int_D2_ok"].lstrip("-")
st=V["strat"]; T["strat_low"]=min(st.values(),key=float); T["strat_high"]=max(st.values(),key=float)
for k,v in st.items(): T["s_"+k]=v
for k,v in V["plc"].items(): T["p_"+k]=v
c=h1[h1.defense=="none"]
for fam,key in [("severity","sev_inj"),("technique","tech_inj"),("ioc","ioc_inj")]: T[key]=pp(wavg(c[c.family==fam],"inj_rate"))
cn=core[(core.report_kind=="clean")&(core.defense=="none")]
T["n_sev_clean"]=f"{(cn.family=='severity').sum():,}"; T["n_tech_clean"]=f"{(cn.family=='technique').sum():,}"
r=lambda m,f: c[(c.model==m)&(c.family==f)].iloc[0]
T["q3b_sev"]=f1(r("Qwen2.5-3B","severity").inj_rate); T["granite_ioc_clean"]=f1(r("Granite-3.3-2B","ioc").clean_rate); T["q15_ioc_inj"]=f1(r("Qwen2.5-1.5B","ioc").inj_rate)
for m in ORDER:
    T[f"m_{m}_inj"]=V["inj_none_model"][m]; T[f"ci_{m}"]=V["clean_int_none"][m]; T[f"a_{m}"]=V["auth_none"][m]
oth=[float(V["inj_none_model"][m]) for m in ORDER if m!="Qwen3-4B"]; T["others_min"]=f"{min(oth):.0f}"; T["others_max"]=f"{max(oth):.0f}"
inj=core[(core.report_kind=="inj")&(core.defense=="none")]
pv=inj.pivot_table(index="strategy",columns="model",values="hit")*100
T["q3_correction"]=f"{pv.loc['correction','Qwen3-4B']:.0f}"; T["q3_system"]=f"{pv.loc['system','Qwen3-4B']:.0f}"; T["q3_urgency"]=f"{pv.loc['urgency','Qwen3-4B']:.0f}"
ij=core[core.report_kind=="inj"]; cl=core[core.report_kind=="clean"]
T["abs_q3b_inj"]=pp(ij[(ij.model=="Qwen2.5-3B")&(ij.defense=="none")].abstain.mean())
T["abs_clean_none"]=pp(cl[cl.defense=="none"].abstain.mean()); T["abs_clean_D2"]=pp(cl[cl.defense=="D2"].abstain.mean())
h4d=pd.DataFrame(V["h4"]); dabd=pd.DataFrame(V["dab"])
g4=lambda m,d,k: h4d[(h4d.model==m)&(h4d.defense==d)].iloc[0][k]
gd=lambda m,d: dabd[(dabd.model==m)&(dabd.defense==d)].iloc[0]["best_of_3_rate"]
bo=[g4(m,"none","best_of_8_rate") for m in FULL]; T["bo8_min"]=f0(min(bo)); T["bo8_max"]=f0(max(bo))
oo=[g4(m,"none","best_of_8_rate") for m in FULL if m!="Qwen3-4B"]; T["bo8_other_min"]=f0(min(oo)); T["bo8_other_max"]=f0(max(oo))
sg=[g4(m,"none","mean_single_rate") for m in FULL]; T["single_min"]=f0(min(sg)); T["single_max"]=f0(max(sg))
T["bo8_Qwen3-4B_none"]=f1(g4("Qwen3-4B","none","best_of_8_rate")); T["bo8_Qwen3-4B_D2"]=f1(g4("Qwen3-4B","D2","best_of_8_rate"))
for m in ["Qwen3-4B","Phi-4-mini"]:
    for d in DEF: T[f"da_{m}_{d}"]=f0(gd(m,d))
sm=pd.DataFrame(V["sam"])
T["sam_Qwen3-4B_none_mean"]=f1(sm[(sm.model=="Qwen3-4B")&(sm.defense=="none")].iloc[0].mean_sampled_rate); T["sam_Qwen3-4B_none_greedy"]=f1(sm[(sm.model=="Qwen3-4B")&(sm.defense=="none")].iloc[0].greedy_rate_same_cases)
T["sam_gain_max"]=f"{((sm.any_of_3-sm.mean_sampled_rate)*100).max():.1f}"

# ---- extra tokens ----
cs=pd.read_json(BENCH)
L=cs.report_clean.str.len(); T["rep_mean"]=f"{L.mean():,.0f}"; T["rep_min"]=str(int(L.min())); T["rep_max"]=str(int(L.max()))
ioc=c[c.family=="ioc"]; T["ioc_cf"]=pp(wavg(ioc,"rd_inj_minus_clean")); T["ioc_clean_pool"]=pp(wavg(ioc,"clean_rate"))
T["ioc_overstate"]=f"{wavg(ioc,'inj_rate')/wavg(ioc,'rd_inj_minus_clean'):.1f}$\\times$"
T["q15_ioc_cf"]=S(f"{r('Qwen2.5-1.5B','ioc').rd_inj_minus_clean*100:+.1f}"); T["granite_ioc_inj"]=f1(r("Granite-3.3-2B","ioc").inj_rate); T["granite_ioc_cf"]=S(f"{r('Granite-3.3-2B','ioc').rd_inj_minus_clean*100:+.1f}")
a_=ij[ij.defense=="none"].groupby("model").abstain.mean(); b_=cl[cl.defense=="none"].groupby("model").abstain.mean(); esc=((a_-b_)*100)
T["esc_q3b"]=f"{esc['Qwen2.5-3B']:.1f}"; T["esc_q3"]=f"{esc['Qwen3-4B']:.1f}"; T["esc_other_max"]=f"{esc.drop(['Qwen2.5-3B','Qwen3-4B']).max():.1f}"
sf=(inj.pivot_table(index="strategy",columns="family",values="hit")*100)
for st_ in sf.index:
    for fam in sf.columns: T[f"sf_{st_}_{fam}"]=f"{sf.loc[st_,fam]:.0f}"
T["sf_weak_max"]=f"{sf.loc[['urgency','policy']].values.max():.0f}"
def tri(m,d):
    t=tw[(tw.model==m)&(tw.defense==d)]; n=t.n.sum()
    return "/".join(f"{t[k].sum()/n*100:.0f}" for k in ["compromised","abstained_manual_review","resisted_normal_output"])
for key,m in [("q15","Qwen2.5-1.5B"),("phi","Phi-4-mini"),("q3","Qwen3-4B")]:
    for d in ["none","D2"]: T[f"o_{key}_{d}"]=tri(m,d)


sv=ij[(ij.defense=="none")&(ij.family=="severity")]; hh=sv[sv.hit==1]
T["sev_hits"]=str(len(hh)); T["sev_cases"]=f"{len(sv):,}"; T["sev_cons"]=str(int(hh.consequence.sum())); T["sev_cons_pct"]=f"{hh.consequence.mean()*100:.0f}"; T["sev_cons_all"]=f"{sv.consequence.mean()*100:.1f}"

# sampled table
lines=[r"\begin{tabular}{@{}llrrrr@{}}",r"\toprule",r"Model & Defense & Mean & Any of 3 & All 3 & Greedy \\",r"\midrule"]
for m in FULL:
    for d in ["none","D2"]:
        x=sm[(sm.model==m)&(sm.defense==d)].iloc[0]
        lines.append(f"{SHORT[m] if d=='none' else ''} & {d} & {x.mean_sampled_rate*100:.1f} & {x.any_of_3*100:.1f} & {x.all_3*100:.1f} & {x.greedy_rate_same_cases*100:.1f} \\\\")
lines+=[r"\bottomrule",r"\end{tabular}"]; open("tabs/sampled.tex","w").write("\n".join(lines))
assert len(V["regen_models"])==1, "text differences are not from a single model: fix the sentence"

# ---- tokens for the restructured paper (extra analyses) ----
import json as _j, re as _re
Xn=_j.load(open("extra_numbers.json"))
bd=pd.DataFrame(V["h4"])
b2=[bd[(bd.model==m)&(bd.defense=="D2")].iloc[0]["best_of_8_rate"] for m in FULL]
T["bo8_D2_min"]=f"{min(b2)*100:.0f}"; T["bo8_D2_max"]=f"{max(b2)*100:.0f}"
O=Xn["ops"]
for fam,key in [("severity","sev"),("ioc","ioc"),("technique","tech")]:
    c_,i_,e_,lo_,hi_=O[f"{fam}|none"]
    T[f"ops_{key}_induced"]=f"{e_:.1f}"; T[f"ops_{key}_lo"]=f"{lo_:+.1f}".lstrip("+"); T[f"ops_{key}_hi"]=f"{hi_:+.1f}".lstrip("+")
    T[f"ops_{key}_D2"]=f"{O[f'{fam}|D2'][2]:.1f}"
T["ops_ioc_clean"]=f"{O['ioc|none'][0]:.1f}"
Vv={ (r[0],r[1]):r for r in Xn["validator"] }
T["val_inj_human"]=f"{Vv[('Injected reports','External check')][5]:.1f}"
T["val_clean_human"]=f"{Vv[('Clean twins','External check')][5]:.1f}"
T["val_inj_none_human"]=f"{Vv[('Injected reports','No defense')][5]:.1f}"
T["val_inj_ok"]=f"{Vv[('Injected reports','No defense')][2]:.1f}"
ld=Xn["load_clean"]; T["load_q3"]=f"{ld['Qwen3-4B']:.1f}"; T["load_phi"]=f"{ld['Phi-4-mini']:.1f}"
oth=[v for k,v in ld.items() if k not in ("Qwen3-4B","Phi-4-mini")]; T["load_other_min"]=f"{min(oth):.0f}"; T["load_other_max"]=f"{max(oth):.0f}"
L=Xn["lomo"]; loo={k:v for k,v in L.items() if k.startswith("Without")}
T["lomo_n"]=str(len(L))
T["lomo_atk_min"]=f"{min(v['atk'][0] for v in loo.values())*100:.1f}"; T["lomo_atk_max"]=f"{max(v['atk'][0] for v in loo.values())*100:.1f}"
T["four_atk"]=f"{L['Original four (confirmatory)']['atk'][0]*100:.1f}"
T["lomo_D2_abs_min"]=f"{abs(max(v['def_D2'][0] for v in loo.values()))*100:.1f}"; T["lomo_D2_abs_max"]=f"{abs(min(v['def_D2'][0] for v in loo.values()))*100:.1f}"
T["four_D2_abs"]=f"{abs(L['Original four (confirmatory)']['def_D2'][0])*100:.1f}"
d1=[v['def_D1'][0]*100 for v in loo.values()]; T["lomo_D1_min"]=S(f"{min(d1):.1f}"); T["lomo_D1_max"]=f"{max(d1):.1f}"
T["lomo_D1_zero"]=str(sum(1 for v in L.values() if v['def_D1'][1]<=0<=v['def_D1'][2]))
assert all(v["atk"][1]>0 for v in L.values()) and all(v["def_D2"][2]<0 for v in L.values()) and all(v["def_D3"][0]>0 for v in L.values())
# math minus in generated tables
for fn in ["tabs/lomo.tex","tabs/ops.tex","tabs/validator.tex"]:
    t=open(fn).read(); t=_re.sub(r"(?<![\w$])-(?=\d)","$-$",t); open(fn,"w").write(t)

# fill template
tpl=open("paper_template.tex").read()
missing=set()
def rep(m):
    k=m.group(1)
    if k not in T: missing.add(k); return "??"+k+"??"
    return str(T[k])
out=re.sub(r"@@([\w\.\-]+)@(?:@)?",rep,tpl)
open("paper.tex","w").write(out)
print("missing tokens:",sorted(missing))
print({k:T[k] for k in ["strat_low","strat_high","bo8_min","bo8_max","others_min","others_max","single_min","single_max","bo8_other_min","bo8_other_max","sam_gain_max","abs_q3b_inj","q3_correction","q3_system","q3_urgency","def_D2_ok","int_D2_ok_abs","n_sev_clean"]})
