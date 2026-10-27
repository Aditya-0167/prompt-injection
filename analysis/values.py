import pandas as pd, numpy as np, json, zipfile, os, collections
import os as _o
ROOT=_o.environ.get("ARTIFACT_ROOT",_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
S=ROOT+"/statistics/"
BENCH=ROOT+"/benchmark/frozen_benchmark.json"
ORDER=["Qwen3-4B","Phi-4-mini","Qwen2.5-3B","Granite-3.3-2B","SmolLM2-1.7B","Qwen2.5-1.5B"]
SHORT={"Qwen3-4B":"Qwen3-4B","Phi-4-mini":"Phi-4-mini","Qwen2.5-3B":"Qwen2.5-3B","Granite-3.3-2B":"Granite-3.3-2B","SmolLM2-1.7B":"SmolLM2-1.7B","Qwen2.5-1.5B":"Qwen2.5-1.5B"}
FULL=["Qwen3-4B","Phi-4-mini","Qwen2.5-3B","Granite-3.3-2B"]       # models with all tiers
FAM=["severity","ioc","technique"]; DEF=["none","D1","D2","D3"]
rows=pd.read_csv(S+"all_rows.csv")
core=rows[rows.tier=="core"]
h1=pd.read_csv(S+"H1_attack_effect.csv"); h2=pd.read_csv(S+"H2_defense_effect.csv"); h3=pd.read_csv(S+"H3_defense_cost.csv")
pooled=pd.read_csv(S+"pooled.csv"); h4=pd.read_csv(S+"H4_adaptive_best_of_8.csv")
dab=pd.read_csv(S+"defense_aware_best_of_3.csv"); dabs=pd.read_csv(S+"defense_aware_by_strategy.csv")
sam=pd.read_csv(S+"sampled_decoding.csv"); tw=pd.read_csv(S+"outcome_three_way.csv"); cov=pd.read_csv(S+"coverage.csv")
def pp(x,d=1): return f"{x*100:.{d}f}"
def wavg(df,col,w="n"): return (df[col]*df[w]).sum()/df[w].sum()
V={}
V["rows_total"]=f"{len(rows):,}"
V["rows_core"]=f"{len(core):,}"
V["n_models"]=str(core.model.nunique()); V["n_cases"]=str(core.case_id.nunique())
# pooled
def P(label_sub):
    r=pooled[pooled.contrast.str.contains(label_sub,regex=False)]
    return r.iloc[0]
for d in DEF:
    r=P(f"attack effect (inj - clean), defense={d}")
    V[f"atk_{d}"]=pp(r.pooled_rd); V[f"atk_{d}_lo"]=pp(r.lo); V[f"atk_{d}_hi"]=pp(r.hi)
for d in DEF[1:]:
    r=P(f"defense effect on injected ({d} - none)")
    V[f"def_{d}"]=pp(r.pooled_rd); V[f"def_{d}_lo"]=pp(r.lo); V[f"def_{d}_hi"]=pp(r.hi)
    r=P(f"clean integrity change ({d} - none)")
    V[f"int_{d}"]=pp(r.pooled_rd); V[f"int_{d}_lo"]=pp(r.lo); V[f"int_{d}_hi"]=pp(r.hi)
# clean-twin rates
c=h1[h1.defense=="none"]
V["clean_sev_max"]=pp(c[c.family=="severity"].clean_rate.max()); V["clean_tech_max"]=pp(c[c.family=="technique"].clean_rate.max())
V["clean_ioc_max"]=pp(c[c.family=="ioc"].clean_rate.max()); V["clean_ioc_max_model"]=c.loc[c[c.family=="ioc"].clean_rate.idxmax(),"model"]
V["inj_overall_none"]=pp(wavg(c,"inj_rate"))
V["clean_overall_none"]=pp(wavg(c,"clean_rate"))
# attack effect significant after Holm
V["h1_sig"]=str(int((c.p_holm<0.05).sum())); V["h1_tests"]=str(len(c))
print(json.dumps(V,indent=0)[:3000])
print(c[["model","family","clean_rate","inj_rate","rd_inj_minus_clean","p_holm"]].round(3).to_string(index=False))

# ---------- more values ----------
# per-model averages over families (undefended attack effect; defended changes)
def model_avg(df, col, model, defense=None):
    t=df[(df.model==model)&((df.defense==defense) if defense else True)]
    return wavg(t,col)
V["auth_none"]={m: pp(wavg(h3[(h3.metric=="authorized_success")&(h3.model==m)],"none_rate")) for m in ORDER}
V["clean_int_none"]={m: pp(core[(core.model==m)&(core.report_kind=="clean")&(core.defense=="none")].integrity_ok.mean()) for m in ORDER}
V["inj_none_model"]={m: pp(wavg(h1[(h1.model==m)&(h1.defense=="none")],"inj_rate")) for m in ORDER}
V["clean_none_model"]={m: pp(wavg(h1[(h1.model==m)&(h1.defense=="none")],"clean_rate")) for m in ORDER}
# Spearman between authorized-instruction success and undefended susceptibility (n=6, descriptive)
from scipy.stats import spearmanr
a=[float(V["auth_none"][m]) for m in ORDER]; b=[float(V["inj_none_model"][m]) for m in ORDER]
rho,pval=spearmanr(a,b); V["rho"]=f"{rho:.2f}"; V["rho_p"]=f"{pval:.2f}"
# strategy and placement pooled (undefended, core)
inj=core[(core.report_kind=="inj")&(core.defense=="none")]
st=inj.groupby("strategy").hit.mean().sort_values(ascending=False)
V["strat"]={k: pp(v) for k,v in st.items()}; V["strat_order"]=list(st.index)
pl=inj.groupby("placement").hit.mean().sort_values(ascending=False)
V["plc"]={k: pp(v) for k,v in pl.items()}
# three-way pooled
g=tw.groupby("defense")[["n","compromised","abstained_manual_review","resisted_normal_output"]].sum()
for d in DEF:
    V[f"tw_comp_{d}"]=pp(g.loc[d,"compromised"]/g.loc[d,"n"]); V[f"tw_abs_{d}"]=pp(g.loc[d,"abstained_manual_review"]/g.loc[d,"n"]); V[f"tw_res_{d}"]=pp(g.loc[d,"resisted_normal_output"]/g.loc[d,"n"])
# abstention on clean reports under D2 for the two Qwen2.5 models
cl=core[core.report_kind=="clean"]
V["abst_clean_D2_q15"]=pp(cl[(cl.model=="Qwen2.5-1.5B")&(cl.defense=="D2")].abstain.mean())
V["abst_clean_D2_q3"]=pp(cl[(cl.model=="Qwen2.5-3B")&(cl.defense=="D2")].abstain.mean())
V["int_q15_none"]=pp(cl[(cl.model=="Qwen2.5-1.5B")&(cl.defense=="none")].integrity_ok.mean()); V["int_q15_D2"]=pp(cl[(cl.model=="Qwen2.5-1.5B")&(cl.defense=="D2")].integrity_ok.mean())
V["int_q3_none"]=pp(cl[(cl.model=="Qwen2.5-3B")&(cl.defense=="none")].integrity_ok.mean()); V["int_q3_D2"]=pp(cl[(cl.model=="Qwen2.5-3B")&(cl.defense=="D2")].integrity_ok.mean())
# H2 significance counts
V["h2_total"]=str(len(h2)//3)
for d in DEF[1:]:
    t=h2[h2.defense==d]
    V[f"h2_sig_better_{d}"]=str(int(((t.p_holm<0.05)&(t.rd_defended_minus_undefended<0)).sum()))
    V[f"h2_sig_worse_{d}"]=str(int(((t.p_holm<0.05)&(t.rd_defended_minus_undefended>0)).sum()))
# D2 on the 4 strongest-utility models (clean integrity >= 75%): benefit and cost
ok=[m for m in ORDER if float(V["clean_int_none"][m])>=75]
V["ok_models"]=ok
for d in ["D2"]:
    t=h2[(h2.defense==d)&(h2.model.isin(ok))]; V["def_D2_ok"]=pp(wavg(t,"rd_defended_minus_undefended"))
    t=h3[(h3.defense==d)&(h3.metric=="clean_integrity")&(h3.model.isin(ok))]; V["int_D2_ok"]=pp(wavg(t,"rd"))
# adaptive
V["h4"]=h4.round(4).to_dict("records"); V["dab"]=dab.round(4).to_dict("records")
# reproducibility across sessions (rows whose key was generated in >=2 distinct timestamps)
import glob as _g, os as _os2
_repro=_os2.path.join(_os2.path.dirname(_os2.path.abspath(__file__)),"reproducibility.json")
_zips=sorted(_g.glob(ROOT+"/checkpoints/cti_*.zip"))
if _zips:
    seen=collections.defaultdict(dict)
    for z in _zips:
        with zipfile.ZipFile(z) as zz:
            name=[x for x in zz.namelist() if x.endswith("raw/rows.jsonl")][0]
            for line in zz.open(name):
                try: r=json.loads(line)
                except: continue
                seen[r["key"]][r["timestamp"]]=(r["raw_output"],r["hit"],r["integrity_ok"])
    regen=[v_ for v_ in seen.values() if len(v_)>1]
    diff_text_keys=[k for k,v_ in seen.items() if len(v_)>1 and len({x[0] for x in v_.values()})>1]
    diff_score=[v_ for v_ in regen if len({(x[1],x[2]) for x in v_.values()})>1]
    _R={"regen_n":len(regen),"regen_text_diff":len(diff_text_keys),"regen_score_diff":len(diff_score),"text_diff_models":sorted({k.split("|")[0] for k in diff_text_keys})}
    json.dump(_R,open(_repro,"w"))
else:
    _R=json.load(open(_repro))
V["regen_n"]=f"{_R['regen_n']:,}"; V["regen_text_diff"]=str(_R["regen_text_diff"]); V["regen_score_diff"]=str(_R["regen_score_diff"]); V["regen_models"]=_R["text_diff_models"]
# parse validity and abstention
q=pd.read_csv(S+"quality.csv"); V["parse_min"]=pp(q.parser_valid.min()); V["parse_all"]=pp((q.parser_valid*q.n).sum()/q.n.sum())
# sampled
V["sam"]=sam.round(4).to_dict("records")
if __name__=="__main__":
    for k in ["auth_none","clean_int_none","inj_none_model","clean_none_model","rho","rho_p","strat","plc","tw_comp_none","tw_abs_none","tw_comp_D2","tw_abs_D2","tw_res_D2","h2_sig_better_D1","h2_sig_worse_D1","h2_sig_better_D2","h2_sig_worse_D2","h2_sig_better_D3","h2_sig_worse_D3","ok_models","def_D2_ok","int_D2_ok","regen_n","regen_text_diff","regen_score_diff","parse_min","parse_all","abst_clean_D2_q15","abst_clean_D2_q3","int_q15_none","int_q15_D2","int_q3_none","int_q3_D2"]:
        print(k,"=",V[k])
    print(pd.DataFrame(V["h4"]).to_string()); print(pd.DataFrame(V["dab"]).to_string()); print(pd.DataFrame(V["sam"]).to_string())


# ---- headline pooled numbers: use the seeded leave-one-out function so all numbers agree ----
import os as _os, json as _json
if _os.path.exists(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"extra_numbers.json")):
    _all=_json.load(open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"extra_numbers.json")))
    _p=_all.get("pooled_six")
    _s=lambda x: f"{x*100:.1f}"
    if _p: V["atk_none"],V["atk_none_lo"],V["atk_none_hi"]=[_s(x) for x in _p["atk"]]
    for _d in ["D1","D2","D3"]:
        if not _p: break
        V["def_"+_d],V["def_"+_d+"_lo"],V["def_"+_d+"_hi"]=[_s(x) for x in _p["def_"+_d]]
        V["int_"+_d],V["int_"+_d+"_lo"],V["int_"+_d+"_hi"]=[_s(x) for x in _p["int_"+_d]]
