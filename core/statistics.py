"""Outcome-specific epidemiological statistics for SI-HIS.

The module deliberately separates disease, severity and mortality outcomes.
Sparse binary outcomes use a Firth-style bias-reduced logistic fallback when
ordinary logistic regression is vulnerable to zero cells/separation.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

AGE_GROUPS=["<1 tahun","1-4 tahun","5-9 tahun","10-14 tahun","15-19 tahun","20-24 tahun","25-34 tahun","35-44 tahun","45-54 tahun","55-64 tahun","65-74 tahun","75-84 tahun","≥85 tahun"]
AGE_BINS=[-np.inf,1,5,10,15,20,25,35,45,55,65,75,85,np.inf]
INDEPENDENT_VARS=["Umur","Jenis Kelamin","Pekerjaan","Status Imunisasi","Status Komorbid","Riwayat Perjalanan","Faktor Risiko Lain"]


def add_age_groups(df):
    work=df.copy(deep=True)
    if "Umur" not in work.columns:return work
    age=pd.to_numeric(work["Umur"],errors="coerce")
    work["Kelompok_Umur"]=pd.cut(age,bins=AGE_BINS,labels=AGE_GROUPS,right=False,include_lowest=True)
    work["Kategori_Umur"]=work["Kelompok_Umur"].astype("string").map({label:i+1 for i,label in enumerate(AGE_GROUPS)}).astype("Int64")
    return work


def prepare_binary_target(df,target="Is_Konfirm"):
    work=add_age_groups(df)
    if target not in work.columns:work[target]=np.nan
    values=pd.to_numeric(work[target],errors="coerce")
    if values.notna().sum()==0 and target=="Is_Konfirm" and "Diagnosis Konfirm" in work.columns:
        values=work["Diagnosis Konfirm"].astype(str).str.strip().str.lower().ne("bukan").astype(int)
    work[target]=values.where(values.isin([0,1]))
    return work


def _valid_contingency(ct):
    if ct is None or ct.empty:return pd.DataFrame()
    ct=ct.copy()
    return ct.loc[ct.sum(axis=1)>0,ct.sum(axis=0)>0]


def _chi_square(ct):
    valid=_valid_contingency(ct)
    if valid.shape[0]<2 or valid.shape[1]<2:return np.nan,np.nan
    try:
        chi2,p,_,_=stats.chi2_contingency(valid)
        return float(chi2),float(p)
    except (ValueError,ZeroDivisionError,TypeError):return np.nan,np.nan


def _or_by_level(ct):
    """Crude OR versus the first observed exposure level, with 0.5 continuity correction."""
    if ct is None or ct.empty or 0 not in ct.columns or 1 not in ct.columns:return pd.DataFrame()
    levels=list(ct.index)
    if len(levels)<2:return pd.DataFrame()
    ref=levels[0];ref0=float(ct.loc[ref,0]);ref1=float(ct.loc[ref,1]);rows=[]
    for level in levels:
        if level==ref:
            rows.append({"Kategori":level,"Referensi":ref,"cOR":1.0,"cOR Lower 95%":np.nan,"cOR Upper 95%":np.nan});continue
        a=float(ct.loc[level,1])+.5;b=float(ct.loc[level,0])+.5;c=ref1+.5;d=ref0+.5
        odds=(a*d)/(b*c);se=math.sqrt(1/a+1/b+1/c+1/d)
        rows.append({"Kategori":level,"Referensi":ref,"cOR":odds,"cOR Lower 95%":math.exp(math.log(odds)-1.96*se),"cOR Upper 95%":math.exp(math.log(odds)+1.96*se)})
    return pd.DataFrame(rows).round(4)

def _binary_table(work,variable,target="Is_Konfirm"):
    if variable not in work.columns or target not in work.columns:
        return {"crosstab":pd.DataFrame(),"chi2":np.nan,"p_value":np.nan,"or_by_group":pd.DataFrame(),"status":"variable tidak tersedia"}
    d=work.dropna(subset=[variable,target]).copy()
    if d.empty:return {"crosstab":pd.DataFrame(),"chi2":np.nan,"p_value":np.nan,"or_by_group":pd.DataFrame(),"status":"tidak ada observasi lengkap"}
    x=d[variable].astype(str);y=pd.to_numeric(d[target],errors="coerce")
    ct=pd.crosstab(x,y);chi2,p=_chi_square(ct);ors=_or_by_level(ct)
    status="ok" if pd.notna(p) else "outcome hanya memiliki satu kategori atau tabel tidak memenuhi syarat uji"
    return {"crosstab":ct,"chi2":chi2,"p_value":p,"or_by_group":ors,"status":status}


def _age_analysis(work,target="Is_Konfirm"):
    if "Kelompok_Umur" not in work.columns or target not in work.columns:
        return {"crosstab":pd.DataFrame(),"chi2":np.nan,"p_value":np.nan,"or_by_group":pd.DataFrame(),"status":"variabel umur/outcome tidak tersedia"}
    d=work.dropna(subset=["Kelompok_Umur",target]).copy()
    if d.empty:return {"crosstab":pd.DataFrame(),"chi2":np.nan,"p_value":np.nan,"or_by_group":pd.DataFrame(),"status":"tidak ada observasi lengkap"}
    age=d["Kelompok_Umur"].astype(str);y=pd.to_numeric(d[target],errors="coerce")
    ct=pd.crosstab(age,y);ct=ct.reindex([g for g in AGE_GROUPS if g in ct.index])
    chi2,p=_chi_square(ct);levels=[g for g in AGE_GROUPS if g in ct.index and ct.loc[g].sum()>0];rows=[]
    if levels:
        ref_name=levels[0];ref=ct.loc[ref_name]
        for level in levels:
            if level==ref_name:
                rows.append({"Kelompok Umur":level,"Referensi":ref_name,"OR":1.0,"OR_Lower_95%":np.nan,"OR_Upper_95%":np.nan});continue
            row=ct.loc[level];a=float(row.get(1,0))+.5;b=float(row.get(0,0))+.5;c=float(ref.get(1,0))+.5;e=float(ref.get(0,0))+.5
            or_value=(a*e)/(b*c);se=math.sqrt(1/a+1/b+1/c+1/e)
            rows.append({"Kelompok Umur":level,"Referensi":ref_name,"OR":or_value,"OR_Lower_95%":math.exp(math.log(or_value)-1.96*se),"OR_Upper_95%":math.exp(math.log(or_value)+1.96*se)})
    status="ok" if pd.notna(p) else "outcome hanya memiliki satu kategori atau tabel tidak memenuhi syarat uji"
    return {"crosstab":ct,"chi2":chi2,"p_value":p,"or_by_group":pd.DataFrame(rows).round(4),"status":status}


def _firth_fit(X,y,max_iter=100,tol=1e-7):
    X=np.asarray(X,float);y=np.asarray(y,float);p=X.shape[1];beta=np.zeros(p,float)
    def objective(b):
        eta=np.clip(X@b,-35,35);mu=1/(1+np.exp(-eta));w=np.clip(mu*(1-mu),1e-10,None);info=X.T@(w[:,None]*X)+np.eye(p)*1e-10
        sign,logdet=np.linalg.slogdet(info)
        if sign<=0:return np.inf
        ll=np.sum(y*np.log(np.clip(mu,1e-12,1))+(1-y)*np.log(np.clip(1-mu,1e-12,1)))
        return -(ll+0.5*logdet)
    for _ in range(max_iter):
        eta=np.clip(X@beta,-35,35);mu=1/(1+np.exp(-eta));w=np.clip(mu*(1-mu),1e-10,None);info=X.T@(w[:,None]*X)+np.eye(p)*1e-10;inv=np.linalg.pinv(info)
        h=np.sum((X@inv)*X,axis=1)*w;score=X.T@(y-mu+h*(0.5-mu));step=inv@score
        if not np.all(np.isfinite(step)):return None,None,False
        old=beta.copy();scale=1.0;old_obj=objective(old);candidate=old+step
        while scale>=1/128:
            candidate=old+scale*step
            if objective(candidate)<=old_obj:break
            scale/=2
        beta=candidate
        if np.max(np.abs(beta-old))<tol:return beta,inv,True
    return beta,inv,False


def _has_sparse_cells(model_df,variables,target):
    for col in variables:
        ct=pd.crosstab(model_df[col].astype(str),model_df[target])
        if ct.shape[0]>=2 and ct.shape[1]>=2 and (ct==0).any().any():return True
    return False


def multivariable_logistic(df,target="Is_Konfirm"):
    work=prepare_binary_target(df,target)
    variables=["Kelompok_Umur","Jenis Kelamin","Pekerjaan","Status Imunisasi","Status Komorbid","Riwayat Perjalanan","Faktor Risiko Lain"]
    available=[v for v in variables if v in work.columns]
    if not available:return pd.DataFrame([{"Variabel":"MODEL_NOT_RUN","Keterangan":"Tidak tersedia variabel independen."}])
    model_df=work[available+[target]].dropna().copy()
    if model_df.empty or model_df[target].nunique()<2 or len(model_df)<30:
        return pd.DataFrame([{"Variabel":"MODEL_NOT_RUN","Keterangan":"Outcome harus memiliki dua kategori (0/1) dan minimal 30 observasi lengkap."}])
    for col in available:model_df[col]=model_df[col].astype(str)
    X=pd.get_dummies(model_df[available],columns=available,drop_first=True,dtype=float)
    y=pd.to_numeric(model_df[target],errors="coerce").astype(int)
    X=X.replace([np.inf,-np.inf],np.nan).dropna();y=y.loc[X.index]
    X=X[[c for c in X.columns if X[c].nunique(dropna=False)>1]]
    if X.empty or y.nunique()<2:return pd.DataFrame([{"Variabel":"MODEL_NOT_RUN","Keterangan":"Tidak tersedia variasi prediktor/outcome yang memadai."}])
    X_const=sm.add_constant(X,has_constant="add");Xmat=X_const.to_numpy(float);use_firth=_has_sparse_cells(model_df,available,target)
    if use_firth:
        beta,cov,converged=_firth_fit(Xmat,y.to_numpy())
        if beta is not None and cov is not None and converged:
            se=np.sqrt(np.clip(np.diag(cov),1e-12,None));z=beta/se;pvals=2*stats.norm.sf(np.abs(z));lower=beta-1.96*se;upper=beta+1.96*se
            out=pd.DataFrame({"Variabel":list(X_const.columns),"Koefisien (β)":beta,"OR Adjusted":np.exp(np.clip(beta,-50,50)),"OR Lower 95%":np.exp(np.clip(lower,-50,50)),"OR Upper 95%":np.exp(np.clip(upper,-50,50)),"p-value":pvals,"Metode":"Firth Logistic Regression"})
            return out[out["Variabel"]!="const"].round(4)
    try:
        model=sm.Logit(y,X_const).fit(disp=False,maxiter=300)
        conf=model.conf_int();out=pd.DataFrame({"Variabel":model.params.index,"Koefisien (β)":model.params.values,"OR Adjusted":np.exp(model.params.values),"OR Lower 95%":np.exp(conf[0].values),"OR Upper 95%":np.exp(conf[1].values),"p-value":model.pvalues.values,"Metode":"Logistic Regression"})
        return out[out["Variabel"]!="const"].round(4)
    except Exception as exc:
        return pd.DataFrame([{"Variabel":"MODEL_ERROR","Keterangan":str(exc)}])


def hitung_bivariat_lengkap(df,var_indep=None,var_dep_binary=None):
    """Run complete bivariate + multivariable analysis for one explicit outcome.

    If var_dep_binary is supplied without var_indep, every configured
    independent variable is analyzed against that outcome. This is the
    foundation for disease, severity and mortality to be modeled separately.
    """
    target=var_dep_binary or "Is_Konfirm"
    work=prepare_binary_target(df,target)
    if var_indep is not None:return _binary_table(work,var_indep,target)
    results={"Outcome":target,"Umur":_age_analysis(work,target)}
    for variable in INDEPENDENT_VARS[1:]:
        if variable in work.columns:results[variable]=_binary_table(work,variable,target)
    results["MULTIVARIAT — Logistic Regression"]=multivariable_logistic(work,target)
    return results
