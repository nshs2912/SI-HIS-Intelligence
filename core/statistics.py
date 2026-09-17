"""Epidemiological statistical analysis layer."""
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
    # Convert to string before mapping so pandas categorical behavior cannot
    # propagate into downstream crosstab/reindex operations.
    work["Kategori_Umur"]=work["Kelompok_Umur"].astype("string").map({label:i+1 for i,label in enumerate(AGE_GROUPS)}).astype("Int64")
    return work

def prepare_binary_target(df,target="Is_Konfirm"):
    work=add_age_groups(df)
    if target not in work.columns:work[target]=np.nan
    values=pd.to_numeric(work[target],errors="coerce")
    if values.notna().sum()==0:
        if target=="Is_Konfirm" and "Diagnosis Konfirm" in work.columns:
            values=work["Diagnosis Konfirm"].astype(str).str.strip().str.lower().ne("bukan").astype(int)
    # Binary targets are normalized to 0/1. Values outside 0/1 are treated as missing.
    values=values.where(values.isin([0,1]))
    work[target]=values
    return work

def _valid_contingency(ct):
    if ct is None or ct.empty:return pd.DataFrame()
    ct=ct.copy()
    ct=ct.loc[ct.sum(axis=1)>0,ct.sum(axis=0)>0]
    return ct

def _chi_square(ct):
    valid=_valid_contingency(ct)
    if valid.shape[0]<2 or valid.shape[1]<2:return np.nan,np.nan
    try:
        chi2,p_value,_,_=stats.chi2_contingency(valid)
        return float(chi2),float(p_value)
    except (ValueError,ZeroDivisionError,TypeError):return np.nan,np.nan

def _binary_table(work,variable,target="Is_Konfirm"):
    if variable not in work.columns or target not in work.columns:
        return {"crosstab":pd.DataFrame(),"chi2":np.nan,"p_value":np.nan,"status":"variable tidak tersedia"}
    d=work.dropna(subset=[variable,target]).copy()
    if d.empty:return {"crosstab":pd.DataFrame(),"chi2":np.nan,"p_value":np.nan,"status":"tidak ada observasi lengkap"}
    # Cast to string for categorical predictors; this avoids pandas-version
    # dependent behavior with categorical indexes.
    x=d[variable].astype(str)
    y=pd.to_numeric(d[target],errors="coerce")
    ct=pd.crosstab(x,y)
    chi2,p_value=_chi_square(ct)
    status="ok" if pd.notna(p_value) else "outcome hanya memiliki satu kategori atau tabel tidak memenuhi syarat uji"
    return {"crosstab":ct,"chi2":chi2,"p_value":p_value,"status":status}

def _age_analysis(work,target="Is_Konfirm"):
    if "Kelompok_Umur" not in work.columns or target not in work.columns:
        return {"crosstab":pd.DataFrame(),"chi2":np.nan,"p_value":np.nan,"or_by_group":pd.DataFrame(),"status":"variabel umur/outcome tidak tersedia"}
    d=work.dropna(subset=["Kelompok_Umur",target]).copy()
    if d.empty:return {"crosstab":pd.DataFrame(),"chi2":np.nan,"p_value":np.nan,"or_by_group":pd.DataFrame(),"status":"tidak ada observasi lengkap"}
    # Do not pass observed=True to pd.crosstab: older pandas versions used by
    # some Streamlit deployments can reject that keyword. String conversion
    # also removes unused categorical levels safely.
    age=d["Kelompok_Umur"].astype(str)
    y=pd.to_numeric(d[target],errors="coerce")
    ct=pd.crosstab(age,y)
    ct=ct.reindex([g for g in AGE_GROUPS if g in ct.index])
    chi2,p_value=_chi_square(ct)
    levels=[g for g in AGE_GROUPS if g in ct.index and ct.loc[g].sum()>0]
    rows=[]
    if levels:
        ref_name=levels[0];ref=ct.loc[ref_name]
        for level in levels:
            if level==ref_name:
                rows.append({"Kelompok Umur":level,"Referensi":ref_name,"OR":1.0,"OR_Lower_95%":np.nan,"OR_Upper_95%":np.nan});continue
            row=ct.loc[level]
            a,b,c,e=float(row.get(1,0))+.5,float(row.get(0,0))+.5,float(ref.get(1,0))+.5,float(ref.get(0,0))+.5
            or_value=(a*e)/(b*c);se=math.sqrt(1/a+1/b+1/c+1/e)
            rows.append({"Kelompok Umur":level,"Referensi":ref_name,"OR":or_value,"OR_Lower_95%":math.exp(math.log(or_value)-1.96*se),"OR_Upper_95%":math.exp(math.log(or_value)+1.96*se)})
    status="ok" if pd.notna(p_value) else "outcome hanya memiliki satu kategori atau tabel tidak memenuhi syarat uji"
    return {"crosstab":ct,"chi2":chi2,"p_value":p_value,"or_by_group":pd.DataFrame(rows).round(4),"status":status}

def multivariable_logistic(df,target="Is_Konfirm"):
    work=prepare_binary_target(df,target)
    variables=["Kelompok_Umur","Jenis Kelamin","Pekerjaan","Status Imunisasi","Status Komorbid","Riwayat Perjalanan","Faktor Risiko Lain"]
    available=[v for v in variables if v in work.columns]
    if not available:return pd.DataFrame()
    model_df=work[available+[target]].dropna().copy()
    if model_df.empty or model_df[target].nunique()<2 or len(model_df)<30:
        return pd.DataFrame([{"Variabel":"MODEL_NOT_RUN","Keterangan":"Outcome harus memiliki dua kategori (0/1) dan minimal 30 observasi lengkap."}])
    for col in available:model_df[col]=model_df[col].astype(str)
    X=pd.get_dummies(model_df[available],columns=available,drop_first=True,dtype=float)
    y=pd.to_numeric(model_df[target],errors="coerce").astype(int)
    X=X.replace([np.inf,-np.inf],np.nan).dropna();y=y.loc[X.index]
    X=X[[c for c in X.columns if X[c].nunique(dropna=False)>1]]
    if X.empty or y.nunique()<2:return pd.DataFrame([{"Variabel":"MODEL_NOT_RUN","Keterangan":"Tidak tersedia variasi prediktor/outcome yang memadai."}])
    try:
        model=sm.Logit(y,sm.add_constant(X,has_constant="add")).fit(disp=False,maxiter=300)
        conf=model.conf_int()
        out=pd.DataFrame({"Variabel":model.params.index,"Koefisien (β)":model.params.values,"OR Adjusted":np.exp(model.params.values),"OR Lower 95%":np.exp(conf[0].values),"OR Upper 95%":np.exp(conf[1].values),"p-value":model.pvalues.values})
        out=out[out["Variabel"]!="const"].copy();out["Signifikan (p<0,05)"]=out["p-value"]<.05
        return out.round(4)
    except Exception as exc:
        return pd.DataFrame([{"Variabel":"MODEL_ERROR","Keterangan":str(exc)}])

def hitung_bivariat_lengkap(df,var_indep=None,var_dep_binary=None):
    # Backward-compatible single-variable API used by existing callers.
    if var_indep is not None and var_dep_binary is not None:
        work=prepare_binary_target(df,var_dep_binary)
        return _binary_table(work,var_indep,var_dep_binary)
    work=prepare_binary_target(df)
    results={"Umur":_age_analysis(work)}
    for variable in INDEPENDENT_VARS[1:]:
        if variable in work.columns:results[variable]=_binary_table(work,variable)
    results["MULTIVARIAT — Logistic Regression"]=multivariable_logistic(work)
    return results
