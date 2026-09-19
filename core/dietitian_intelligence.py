"""Clinical Dietitian Intelligence — personalized nutrition decision support."""
from __future__ import annotations
import pandas as pd
import numpy as np

def _first(df,names):return next((n for n in names if n in df.columns),None)

def anthropometric_trajectory(df):
    pid=_first(df,["Patient ID","patient_id","ID Pasien"]);date=_first(df,["Date","Tanggal"]);weight=_first(df,["Weight","Berat Badan"]);bmi=_first(df,["BMI","IMT"])
    if not pid or not date:return {"status":"INSUFFICIENT_DATA"}
    x=df.copy();x["_date"]=pd.to_datetime(x[date],errors="coerce");cols=[pid,date]+([weight] if weight else [])+([bmi] if bmi else [])
    return {"status":"OK","table":x.sort_values([pid,"_date"])[cols].copy()}

def nutrition_adequacy(df):
    kcal=_first(df,["Calories","Kcal","Energi"]);protein=_first(df,["Protein","Protein g"]);fiber=_first(df,["Fiber","Serat"]);sodium=_first(df,["Sodium","Natrium"]);sugar=_first(df,["Sugar","Gula"])
    if not any([kcal,protein,fiber,sodium,sugar]):return {"status":"INSUFFICIENT_DATA"}
    out={}
    for name,col in [("kcal",kcal),("protein",protein),("fiber",fiber),("sodium",sodium),("sugar",sugar)]:
        if col:out[name]={"mean":float(pd.to_numeric(df[col],errors="coerce").mean())}
    return {"status":"OK","summary":out}

def disease_specific_pattern(df):
    disease=_first(df,["Diagnosis","Condition","Disease"]);diet=_first(df,["Diet Plan","Diet","Pola Makan"])
    if not disease or not diet:return {"status":"INSUFFICIENT_DATA"}
    return {"status":"OK","table":df[[disease,diet]].copy()}

def adherence_signal(df):
    c=_first(df,["Adherence","Kepatuhan Diet","Diet Adherence"])
    if not c:return {"status":"INSUFFICIENT_DATA"}
    x=df.copy();x["_adherence"]=pd.to_numeric(x[c],errors="coerce");return {"status":"OK","mean_adherence":float(x["_adherence"].mean()) if x["_adherence"].notna().any() else None}

def dietitian_intelligence(df):
    return {"module":"Dietitian Intelligence","anthropometric":anthropometric_trajectory(df),"nutrition_adequacy":nutrition_adequacy(df),"disease_pattern":disease_specific_pattern(df),"adherence":adherence_signal(df),"guardrail":"Nutrition recommendations are decision support and require dietitian/clinical review; clinical context and medication effects must be considered."}
