"""Clinical Laboratory Intelligence — separate from epidemiological intelligence."""
from __future__ import annotations
import pandas as pd
import numpy as np

LAB_ALIASES={
    "patient_id":["Patient ID","patient_id","ID Pasien","NIK"],
    "test":["Test","Pemeriksaan","Nama Pemeriksaan","Analyte"],
    "result":["Result","Hasil","Nilai"],
    "unit":["Unit","Satuan"],
    "date":["Date","Tanggal","Tanggal Pemeriksaan"],
    "low":["Reference Low","Ref Low","Nilai Rujukan Min"],
    "high":["Reference High","Ref High","Nilai Rujukan Max"],
}

def _col(df,key):
    for c in LAB_ALIASES[key]:
        if c in df.columns:return c
    return None

def reference_range_intelligence(df):
    result=df.copy(deep=True)
    r=_col(result,"result"); lo=_col(result,"low"); hi=_col(result,"high")
    if not r:return {"status":"INSUFFICIENT_DATA","reason":"Result/Hasil column not found."}
    result["_result"]=pd.to_numeric(result[r],errors="coerce")
    if lo and hi:
        result["_low"]=pd.to_numeric(result[lo],errors="coerce"); result["_high"]=pd.to_numeric(result[hi],errors="coerce")
        result["reference_flag"]=np.select([result["_result"]<result["_low"],result["_result"]>result["_high"]],["LOW","HIGH"],default="NORMAL")
    else:
        result["reference_flag"]="UNKNOWN"
    return {"status":"OK","table":result.drop(columns=[c for c in ["_result","_low","_high"] if c in result],errors="ignore")}

def detect_lab_anomalies(df):
    rr=reference_range_intelligence(df)
    if rr["status"]!="OK": return rr
    t=rr["table"]; flag=t["reference_flag"]
    return {"status":"OK","abnormal_count":int(flag.isin(["LOW","HIGH"]).sum()),"critical_candidate_count":0,"abnormal_results":t.loc[flag.isin(["LOW","HIGH"])].copy()}

def lab_delta_check(df, threshold=0.2):
    pid=_col(df,"patient_id"); test=_col(df,"test"); r=_col(df,"result"); date=_col(df,"date")
    if not all([pid,test,r,date]): return {"status":"INSUFFICIENT_DATA","reason":"Patient, test, result and date are required."}
    x=df.copy();x["_r"]=pd.to_numeric(x[r],errors="coerce");x["_date"]=pd.to_datetime(x[date],errors="coerce");x=x.sort_values([pid,test,"_date"])
    x["delta"]=x.groupby([pid,test])["_r"].diff();x["delta_pct"]=x.groupby([pid,test])["_r"].pct_change()*100
    x["delta_flag"]=x["delta_pct"].abs().ge(float(threshold)*100)
    return {"status":"OK","table":x[[pid,test,date,r,"delta","delta_pct","delta_flag"]].copy()}

def critical_value_detection(df, critical_rules=None):
    critical_rules=critical_rules or {}
    test=_col(df,"test");r=_col(df,"result")
    if not test or not r:return {"status":"INSUFFICIENT_DATA","reason":"Test and result are required."}
    x=df.copy();x["_result"]=pd.to_numeric(x[r],errors="coerce");flags=[]
    for i,row in x.iterrows():
        rule=critical_rules.get(str(row[test]),{})
        hit=(("low" in rule and pd.notna(row["_result"]) and row["_result"]<rule["low"]) or ("high" in rule and pd.notna(row["_result"]) and row["_result"]>rule["high"]))
        flags.append(bool(hit))
    x["critical_flag"]=flags
    return {"status":"OK","table":x.drop(columns=["_result"]),"critical_count":int(x["critical_flag"].sum())}

def lab_data_quality(df):
    issues={}
    issues["duplicate_rows"]=int(df.duplicated().sum())
    issues["missing_result"]=int(pd.to_numeric(df[_col(df,"result")],errors="coerce").isna().sum()) if _col(df,"result") else len(df)
    return {"status":"OK","issues":issues}

def lab_trajectory(df, patient_id=None):
    pid=_col(df,"patient_id");test=_col(df,"test");r=_col(df,"result");date=_col(df,"date")
    if not all([pid,test,r,date]):return {"status":"INSUFFICIENT_DATA"}
    x=df.copy();x["_date"]=pd.to_datetime(x[date],errors="coerce");x["_result"]=pd.to_numeric(x[r],errors="coerce")
    if patient_id is not None:x=x[x[pid].astype(str)==str(patient_id)]
    x=x.sort_values([pid,test,"_date"]);x["trend"]=x.groupby([pid,test])["_result"].diff()
    return {"status":"OK","table":x[[pid,test,date,r,"trend"]].copy()}

def laboratory_intelligence(df, patient_id=None):
    out={"module":"Laboratory Intelligence","intended_purpose":"Clinical laboratory decision support","reference_range":reference_range_intelligence(df),"anomalies":detect_lab_anomalies(df),"delta_check":lab_delta_check(df),"critical_values":critical_value_detection(df),"trajectory":lab_trajectory(df,patient_id),"data_quality":lab_data_quality(df)}
    out["guardrail"]="Signal/recommendation for professional review; not an autonomous diagnosis or treatment order."
    return out
