"""SI-HIS Incident & Outbreak Reasoning Engine v2.

Evidence-weighted investigation triage for acute clusters, outbreaks,
common-exposure events and non-infectious incidents. This module deliberately
separates signal strength from etiologic or legal classification.
"""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd

GEO_LEVELS = ("Puskesmas", "Desa/Kelurahan", "Kecamatan", "Kabupaten", "Provinsi")
TIME_WINDOWS_MIN = (15, 30, 60, 180, 360, 720, 1440)

def _dt(df, names):
    for c in names:
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce")
            if s.notna().any():
                return s
    return pd.Series(pd.NaT, index=df.index)

def _first_col(df, names):
    for c in names:
        if c in df.columns:
            return c
    return None

def _geo_col(df):
    for c in GEO_LEVELS:
        if c in df.columns and df[c].notna().any():
            return c
    return None

def temporal_scan(df, windows=TIME_WINDOWS_MIN):
    t=_dt(df,("Tanggal Onset","Tanggal Sakit","Tanggal Pemeriksaan")).dropna().sort_values()
    if t.empty: return {"status":"insufficient_data","windows":[]}
    a=t.astype("int64").to_numpy()
    rows=[]
    for m in windows:
        d=int(pd.Timedelta(minutes=m).value)
        right=np.searchsorted(a,a+d,side="right")
        counts=right-np.arange(len(a))
        i=int(np.argmax(counts))
        n=int(counts[i])
        rows.append({"minutes":m,"cases":n,"start":t.iloc[i],"end":t.iloc[i]+pd.Timedelta(minutes=m),"density_per_hour":round(n/max(m/60,1/60),2)})
    return {"status":"ok","windows":rows,"best":max(rows,key=lambda x:x["density_per_hour"])}

def finest_spatiotemporal_cluster(df, windows=TIME_WINDOWS_MIN):
    geo=_geo_col(df)
    t=_dt(df,("Tanggal Onset","Tanggal Sakit","Tanggal Pemeriksaan"))
    if not geo or t.notna().sum()<2: return {"status":"insufficient_data","level":geo}
    w=df.copy();w["_t"]=t;w=w.dropna(subset=["_t",geo])
    best=None
    for unit,g in w.groupby(geo,dropna=False):
        scan=temporal_scan(g,windows)
        if scan.get("status")!="ok": continue
        b=scan["best"]
        cand={"unit":str(unit),"level":geo,**b}
        if best is None or (cand["cases"],cand["density_per_hour"])>(best["cases"],best["density_per_hour"]):
            best=cand
    return {"status":"signal" if best and best["cases"]>=5 else "no_signal","level":geo,"best_cluster":best}

def symptom_syndrome_clustering(df):
    candidates=("Gejala Utama","Gejala","Keluhan Utama","Symptom","Symptoms","Syndrome","Keluhan")
    c=_first_col(df,candidates)
    if not c: return {"status":"unavailable","column":None}
    s=df[c].fillna("").astype(str).str.lower().str.strip()
    valid=s[s.ne("") & ~s.isin({"nan","none","-","tidak ada"})]
    if valid.empty: return {"status":"unavailable","column":c}
    counts=valid.value_counts().head(10)
    return {"status":"ok","column":c,"unique_syndromes":int(valid.nunique()),"top_syndromes":counts.to_dict(),
            "interpretation":"Pengelompokan berbasis label gejala membantu membentuk sindrom operasional; bukan diagnosis klinis."}

def exposure_attack_rate(df):
    exp=_first_col(df,("Status Paparan","Terpapar","Exposed","Exposure Status","Riwayat Paparan"))
    outcome=_first_col(df,("Kasus","Case","Is_Case","Diagnosis Konfirm","Diagnosis Suspek","Diagnosis Probabel"))
    if not exp: return {"status":"unavailable","reason":"Kolom status paparan belum tersedia."}
    x=df[exp].astype(str).str.lower().str.strip()
    exposed=x.isin({"ya","yes","1","true","terpapar","exposed"})
    unexposed=x.isin({"tidak","no","0","false","tidak terpapar","unexposed"})
    if not (exposed.any() and unexposed.any()): return {"status":"insufficient_data","reason":"Kelompok exposed dan non-exposed belum lengkap."}
    if outcome:
        # A positive/recorded case is assumed only when a binary case field exists.
        vals=df[outcome]
        if pd.api.types.is_numeric_dtype(vals):
            case=pd.to_numeric(vals,errors="coerce").fillna(0).gt(0)
        else:
            z=vals.astype(str).str.lower().str.strip()
            case=z.isin({"ya","yes","1","true","kasus","konfirm","positif","terpapar"})
    else:
        case=pd.Series(True,index=df.index)
    ea=int((exposed&case).sum()); en=int(exposed.sum()); ua=int((unexposed&case).sum()); un=int(unexposed.sum())
    ar_e=ea/en if en else np.nan; ar_u=ua/un if un else np.nan
    rr=ar_e/ar_u if ar_u and np.isfinite(ar_u) else np.nan
    return {"status":"ok","exposed_n":en,"exposed_cases":ea,"unexposed_n":un,"unexposed_cases":ua,
            "attack_rate_exposed_pct":round(ar_e*100,2),"attack_rate_unexposed_pct":round(ar_u*100,2),
            "risk_ratio":round(float(rr),3) if np.isfinite(rr) else None,
            "interpretation":"Attack rate dan risk ratio hanya dapat diinterpretasikan bila denominator exposed/non-exposed dan definisi kasus valid."}

def facility_surge(df):
    facility=_first_col(df,("Puskesmas","Fasilitas Kesehatan","Facility","Rumah Sakit","RS","IGD"))
    t=_dt(df,("Tanggal Onset","Tanggal Pemeriksaan","Tanggal Pelaporan","Tanggal Sakit"))
    if not facility or t.notna().sum()==0: return {"status":"unavailable"}
    w=df.copy();w["_t"]=t;w=w.dropna(subset=["_t",facility]);daily=w.groupby([facility,w["_t"].dt.floor("D")]).size().reset_index(name="cases")
    if daily.empty:return {"status":"unavailable"}
    latest=daily["_t"].max();cur=daily[daily["_t"]==latest]
    rows=[]
    for f,g in daily.groupby(facility):
        hist=g[g["_t"]<latest]["cases"]
        cur_n=int(g.loc[g["_t"]==latest,"cases"].sum())
        base=float(hist.tail(28).mean()) if len(hist) else 0
        ratio=cur_n/max(base,1)
        rows.append({"facility":str(f),"latest_cases":cur_n,"baseline_daily_mean":round(base,2),"surge_ratio":round(ratio,2),"signal":bool(cur_n>=5 and ratio>=3)})
    out=pd.DataFrame(rows).sort_values(["signal","surge_ratio","latest_cases"],ascending=False)
    return {"status":"ok","latest_date":latest,"surge_facilities":out.head(10)}

def severity_mortality_burst(df):
    t=_dt(df,("Tanggal Onset","Tanggal Sakit","Tanggal Pemeriksaan"))
    if t.notna().sum()==0:return {"status":"unavailable"}
    sev=_first_col(df,("Severity","Keparahan","Is_Severe","Status Keparahan"))
    death=_first_col(df,("Is_Meninggal","Status Penderita","Death"))
    w=df.copy();w["_t"]=t;w=w.dropna(subset=["_t"]);latest=w["_t"].dt.floor("D").max()
    cur=w[w["_t"].dt.floor("D")==latest]; hist=w[w["_t"].dt.floor("D")<latest]
    def count_flag(data,col,death_mode=False):
        if not col:return 0
        s=data[col].astype(str).str.lower().str.strip()
        if death_mode:return int(s.isin({"1","true","ya","meninggal","death","died"}).sum())
        return int(s.isin({"1","true","ya","berat","severe","kritis","critical"}).sum())
    cur_sev=count_flag(cur,sev); cur_death=count_flag(cur,death,True)
    hist_daily=hist.groupby(hist["_t"].dt.floor("D")).size()
    base=float(hist_daily.tail(28).mean()) if len(hist_daily) else 0
    return {"status":"ok","latest_date":latest,"latest_cases":int(len(cur)),"latest_severe":cur_sev,"latest_deaths":cur_death,
            "case_burst_ratio":round(len(cur)/max(base,1),2),"severity_signal":cur_sev>0,"mortality_signal":cur_death>0}

def environmental_corrobation(df):
    cols=("Suhu","Curah Hujan","Kualitas Udara","Banjir","Longsor","Gempa","Kebakaran","Paparan Kimia","Air Tercemar","Makanan Bersama","Mass Gathering","Event")
    found={}
    for c in cols:
        if c in df.columns:
            vals=df[c].dropna()
            if len(vals): found[c]=vals.astype(str).value_counts().head(5).to_dict()
    return {"status":"available" if found else "unavailable","fields":found}

def incident_reasoning_v2(df, acute_event=None, infectious=None):
    temporal=temporal_scan(df); spatial=finest_spatiotemporal_cluster(df)
    symptoms=symptom_syndrome_clustering(df); attack=exposure_attack_rate(df)
    facility=facility_surge(df); severity=severity_mortality_burst(df); env=environmental_corrobation(df)
    signals=[]
    if temporal.get("best",{}).get("cases",0)>=5: signals.append("TEMPORAL_BURST")
    if spatial.get("best_cluster",{}).get("cases",0)>=5: signals.append("FINEST_SPATIOTEMPORAL_CLUSTER")
    if facility.get("status")=="ok" and isinstance(facility.get("surge_facilities"),pd.DataFrame) and facility["surge_facilities"].get("signal",pd.Series(dtype=bool)).any(): signals.append("FACILITY_SURGE")
    if severity.get("mortality_signal") or severity.get("severity_signal"): signals.append("SEVERITY_MORTALITY_BURST")
    if attack.get("status")=="ok" and attack.get("risk_ratio") is not None and attack["risk_ratio"]>=2: signals.append("EXPOSURE_ASSOCIATION")
    if symptoms.get("status")=="ok" and symptoms.get("unique_syndromes",0)<=3: signals.append("SYNDROME_CONCENTRATION")
    if acute_event and acute_event.get("signals"): signals.extend(acute_event["signals"])
    signals=list(dict.fromkeys(signals))
    evidence=min(100, len(signals)*15 + (20 if spatial.get("status")=="signal" else 0) + (15 if attack.get("status")=="ok" else 0))
    priority="URGENT_INVESTIGATION" if evidence>=60 or "MASS_CASUALTY_BURST" in signals else ("TARGETED_INVESTIGATION" if evidence>=30 else "ROUTINE_SURVEILLANCE")
    hypotheses=["Acute infectious outbreak","Keracunan/common exposure","Foodborne/toxin event","Paparan kimia/lingkungan","Mass gathering/common exposure","Bencana/incident non-infeksi"]
    return {"version":"2.0","status":"SIGNAL" if signals else "NO_STRONG_SIGNAL","priority":priority,
            "evidence_score":evidence,"signals":signals,"temporal":temporal,"spatiotemporal":spatial,
            "symptom_syndromes":symptoms,"attack_rate":attack,"facility_surge":facility,
            "severity_mortality":severity,"environmental":env,"hypotheses":hypotheses if signals else [],
            "guardrail":"Evidence score adalah skor triage internal, bukan probabilitas diagnosis. Tidak menetapkan etiologi, KLB legal, atau status bencana."}
