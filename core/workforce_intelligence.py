"""Corporate Workforce Health Intelligence for SI-HIS / NutriMed MyLab.

Aggregates employee-linked health signals for corporate decision support.
The corporate layer intentionally avoids exposing individual clinical records:
- employee app remains the personal longitudinal health companion;
- corporate dashboard receives aggregate, de-identified workforce signals;
- minimum cohort threshold is enforced before subgroup display;
- predictions are surveillance/planning signals, not employment decisions.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

MIN_COHORT = 10
RANDOM_STATE = 42

def _num(s):
    return pd.to_numeric(s, errors="coerce")

def _binary(s):
    if s is None:
        return pd.Series(dtype="float")
    x=s.copy()
    n=_num(x)
    t=x.astype(str).str.strip().str.lower()
    m=t.map({"ya":1,"yes":1,"true":1,"1":1,"positif":1,"ada":1,
             "tidak":0,"no":0,"false":0,"0":0,"negatif":0,"tidak ada":0})
    return n.where(n.isin([0,1]),m)

def _age_band(s):
    a=_num(s)
    return pd.cut(a,[-np.inf,25,35,45,55,65,np.inf],
                  labels=["<25","25–34","35–44","45–54","55–64","65+"])

def prepare_workforce(df):
    """Normalize available employee-health fields without requiring employee PII."""
    d=df.copy()
    if "Umur" in d:
        d["Age_Group"]=_age_band(d["Umur"])
    if "Tanggal Sakit" in d:
        d["Tanggal Sakit"]=pd.to_datetime(d["Tanggal Sakit"],errors="coerce")
    if "Is_Meninggal" in d:
        d["_death"]=_binary(d["Is_Meninggal"]).fillna(0)
    elif "Status Penderita" in d:
        d["_death"]=d["Status Penderita"].astype(str).str.lower().eq("meninggal").astype(int)
    else:
        d["_death"]=0
    for col in ["BMI","Tekanan Darah","Sistolik","Diastolik","Gula Darah","Glukosa",
                "Kolesterol","Lingkar Perut"]:
        if col in d:
            d[col]=_num(d[col])
    return d

def workforce_overview(df):
    d=prepare_workforce(df)
    n=len(d)
    if not n:
        return {"status":"empty","message":"Belum ada data workforce."}
    departments=d["Departemen"].nunique() if "Departemen" in d else None
    sites=d["Lokasi Kerja"].nunique() if "Lokasi Kerja" in d else (d["Kabupaten"].nunique() if "Kabupaten" in d else None)
    metrics={"employees_observed":n}
    if departments is not None: metrics["departments"]=int(departments)
    if sites is not None: metrics["sites"]=int(sites)
    metrics["deaths"]=int(d["_death"].sum())
    if "Status Komorbid" in d:
        metrics["comorbidity_prevalence_pct"]=round(float(_binary(d["Status Komorbid"]).mean()*100),2)
    if "BMI" in d:
        metrics["mean_bmi"]=round(float(d["BMI"].mean()),2)
        metrics["bmi_high_pct"]=round(float((d["BMI"]>=25).mean()*100),2)
    if "Tekanan Darah" in d:
        metrics["high_bp_signal_pct"]=round(float((d["Tekanan Darah"]>=140).mean()*100),2)
    if "Gula Darah" in d:
        metrics["high_glucose_signal_pct"]=round(float((d["Gula Darah"]>=126).mean()*100),2)
    return {"status":"ok","metrics":metrics,"data":d}

def _aggregate_rate(d, group):
    if group not in d:
        return pd.DataFrame()
    rows=[]
    for key,g in d.groupby(group,dropna=False):
        n=len(g)
        if n < MIN_COHORT: continue
        row={group:str(key) if pd.notna(key) else "Tidak diketahui","N":n}
        if "_death" in g: row["Mortality_Count"]=int(g["_death"].sum())
        if "Status Komorbid" in g:
            row["Comorbidity_%"]=round(float(_binary(g["Status Komorbid"]).mean()*100),2)
        if "BMI" in g:
            row["BMI_High_%"]=round(float((g["BMI"]>=25).mean()*100),2)
        if "Tekanan Darah" in g:
            row["High_BP_Signal_%"]=round(float((g["Tekanan Darah"]>=140).mean()*100),2)
        if "Gula Darah" in g:
            row["High_Glucose_Signal_%"]=round(float((g["Gula Darah"]>=126).mean()*100),2)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("N",ascending=False).reset_index(drop=True)

def workforce_risk_segments(df):
    d=prepare_workforce(df)
    candidates=[]
    for col in ["Departemen","Unit Kerja","Lokasi Kerja","Pekerjaan","Age_Group","Jenis Kelamin"]:
        t=_aggregate_rate(d,col)
        if not t.empty:
            t.insert(0,"Dimension",col)
            candidates.append(t)
    return pd.concat(candidates,ignore_index=True) if candidates else pd.DataFrame()

def workforce_temporal(df):
    d=prepare_workforce(df)
    if "Tanggal Sakit" not in d or d["Tanggal Sakit"].isna().all():
        return pd.DataFrame()
    x=d.dropna(subset=["Tanggal Sakit"]).groupby(d["Tanggal Sakit"].dt.to_period("M")).size().rename("Health_Events").reset_index()
    x["Tanggal"]=x["Tanggal Sakit"].dt.to_timestamp()
    x["MoM_%"]=x["Health_Events"].pct_change()*100
    return x.drop(columns=["Tanggal Sakit"])

def workforce_predictive_signal(df):
    """Predict a generic workforce-health risk only when a valid binary health target exists."""
    d=prepare_workforce(df)
    target=None
    for col in ["High_Risk","Risiko_Tinggi","Health_Risk","Status Risiko"]:
        if col in d:
            y=_binary(d[col])
            if y.notna().sum()>=50 and y.dropna().nunique()==2:
                target=y
                break
    if target is None:
        return {"status":"not_available","message":"Target risiko workforce belum tersedia. Hubungkan hasil skrining/MCU tervalidasi sebagai target model; sistem tidak membuat label risiko dari asumsi."}
    feature_candidates=[c for c in ["Umur","Jenis Kelamin","Pekerjaan","Departemen","BMI","Tekanan Darah","Gula Darah","Kolesterol","Merokok","Aktivitas Fisik","Status Komorbid"] if c in d.columns]
    work=d[feature_candidates].copy()
    valid=target.notna()
    work=work.loc[valid]; y=target.loc[valid].astype(int)
    if len(work)<50 or y.nunique()<2:
        return {"status":"not_available","message":"Data valid untuk model predictive workforce belum mencukupi."}
    numeric=[c for c in feature_candidates if c in ["Umur","BMI","Tekanan Darah","Gula Darah","Kolesterol"]]
    categorical=[c for c in feature_candidates if c not in numeric]
    transformers=[]
    if numeric: transformers.append(("num",SimpleImputer(strategy="median"),numeric))
    if categorical: transformers.append(("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))]),categorical))
    prep=ColumnTransformer(transformers)
    model=Pipeline([("prep",prep),("model",RandomForestClassifier(n_estimators=300,min_samples_leaf=5,class_weight="balanced",random_state=RANDOM_STATE,n_jobs=-1))])
    cut=max(int(len(work)*.8),1)
    Xtr,Xte=work.iloc[:cut],work.iloc[cut:]; ytr,yte=y.iloc[:cut],y.iloc[cut:]
    if ytr.nunique()<2 or yte.nunique()<2:
        return {"status":"not_available","message":"Temporal holdout workforce belum memiliki dua kelas outcome."}
    model.fit(Xtr,ytr)
    prob=model.predict_proba(Xte)[:,1]
    return {"status":"ok","target":"workforce_health_risk","train_rows":len(Xtr),"holdout_rows":len(Xte),
            "mean_predicted_risk_pct":round(float(prob.mean()*100),2),
            "high_risk_signal_pct":round(float((prob>=0.5).mean()*100),2),
            "features":feature_candidates,
            "note":"Prediksi adalah sinyal agregat untuk perencanaan program kesehatan; bukan dasar keputusan ketenagakerjaan individual."}

def build_workforce_intelligence(df, company_name="Perusahaan"):
    overview=workforce_overview(df)
    if overview.get("status")!="ok":
        return overview
    d=overview["data"]
    return {
        "status":"ok","company":company_name,
        "overview":overview["metrics"],
        "segment_analysis":workforce_risk_segments(d),
        "temporal":workforce_temporal(d),
        "predictive":workforce_predictive_signal(d),
        "privacy":{"min_cohort":MIN_COHORT,"individual_health_records_exposed":False,
                   "principle":"Corporate dashboard receives aggregate workforce intelligence; NutriMed MyLab remains the individual longitudinal health layer."},
        "care_loop":"SCREENING → ANALYSIS → RISK STRATIFICATION → PREVENTION → CLINICAL REFERRAL → OUTCOME → RE-ANALYSIS",
    }


def workforce_policy_narrative(result):
    if not isinstance(result,dict) or result.get("status")!="ok":
        return "Workforce Health Intelligence belum dapat dibentuk karena data workforce belum mencukupi."
    m=result.get("overview",{})
    n=int(m.get("employees_observed",0))
    parts=[
        "### 🏢 WORKFORCE HEALTH INTELLIGENCE — NARASI AKADEMIK & KEBIJAKAN",
        "",
        f"Modul ini memandang perusahaan sebagai **populasi kesehatan**. Dengan {n:,} observasi kesehatan yang tersedia, SI-HIS mengubah data skrining, MCU, kunjungan kesehatan, perilaku dan outcome menjadi gambaran agregat mengenai status kesehatan tenaga kerja.",
        "",
        "Secara akademik, pendekatan ini mengikuti prinsip population health: unit analisis bukan hanya individu, tetapi distribusi faktor risiko, burden penyakit, perubahan temporal, kelompok kerja, lokasi kerja, dan outcome. Tujuannya adalah menemukan pola yang dapat ditindaklanjuti melalui promosi kesehatan, pencegahan, early detection, rujukan, dan evaluasi outcome.",
        "",
        "### 🔬 Dari Individual Health ke Workforce Health",
        "NutriMed MyLab menjadi **Personal Health Companion** bagi karyawan. Data yang berasal dari aktivitas kesehatan karyawan—misalnya skrining, hasil laboratorium, aktivitas, konsultasi, diet, pengobatan, dan follow-up—tetap berada pada lapisan personal. SI-HIS kemudian membentuk **intelligence agregat** untuk perusahaan sehingga manajemen memperoleh gambaran populasi tanpa harus membuka rekam kesehatan individual.",
        "",
        "### 🧠 Empat lapisan intelligence",
        "1. **Descriptive:** siapa yang menggunakan layanan, pola faktor risiko, burden penyakit, dan distribusi menurut unit/lokasi.",
        "2. **Analytical:** hubungan pola kesehatan dengan kelompok kerja, waktu, lokasi, faktor risiko, dan outcome; asosiasi tidak otomatis berarti kausalitas.",
        "3. **Predictive:** memperkirakan perubahan burden atau risiko workforce bila tersedia outcome/target yang tervalidasi dan model melewati validasi temporal.",
        "4. **Preventive & Prescriptive:** menerjemahkan sinyal menjadi program kesehatan populasi—misalnya skrining ulang, edukasi, intervensi gaya hidup, occupational-health review, rujukan dan monitoring—dengan keputusan akhir berada pada tenaga kesehatan dan manajemen sesuai kewenangan.",
        "",
        "### 🎯 Untuk pengambil keputusan perusahaan",
        "Dashboard sebaiknya menjawab lima pertanyaan: **(1) bagaimana status kesehatan workforce saat ini; (2) faktor risiko apa yang paling sering muncul; (3) unit/lokasi mana yang menunjukkan perubahan pola; (4) apa yang diproyeksikan bila tren berlanjut; dan (5) program preventif apa yang perlu dievaluasi serta bagaimana outcome-nya diukur.**",
        "",
        "### 🛡️ Guardrail tata kelola",
        "Data kesehatan merupakan data pribadi spesifik dalam UU PDP. Karena itu, dashboard korporat dirancang berbasis **agregasi, pembatasan ukuran kelompok, minimisasi data, kontrol akses, audit trail, tujuan pemrosesan yang jelas, dan pemisahan antara data personal karyawan dengan intelligence organisasi**. Ambang minimum cohort pada prototype adalah 10 observasi dan harus dikaji kembali bersama DPO/legal/occupational-health governance perusahaan.",
        "",
        "### 🔄 Closed-loop Workforce Health",
        "SCREENING → PERSONAL HEALTH JOURNEY → POPULATION ANALYSIS → RISK STRATIFICATION → PREDICTION → PREVENTION/PRESCRIPTION → INTERVENTION → OUTCOME → NEW DATA → CONTINUOUS LEARNING",
        "",
        "Dalam implementasi produksi, dashboard tidak boleh menjadi alat untuk menilai kelayakan kerja, promosi, pemutusan hubungan kerja, atau diskriminasi berdasarkan kondisi kesehatan. Fungsi utamanya adalah **workforce health improvement, prevention, occupational-health surveillance, service planning, dan pengukuran outcome**."
    ]
    return "\n".join(parts)
