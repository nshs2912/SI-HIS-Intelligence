"""Healthcare Provider Intelligence for SI-HIS.

Provider-level intelligence for hospitals, clinics, laboratories and pharmacies.
The module is designed as an aggregate decision-support layer and can ingest
FHIR-aligned operational data from RME/LIS/PIS systems.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

MIN_COHORT = 10

PROVIDER_TYPES = {
    "Rumah Sakit": ["rawat jalan","igd","rawat inap","operasi","laboratorium","radiologi","farmasi"],
    "Klinik": ["rawat jalan","tindakan","laboratorium","farmasi","rujukan"],
    "Laboratorium": ["pendaftaran","sampling","pemeriksaan","validasi hasil","rujukan spesimen"],
    "Apotek/Farmasi": ["resep","dispensing","konseling obat","monitoring obat","stok"],
}

FHIR_RESOURCE_MAP = {
    "patient": "Patient",
    "organization": "Organization",
    "location": "Location",
    "practitioner": "Practitioner",
    "encounter": "Encounter",
    "observation": "Observation",
    "condition": "Condition",
    "procedure": "Procedure",
    "diagnostic_report": "DiagnosticReport",
    "specimen": "Specimen",
    "medication": "Medication",
    "medication_request": "MedicationRequest",
    "medication_dispense": "MedicationDispense",
    "service_request": "ServiceRequest",
    "care_plan": "CarePlan",
    "risk_assessment": "RiskAssessment",
    "document_reference": "DocumentReference",
}

def _num(s):
    return pd.to_numeric(s, errors="coerce")

def prepare_provider(df):
    d=df.copy()
    for c in ["Tanggal Sakit","Tanggal","Tanggal Kunjungan","Tanggal Pemeriksaan","Tanggal Resep"]:
        if c in d:
            d[c]=pd.to_datetime(d[c],errors="coerce")
    if "Tanggal Sakit" in d:
        d["_date"]=d["Tanggal Sakit"]
    elif "Tanggal Kunjungan" in d:
        d["_date"]=d["Tanggal Kunjungan"]
    elif "Tanggal" in d:
        d["_date"]=d["Tanggal"]
    else:
        d["_date"]=pd.NaT
    for c in ["Biaya","Tarif","Jumlah","Lama Rawat","Waktu Tunggu","Turnaround Time","Stok","Jumlah Resep"]:
        if c in d: d[c]=_num(d[c])
    return d

def provider_overview(df, provider_type="Rumah Sakit"):
    d=prepare_provider(df)
    n=len(d)
    if n==0: return {"status":"empty","message":"Belum ada data layanan fasyankes."}
    m={"observations":n,"provider_type":provider_type}
    if "Patient_ID" in d: m["unique_patients"]=int(d["Patient_ID"].nunique())
    elif "IHS_Number" in d: m["unique_patients"]=int(d["IHS_Number"].nunique())
    if "Puskesmas" in d: m["facilities"]=int(d["Puskesmas"].nunique())
    if "Diagnosis Konfirm" in d: m["diagnoses"]=int(d["Diagnosis Konfirm"].nunique())
    if "_date" in d and d["_date"].notna().any():
        m["period_start"]=str(d["_date"].min().date()); m["period_end"]=str(d["_date"].max().date())
    if "Waktu Tunggu" in d: m["mean_waiting_time"]=round(float(d["Waktu Tunggu"].mean()),2)
    if "Turnaround Time" in d: m["mean_turnaround_time"]=round(float(d["Turnaround Time"].mean()),2)
    if "Biaya" in d: m["total_cost"]=round(float(d["Biaya"].sum()),2)
    return {"status":"ok","metrics":m,"data":d}

def service_volume(df):
    d=prepare_provider(df)
    if "_date" not in d or d["_date"].isna().all(): return pd.DataFrame()
    x=d.dropna(subset=["_date"]).groupby(d["_date"].dt.to_period("D")).size().rename("Volume").reset_index()
    x["Tanggal"]=x["_date"].dt.to_timestamp()
    x["WoW_%"]=x["Volume"].pct_change(7)*100
    return x.drop(columns=["_date"])

def clinical_mix(df):
    d=prepare_provider(df)
    candidates=["Diagnosis Konfirm","Diagnosis Probabel","Diagnosis Suspek","Jenis Layanan","Service Type"]
    col=next((c for c in candidates if c in d),None)
    if not col: return pd.DataFrame()
    x=d[col].astype(str).replace({"nan":"Tidak diketahui"}).value_counts().reset_index()
    x.columns=["Kategori","Volume"]
    x["Proporsi_%"]=round(x["Volume"]/len(d)*100,2)
    return x.head(30)

def operational_segments(df):
    d=prepare_provider(df)
    candidates=[]
    for col in ["Unit","Departemen","Poliklinik","Lokasi Kerja","Puskesmas","Jenis Layanan","Service Type"]:
        if col in d:
            g=d.groupby(col,dropna=False).size().reset_index(name="Volume")
            g=g[g["Volume"]>=MIN_COHORT].sort_values("Volume",ascending=False)
            if not g.empty:
                g.insert(0,"Dimension",col); candidates.append(g)
    return pd.concat(candidates,ignore_index=True) if candidates else pd.DataFrame()

def lab_intelligence(df):
    d=prepare_provider(df)
    result={"status":"not_available","message":"Data laboratorium belum tersedia."}
    test_col=next((c for c in ["Pemeriksaan","Jenis Pemeriksaan","LOINC","Nama Pemeriksaan"] if c in d),None)
    if test_col:
        g=d[test_col].astype(str).value_counts().head(30).reset_index()
        g.columns=["Pemeriksaan","Volume"]
        result={"status":"ok","test_volume":g}
        if "Turnaround Time" in d:
            result["mean_tat"]=round(float(d["Turnaround Time"].mean()),2)
    return result

def pharmacy_intelligence(df):
    d=prepare_provider(df)
    med_col=next((c for c in ["Nama Obat","Obat","Medication","KFA"] if c in d),None)
    result={"status":"not_available","message":"Data farmasi belum tersedia."}
    if med_col:
        g=d[med_col].astype(str).value_counts().head(30).reset_index()
        g.columns=["Produk","Volume"]
        result={"status":"ok","dispensing_volume":g}
        if "Stok" in d:
            result["stock_mean"]=round(float(d["Stok"].mean()),2)
    return result

def provider_predictive_signal(df, provider_type):
    d=prepare_provider(df)
    if "_date" not in d or d["_date"].notna().sum()<30:
        return {"status":"not_available","message":"Data temporal belum mencukupi untuk predictive provider intelligence."}
    daily=d.dropna(subset=["_date"]).groupby(d["_date"].dt.date).size().astype(float)
    if len(daily)<14:
        return {"status":"not_available","message":"Minimal deret waktu 14 hari diperlukan pada prototype."}
    recent=float(daily.tail(7).mean()); previous=float(daily.iloc[-14:-7].mean()) if len(daily)>=14 else np.nan
    growth=(recent-previous)/previous*100 if previous>0 else np.nan
    return {"status":"ok","provider_type":provider_type,"recent_7d_mean":round(recent,2),
            "previous_7d_mean":round(previous,2),"growth_7d_pct":round(float(growth),2) if pd.notna(growth) else None,
            "interpretation":"Volume layanan meningkat dibanding 7 hari sebelumnya." if pd.notna(growth) and growth>0 else "Volume layanan tidak menunjukkan peningkatan pada perbandingan 7 hari."}

def build_provider_intelligence(df, provider_type="Rumah Sakit"):
    ov=provider_overview(df,provider_type)
    if ov["status"]!="ok": return ov
    return {
        "status":"ok","provider_type":provider_type,"overview":ov["metrics"],
        "service_volume":service_volume(ov["data"]),
        "clinical_mix":clinical_mix(ov["data"]),
        "operational_segments":operational_segments(ov["data"]),
        "lab":lab_intelligence(ov["data"]),
        "pharmacy":pharmacy_intelligence(ov["data"]),
        "provider_specific":provider_specific_intelligence(ov["data"], provider_type),
        "predictive":provider_predictive_signal(ov["data"],provider_type),
        "fhir_resource_map":FHIR_RESOURCE_MAP,
        "care_loop":"PATIENT → ENCOUNTER → SERVICE/DIAGNOSTIC → TREATMENT/PHARMACY → OUTCOME → ANALYSIS → PREDICTION → INTERVENTION → OUTCOME",
    }

def provider_policy_narrative(result):
    if not isinstance(result,dict) or result.get("status")!="ok":
        return "Provider Health Intelligence belum dapat dibentuk."
    p=result.get("provider_type","Fasyankes")
    return f"""### 🏥 {p.upper()} — HEALTHCARE PROVIDER INTELLIGENCE

SI-HIS memandang fasyankes sebagai **clinical-operational population system**. Data pelayanan tidak hanya digunakan untuk laporan volume, tetapi diolah menjadi intelligence mengenai beban layanan, pola penyakit, utilisasi, waktu pelayanan, laboratorium, farmasi, outcome, dan kebutuhan kapasitas.

**Descriptive intelligence** menjawab apa yang terjadi: volume kunjungan, diagnosis, jenis layanan, pemeriksaan, resep dan distribusi layanan.

**Analytical intelligence** mencari pola antar waktu, unit, jenis layanan, diagnosis, pemeriksaan dan outcome. Hasil analitik menunjukkan asosiasi/pola pada data dan tidak otomatis membuktikan hubungan sebab-akibat.

**Predictive intelligence** memperkirakan perubahan volume atau kebutuhan layanan berdasarkan data historis. Model produksi perlu menggunakan validasi temporal berulang, evaluasi error dan pemantauan perubahan pola.

**Preventive & prescriptive intelligence** mengubah sinyal menjadi dukungan keputusan: kapasitas SDM, jadwal layanan, kebutuhan laboratorium, kesiapan obat, follow-up pasien, rujukan, dan program peningkatan mutu. Rekomendasi AI merupakan DSS; keputusan klinis tetap pada tenaga kesehatan dan kebijakan fasyankes.

### 🔗 Interoperabilitas
Arsitektur dirancang mengikuti **HL7 FHIR/SATUSEHAT**. Resource seperti Patient, Encounter, Observation, Condition, DiagnosticReport, Specimen, MedicationRequest, MedicationDispense, ServiceRequest, CarePlan dan RiskAssessment dapat menjadi canonical data layer. SATUSEHAT memang menggunakan HL7 FHIR untuk model data dan API interoperabilitas. 

### 🔄 Closed-loop provider intelligence
PATIENT → ENCOUNTER → DIAGNOSTIC/TREATMENT → PHARMACY → OUTCOME → ANALYSIS → PREDICTION → INTERVENTION → NEW DATA → CONTINUOUS LEARNING

Tujuan akhirnya adalah membuat fasyankes bergerak dari **sekadar digitalisasi transaksi menuju continuous clinical & operational intelligence**."""


# Provider-specific intelligence layers
PROVIDER_INTELLIGENCE_SPEC = {
    "Rumah Sakit": {
        "domains": ["capacity", "clinical_mix", "bed_or_service_demand", "laboratory", "pharmacy", "referral", "quality_outcome"],
        "kpis": ["volume", "unique_patients", "waiting_time", "length_of_stay", "readmission", "referral", "mortality", "cost"],
    },
    "Klinik": {
        "domains": ["outpatient_demand", "clinical_mix", "waiting_time", "referral", "laboratory", "pharmacy", "follow_up"],
        "kpis": ["visit_volume", "unique_patients", "waiting_time", "referral_rate", "follow_up", "cost"],
    },
    "Laboratorium": {
        "domains": ["test_volume", "turnaround_time", "specimen", "quality", "referral", "reagent_demand"],
        "kpis": ["test_volume", "tat", "pending_results", "rejection_rate", "repeat_test", "reagent_demand"],
    },
    "Apotek/Farmasi": {
        "domains": ["prescription", "dispensing", "medication_utilization", "stock", "expiry", "refill"],
        "kpis": ["prescription_volume", "dispensing_volume", "stock_days", "stockout_signal", "expiry_signal", "refill"],
    },
}

def provider_specific_intelligence(df, provider_type):
    """Return provider-specific KPI signals without inventing unavailable fields."""
    d = prepare_provider(df)
    spec = PROVIDER_INTELLIGENCE_SPEC.get(provider_type, {})
    out = {"provider_type": provider_type, "domains": spec.get("domains", []), "kpis": spec.get("kpis", [])}
    if d.empty:
        out["status"] = "empty"
        return out
    out["status"] = "ok"

    if provider_type == "Rumah Sakit":
        out["capacity_signal"] = {
            "waiting_time_mean": round(float(d["Waktu Tunggu"].mean()), 2)
            if "Waktu Tunggu" in d else None,
            "length_of_stay_mean": round(float(d["Lama Rawat"].mean()), 2)
            if "Lama Rawat" in d else None,
        }
        out["outcome_signal"] = {
            "deaths": int(_num(d["Meninggal"]).fillna(0).sum())
            if "Meninggal" in d else None,
        }
    elif provider_type == "Klinik":
        out["access_signal"] = {
            "waiting_time_mean": round(float(d["Waktu Tunggu"].mean()), 2)
            if "Waktu Tunggu" in d else None,
            "referral_rows": int(d["Rujukan"].notna().sum())
            if "Rujukan" in d else None,
        }
    elif provider_type == "Laboratorium":
        out["laboratory_signal"] = {
            "mean_tat": round(float(d["Turnaround Time"].mean()), 2)
            if "Turnaround Time" in d else None,
            "tests": int(len(d)),
        }
    elif provider_type == "Apotek/Farmasi":
        out["pharmacy_signal"] = {
            "prescription_rows": int(len(d)),
            "mean_stock": round(float(d["Stok"].mean()), 2)
            if "Stok" in d else None,
        }
    return out
