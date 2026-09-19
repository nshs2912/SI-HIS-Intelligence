"""Canonical SI-HIS intelligence engine.

The engine separates descriptive analysis from disease-scoped epidemiology and
keeps each outcome independent: disease occurrence, severity and mortality are
modeled as separate dependent variables. KLB early warning uses a versioned
regulatory surveillance ruleset and never declares legal KLB status itself.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import OrderedDict
import hashlib
import copy
from typing import Any
import numpy as np
import pandas as pd
from .analytics import deteksi_bentuk_kurva, deteksi_gelombang, hitung_effective_rt, identifikasi_vulnerable_profile
from .epidemiology import analyze_mortality, analyze_risk, analyze_trias
from .epidemiology.pipeline import classify_temporal_pattern_for_disease, resolve_disease_profile, validate_scope_for_special_analysis
from .disease_intelligence import classify_disease, build_infectious_intelligence
from .acute_event_intelligence import poisoning_vs_disaster_signal, toxicology_differential
from .intelligence_orchestrator import orchestrate_intelligence
from .incident_reasoning import incident_reasoning_v2
from .forecasting import holt_winters_forecast
from .ml_engine import (train_case_severity, train_klb_prediction, train_spatial_outbreak, train_vulnerable_population, robust_forecast, temporal_anomaly_detection, temporal_change_points, growth_risk_prediction, spatial_neighbor_intelligence, vulnerability_clustering, prioritize_continuous_signals, spatiotemporal_risk, time_person_place_risk, disease_specific_growth_model)
from .scope import QueryScope, apply_scope, scope_label
from .spatial import compute_epicenter, analyze_spatial
from .statistics import hitung_bivariat_lengkap
from .surveillance import early_warning, evaluate_klb

@dataclass(frozen=True)
class IntelligenceResult:
    scope: QueryScope
    dataframe: pd.DataFrame
    provenance: dict[str, Any]

DISEASE_COLUMNS=["Diagnosis Konfirm","Diagnosis Probabel","Diagnosis Suspek"]
NON_DISEASE_VALUES={"","nan","none","bukan","tidak ada","-"}


def _disease_per_case(df):
    if df.empty:return pd.Series(index=df.index,dtype="object")
    result=pd.Series("Tidak Teridentifikasi",index=df.index,dtype="object")
    for column in reversed(DISEASE_COLUMNS):
        if column not in df.columns:continue
        values=df[column].astype(str).str.strip();valid=~values.str.lower().isin(NON_DISEASE_VALUES);result.loc[valid]=values.loc[valid]
    return result


def _filter_disease(df,disease):
    if not disease or disease=="Semua Penyakit":return df.copy()
    mask=pd.Series(False,index=df.index)
    for column in DISEASE_COLUMNS:
        if column in df.columns:mask|=df[column].astype(str).str.strip().eq(disease)
    return df.loc[mask].copy()


def _apply_period(df,period_days):
    if df.empty or "Tanggal Sakit" not in df.columns:return df.copy()
    work=df.copy();work["Tanggal Sakit"]=pd.to_datetime(work["Tanggal Sakit"],errors="coerce");dates=work["Tanggal Sakit"].dropna()
    if dates.empty:return work.iloc[0:0].copy()
    end=dates.max().normalize();start=end-pd.Timedelta(days=max(1,int(period_days))-1)
    return work.loc[work["Tanggal Sakit"].between(start,end)].copy()


def _death_series(df):
    death=pd.Series(0.0,index=df.index)
    if "Is_Meninggal" in df.columns:death=pd.to_numeric(df["Is_Meninggal"],errors="coerce").fillna(0).astype(float)
    if "Status Penderita" in df.columns:
        status=df["Status Penderita"].astype(str).str.strip().str.lower().eq("meninggal").astype(float);death=pd.Series(np.maximum(death,status),index=df.index)
    return death


def _top10_diseases(df):
    columns=["NO.","Nama Penyakit","Jumlah Kasus","CFR","Kabupaten","Provinsi"]
    if df.empty:return pd.DataFrame(columns=columns)
    work=df.copy();work["_disease"]=_disease_per_case(work);work["_death"]=_death_series(work);rows=[]
    for disease,group in work.groupby("_disease",dropna=False):
        if not disease or str(disease).lower() in NON_DISEASE_VALUES or disease=="Tidak Teridentifikasi":continue
        n=len(group);deaths=int(group["_death"].sum());district="-";province="-"
        if "Kabupaten" in group:
            counts=group["Kabupaten"].value_counts(dropna=True)
            if not counts.empty:
                district=counts.index[0]
                if "Provinsi" in group:
                    p=group.loc[group["Kabupaten"].eq(district),"Provinsi"].mode()
                    if not p.empty:province=p.iloc[0]
        rows.append({"Nama Penyakit":str(disease),"Jumlah Kasus":n,"CFR":round(deaths/n*100,2) if n else 0.0,"Kabupaten":district,"Provinsi":province})
    out=pd.DataFrame(rows)
    if out.empty:return pd.DataFrame(columns=columns)
    out=out.sort_values(["Jumlah Kasus","Nama Penyakit"],ascending=[False,True]).head(10).reset_index(drop=True);out.insert(0,"NO.",np.arange(1,len(out)+1));return out[columns]


def _trias_summary(df,national_disease_df,disease,geographic_scope):
    total=len(df);national_total=len(national_disease_df);summary=[];top10=pd.DataFrame()
    if "Provinsi" in df.columns and total:
        g=df.groupby("Provinsi",dropna=False).agg(Kasus=("Provinsi","size"),Meninggal=("_death","sum")).reset_index();g["CFR"]=np.where(g["Kasus"]>0,g["Meninggal"]/g["Kasus"]*100,0)
        top_cases=g.sort_values(["Kasus","Provinsi"],ascending=[False,True]).iloc[0];top_cfr=g.sort_values(["CFR","Kasus"],ascending=[False,False]).iloc[0];denominator=national_total if geographic_scope=="Indonesia" else total;share=top_cases["Kasus"]/denominator*100 if denominator else 0
        if geographic_scope=="Indonesia":summary.append(f"Kasus **{disease}** terbanyak berada di **{top_cases['Provinsi']}**, sebanyak **{int(top_cases['Kasus']):,} kasus** ({share:.2f}% dari seluruh kasus {disease} di Indonesia) dengan CFR **{top_cases['CFR']:.2f}%**.")
        else:summary.append(f"Dalam scope **{geographic_scope}**, kasus **{disease}** terbanyak berada di **{top_cases['Provinsi']}**, sebanyak **{int(top_cases['Kasus']):,} kasus** ({share:.2f}% dari kasus pada scope) dengan CFR **{top_cases['CFR']:.2f}%**.")
        if str(top_cfr["Provinsi"])!=str(top_cases["Provinsi"]):summary.append(f"CFR tertinggi bukan berada di wilayah dengan kasus terbanyak, melainkan di **{top_cfr['Provinsi']}**, sebesar **{top_cfr['CFR']:.2f}%** ({int(top_cfr['Meninggal'])} meninggal dari {int(top_cfr['Kasus'])} kasus).")
        else:summary.append(f"Wilayah dengan kasus terbanyak juga memiliki CFR tertinggi, yaitu **{top_cfr['CFR']:.2f}%**.")
        if geographic_scope=="Indonesia":
            top10=g.sort_values(["Kasus","Provinsi"],ascending=[False,True]).head(10)[["Provinsi","Kasus","CFR"]].reset_index(drop=True);top10.insert(0,"NO.",np.arange(1,len(top10)+1))
    if "Umur" in df.columns and total:
        age=pd.to_numeric(df["Umur"],errors="coerce");bins=pd.cut(age,bins=[-np.inf,1,5,10,15,20,25,35,45,55,65,75,85,np.inf],labels=["<1 tahun","1-4 tahun","5-9 tahun","10-14 tahun","15-19 tahun","20-24 tahun","25-34 tahun","35-44 tahun","45-54 tahun","55-64 tahun","65-74 tahun","75-84 tahun","≥85 tahun"],right=False);ag=bins.value_counts(sort=False,dropna=True)
        if not ag.empty:a=ag.idxmax();n=int(ag.max());summary.append(f"Distribusi umur terbesar terdapat pada kelompok **{a}**, sebanyak **{n:,} kasus ({n/total*100:.2f}% dari seluruh kasus {disease} pada scope analisis)**.")
    if "Jenis Kelamin" in df.columns and total:
        sex=df["Jenis Kelamin"].astype(str).value_counts();parts=[f"**{idx}: {int(val):,} kasus ({val/total*100:.2f}%)**" for idx,val in sex.items()];
        if parts:summary.append("Distribusi jenis kelamin menunjukkan "+", ".join(parts)+".")
    return {"narrative":" ".join(summary),"top10_province":top10,"total_scope_cases":total,"national_disease_cases":national_total,"scope":geographic_scope}


def _binary_outcome_from_disease(df,disease):
    """Disease outcome on the full scoped population: 1=target disease, 0=other/non-target."""
    return _disease_per_case(df).eq(disease).astype(int)


def _severity_outcome(df):
    if "Is_Severe" in df.columns:return pd.to_numeric(df["Is_Severe"],errors="coerce")
    if "Severity" in df.columns:return df["Severity"].astype(str).str.strip().str.lower().isin({"berat","severe","critical","kritis"}).astype(int)
    return None


def _outcome_analysis(cohort,target,name,question):
    if target is None:return {"available":False,"outcome":name,"question":question,"status":"Outcome belum tersedia/terdefinisi pada dataset."}
    work=cohort.copy();work["_Outcome"]=pd.to_numeric(target,errors="coerce");valid=work["_Outcome"].isin([0,1]);work=work.loc[valid].copy()
    if len(work)<30 or work["_Outcome"].nunique()<2:return {"available":False,"outcome":name,"question":question,"status":f"Outcome {name} belum memenuhi kecukupan data untuk model (minimal 30 observasi lengkap dan dua kategori outcome).","n":len(work)}
    result=hitung_bivariat_lengkap(work,var_dep_binary="_Outcome")
    return {"available":True,"outcome":name,"question":question,"n":len(work),"events":int(work["_Outcome"].sum()),"event_rate_pct":round(float(work["_Outcome"].mean()*100),2),"analysis":result}


_ML_RESULT_CACHE = OrderedDict()
_ML_CACHE_MAXSIZE = 8

def _ml_cache_key(df):
    """Create a content-based cache key without changing the analytical dataset."""
    cols = list(df.columns)
    h = hashlib.sha256()
    h.update("|".join(map(str, cols)).encode("utf-8"))
    h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()

class IntelligenceEngine:
    def prepare(self,df,scope=None):
        query_scope=(scope or QueryScope()).normalized();scoped=apply_scope(df,query_scope);return IntelligenceResult(query_scope,scoped,{"engine":"SI-HIS Intelligence","scope":query_scope.to_dict(),"scope_isolated":True,"source_rows":len(df),"scoped_rows":len(scoped),"area":scope_label(query_scope)})

    def descriptive(self,df):
        work=df.copy(deep=True);work["_disease"]=_disease_per_case(work);total=len(work);deaths=int(_death_series(work).sum());sex=work["Jenis Kelamin"].value_counts(dropna=False).rename_axis("Jenis Kelamin").reset_index(name="Jumlah Kasus") if "Jenis Kelamin" in work else pd.DataFrame();age=pd.to_numeric(work["Umur"],errors="coerce") if "Umur" in work else pd.Series(dtype=float);bins=pd.cut(age,bins=[-np.inf,1,5,10,15,20,25,35,45,55,65,75,85,np.inf],labels=["<1","1-4","5-9","10-14","15-19","20-24","25-34","35-44","45-54","55-64","65-74","75-84","≥85"],right=False);age_table=bins.value_counts(sort=False,dropna=False).rename_axis("Kelompok Umur").reset_index(name="Jumlah Kasus");province=work.groupby("Provinsi",dropna=False).size().reset_index(name="Jumlah Kasus").sort_values("Jumlah Kasus",ascending=False) if "Provinsi" in work else pd.DataFrame();district=work.groupby(["Provinsi","Kabupaten"],dropna=False).size().reset_index(name="Jumlah Kasus").sort_values("Jumlah Kasus",ascending=False) if "Kabupaten" in work else pd.DataFrame();disease_top=_top10_diseases(work);narrative=[]
        if total:narrative.append(f"Scope analisis mencakup **{total:,} kasus/kunjungan** dengan **{deaths:,} kasus meninggal**.")
        if not disease_top.empty:
            r=disease_top.iloc[0];narrative.append(f"Penyakit dengan beban kasus terbesar adalah **{r['Nama Penyakit']}**, sebanyak **{int(r['Jumlah Kasus']):,} kasus**, dengan CFR **{float(r['CFR']):.2f}%**.")
        if not province.empty:
            r=province.iloc[0];narrative.append(f"Wilayah dengan jumlah kasus terbesar adalah **{r['Provinsi']}**, sebanyak **{int(r['Jumlah Kasus']):,} kasus**.")
        narrative.append("Distribusi ini merupakan gambaran deskriptif; outcome-specific analysis dijalankan hanya ketika penyakit dan prasyarat metodologis tersedia.")
        return {"mode":"descriptive","overview":{"total_cases":total,"deaths":deaths},"top10_diseases":disease_top,"disease_distribution":work["_disease"].value_counts().rename_axis("Nama Penyakit").reset_index(name="Jumlah Kasus"),"province_distribution":province,"district_distribution":district,"sex_distribution":sex,"age_distribution":age_table,"narrative":" ".join(narrative),"provenance":{"engine":"SI-HIS Intelligence","analysis":"descriptive_intelligence","scope_required":False}}

    def analyze(self,df,scope=None,forecast_days=14,include_ml=False,mode="epidemiology"):
        if mode=="descriptive":return self.descriptive(df)
        prepared=self.prepare(df,scope);scoped=_apply_period(prepared.dataframe,prepared.scope.period_days);disease=prepared.scope.disease
        if not disease or disease=="Semua Penyakit":
            return {
                "mode": "epidemiology",
                "eligible": False,
                "message": "Analisis epidemiologi disease-specific memerlukan satu penyakit. Gunakan mode descriptive untuk Semua Penyakit.",
                "scope": prepared.scope.to_dict(),
                "overview": {"total_cases": len(prepared.dataframe), "deaths": int(_death_series(prepared.dataframe).sum())},
                "analysis_sections": {
                    "descriptive": self.descriptive(prepared.dataframe),
                    "person": None, "place": None, "time": None, "spatial": None,
                    "forecast": None, "risk": None, "surveillance": None, "machine_learning": None,
                },
                "provenance": {**prepared.provenance, "analysis_mode": "descriptive_only", "reason": "all_disease_or_missing_disease"},
            }
        work=_filter_disease(scoped,disease)
        if prepared.scope.puskesmas:geographic_level="Puskesmas"
        elif prepared.scope.village:geographic_level="Desa/Kelurahan"
        elif prepared.scope.kecamatan:geographic_level="Kecamatan"
        elif prepared.scope.district:geographic_level="Kabupaten"
        elif prepared.scope.province:geographic_level="Provinsi"
        else:geographic_level="Indonesia"
        eligibility=validate_scope_for_special_analysis(work,disease,geographic_level,10,14);profile=resolve_disease_profile(disease);disease_intel=classify_disease(disease);infectious_intel=build_infectious_intelligence(work,disease);acute_event_intel=poisoning_vs_disaster_signal(work)
        incident_v2=incident_reasoning_v2(work)
        # For communicable diseases, onset is the epidemiological clock when it is available.
        # The original service/examination date is preserved for data-quality review.
        if disease_intel.family=="MENULAR" and "Tanggal Onset" in work.columns:
            onset=pd.to_datetime(work["Tanggal Onset"],errors="coerce")
            if onset.notna().any():
                work=work.copy();work["Tanggal Pemeriksaan Asli"]=work.get("Tanggal Sakit");work["Tanggal Sakit"]=onset
        common={"mode":"epidemiology","eligible":eligibility.eligible,"eligibility":{"reasons":eligibility.reasons,"warnings":eligibility.warnings},"scope":prepared.scope.to_dict(),"overview":{"total_cases":len(work),"provinces":work["Provinsi"].nunique() if "Provinsi" in work else 0,"districts":work["Kabupaten"].nunique() if "Kabupaten" in work else 0,"subdistricts":work["Kecamatan"].nunique() if "Kecamatan" in work else 0,"villages":work["Desa/Kelurahan"].nunique() if "Desa/Kelurahan" in work else 0,"deaths":int(_death_series(work).sum()),"cfr":round(float(_death_series(work).mean()*100),2) if len(work) else 0.0},"disease_profile":profile.__dict__,"disease_intelligence":disease_intel.__dict__,"infectious_intelligence":infectious_intel,"acute_event_intelligence":acute_event_intel,"toxicology_intelligence":toxicology_intel,"intelligence_orchestration":orchestrate_intelligence(work,disease,acute_event_intel,infectious_intel,None),"incident_reasoning_v2":incident_v2,"analysis_sections":{"descriptive":None,"person":None,"place":None,"time":None,"spatial":None,"forecast":None,"risk":None,"surveillance":"klb","machine_learning":None},"provenance":{**prepared.provenance,"analysis_mode":"disease_scoped_epidemiology","ml_enabled":bool(include_ml)}}
        # KLB surveillance is evaluated even when the special statistical scope is
        # not eligible, because early detection and data sufficiency are separate.
        common["klb"] = evaluate_klb(work, disease=profile.name, geographic_scope=geographic_level)
        if not eligibility.eligible:common["message"]="Analisis belum dijalankan karena data/scope belum memenuhi syarat metodologis.";return common
        work=work.copy();work["_death"]=_death_series(work);trias=analyze_trias(work);epi=trias["time"];waves=deteksi_gelombang(epi) if not epi.empty else []
        try:curve=deteksi_bentuk_kurva(epi,profile.name) if len(epi)>=7 else None
        except Exception:curve=None
        spatial=analyze_spatial(work);national_disease_df=_filter_disease(scoped,disease);trias_summary=_trias_summary(work,national_disease_df,profile.name,geographic_level);person=trias["person"].copy();person["Resume TIME + PERSON + PLACE"]=trias_summary["narrative"]
        if not trias_summary["top10_province"].empty:person["10 Besar Wilayah — Provinsi"]=trias_summary["top10_province"]
        disease_cohort=scoped.copy();disease_target=_binary_outcome_from_disease(disease_cohort,disease);severity_target=_severity_outcome(work);mortality_target=_death_series(work)
        outcome_analyses={"Penyakit":_outcome_analysis(disease_cohort,disease_target,"Penyakit",f"Faktor yang berasosiasi dengan kejadian {disease} pada seluruh kasus dalam scope."),"Severity":_outcome_analysis(work,severity_target,"Severity",f"Faktor yang berasosiasi dengan severity pada kasus {disease}."),"Kasus Meninggal":_outcome_analysis(work,mortality_target,"Kasus Meninggal",f"Faktor yang berasosiasi dengan kematian pada kasus {disease}.")}
        common.update({"person":person,"place":trias["place"],"trias_summary":trias_summary,"time":epi,"analysis_dataframe":work.copy(deep=True),"mortality":analyze_mortality(work),"risk":analyze_risk(work),"risk_factors":outcome_analyses["Penyakit"],"outcome_analyses":outcome_analyses,"vulnerable":identifikasi_vulnerable_profile(work),"ews":early_warning(epi),"rt":hitung_effective_rt(epi) if not epi.empty else None,"waves":waves,"forecast":holt_winters_forecast(epi,forecast_days),"spatial":spatial,"epicenters":compute_epicenter(spatial),"temporal_interpretation":classify_temporal_pattern_for_disease(profile,len(waves)),"epidemic_curve_classification":curve,"ml":self.ml_train(work,disease) if include_ml else {"enabled":False,"message":"ML layer tidak dijalankan."}})
        common["analysis_sections"].update({"descriptive":None,"person":person,"place":trias["place"],"time":epi,"spatial":spatial,"forecast":common["forecast"],"risk":common["risk"],"surveillance":{"klb":common["klb"],"ews":common["ews"],"rt":common["rt"]},"machine_learning":common["ml"]})
        return common

    def ml_train(self,df,disease=None):
        # Performance optimization only: identical analytical input reuses the
        # exact same ML result. No rows, fields, algorithms, targets, or metrics
        # are changed. A new/changed dataset produces a new cache key and is
        # recomputed automatically.
        cache_key=_ml_cache_key(df) + "|" + str(disease or "")
        cached=_ML_RESULT_CACHE.get(cache_key)
        if cached is not None:
            _ML_RESULT_CACHE.move_to_end(cache_key)
            return copy.deepcopy(cached)

        daily=analyze_trias(df)["time"]
        anomaly=temporal_anomaly_detection(df)
        changes=temporal_change_points(df)
        growth=growth_risk_prediction(df)
        neighbor=spatial_neighbor_intelligence(df)
        vulnerability=vulnerability_clustering(df)
        spatiotemporal=spatiotemporal_risk(df)
        tpp=time_person_place_risk(df)
        disease_growth=disease_specific_growth_model(df)
        continuous=prioritize_continuous_signals(anomaly,changes,growth,neighbor)
        severity=train_case_severity(df)
        klb=train_klb_prediction(df)
        spatial=train_spatial_outbreak(df)
        vulnerable=train_vulnerable_population(df)
        forecast=robust_forecast(daily,14)
        result={
            "primary_engines": {"case_severity":severity,"klb":klb,"spatial":spatial,"vulnerable":vulnerable,"forecast":forecast},
            "case_severity":severity,"klb":klb,"spatial":spatial,"vulnerable":vulnerable,"forecast":forecast,
            "continuous_intelligence": {"temporal_anomaly":anomaly,"change_points":changes,"growth_risk":growth,"spatial_neighbor":neighbor,"vulnerability_clustering":vulnerability,"spatiotemporal_risk":spatiotemporal,"time_person_place_risk":tpp,"disease_specific_growth":disease_growth,"signal_prioritization":continuous},
            "disease_intelligence": classify_disease(disease).__dict__ if disease else None,"infectious_intelligence": build_infectious_intelligence(df,disease) if disease else None,"acute_event_intelligence": poisoning_vs_disaster_signal(df), "toxicology_intelligence": toxicology_differential(df),
            "intelligence_orchestration": orchestrate_intelligence(df,disease,poisoning_vs_disaster_signal(df),build_infectious_intelligence(df,disease) if disease else None,{"primary_engines":{"case_severity":severity,"klb":klb,"spatial":spatial,"vulnerable":vulnerable,"forecast":forecast}}),
            "architecture": {"primary_engine_count":5,"supporting_signal_modules":9,"loop":"DATA → ANALYSIS → PREDICTION → PRESCRIPTION → NEW DATA → CONTINUOUS LEARNING"}
        }
        _ML_RESULT_CACHE[cache_key]=copy.deepcopy(result)
        _ML_RESULT_CACHE.move_to_end(cache_key)
        while len(_ML_RESULT_CACHE)>_ML_CACHE_MAXSIZE:
            _ML_RESULT_CACHE.popitem(last=False)
        return copy.deepcopy(result)

SIHISIntelligenceEngine=IntelligenceEngine
