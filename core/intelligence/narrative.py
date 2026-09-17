"""Evidence-based narrative synthesis for SI-HIS.

Narratives distinguish signal, evidence, interpretation and uncertainty. The
module intentionally avoids causal or outbreak claims that are not supported by
the underlying method.
"""
from __future__ import annotations
import pandas as pd


def descriptive_narrative(result: dict, label: str) -> str:
    ov=result.get("overview",{}) if isinstance(result,dict) else {}
    total=int(ov.get("total_cases",0) or 0);deaths=int(ov.get("deaths",0) or 0)
    parts=[f"Scope **{label}** mencakup **{total:,} kasus/kunjungan** dan **{deaths:,} kasus meninggal**."]
    top=result.get("top10_diseases") if isinstance(result,dict) else None
    if isinstance(top,pd.DataFrame) and not top.empty:
        r=top.iloc[0];parts.append(f"Beban kasus terbesar adalah **{r.get('Nama Penyakit','-')}**, sebanyak **{int(r.get('Jumlah Kasus',0)):,} kasus** dengan CFR **{float(r.get('CFR',0)):.2f}%**.")
    parts.append("Interpretasi beban harus mempertimbangkan denominator populasi, definisi kasus, kelengkapan pencatatan dan periode observasi; CFR deskriptif tidak dengan sendirinya menunjukkan keganasan penyakit atau penyebab kematian.")
    return " ".join(parts)


def outcome_narrative(outcome: dict) -> str:
    if not isinstance(outcome,dict):return "Outcome belum tersedia."
    name=outcome.get("outcome","Outcome");question=outcome.get("question","")
    if not outcome.get("available",False):return f"**{name}:** {outcome.get('status','Analisis belum tersedia.')}"
    n=int(outcome.get("n",0));events=int(outcome.get("events",0));rate=float(outcome.get("event_rate_pct",0))
    analysis=outcome.get("analysis",{})
    sig=[]
    for factor,obj in analysis.items():
        if not isinstance(obj,dict) or factor=="Outcome":continue
        p=obj.get("p_value")
        if p is not None and pd.notna(p) and float(p)<0.05:sig.append(factor)
    if sig:
        evidence=f"Sinyal asosiasi statistik pada α=0,05 teridentifikasi pada **{', '.join(sig)}**."
    else:
        evidence="Belum ditemukan bukti asosiasi statistik pada α=0,05 pada variabel yang dapat diuji."
    return f"**Outcome {name}:** {question} Dataset analisis mencakup **{n:,} observasi**, dengan **{events:,} event ({rate:.2f}%)**. {evidence} Hasil tersebut adalah asosiasi statistik, bukan bukti kausal; besar efek, CI 95%, confounding, bias dan kualitas pengukuran harus diperiksa sebelum interpretasi epidemiologis."


def spatial_narrative(spatial: pd.DataFrame | None) -> str:
    if not isinstance(spatial,pd.DataFrame) or spatial.empty:return "Koordinat belum cukup untuk analisis spasial."
    if "Spatial_Quality" in spatial.columns:
        quality=str(spatial["Spatial_Quality"].dropna().iloc[0]) if not spatial["Spatial_Quality"].dropna().empty else "unknown"
        if quality not in {"observed","administrative_centroid"}:
            return "Koordinat tersedia tetapi sumber geometrinya belum tervalidasi sebagai koordinat administratif nyata. Analisis spasial inferensial tidak digunakan untuk menyimpulkan hotspot epidemiologis."
    clusters=int(spatial.loc[spatial.get("Cluster",pd.Series(index=spatial.index, dtype=float))!=-1,"Cluster"].nunique()) if "Cluster" in spatial else 0
    parts=[f"DBSCAN menemukan **{clusters} kelompok kepadatan geometrik**."]
    if "Gi_Hotspot" in spatial.columns:
        hot=int(pd.Series(spatial["Gi_Hotspot"]).fillna(False).astype(bool).sum());parts.append(f"Getis-Ord Gi* mengidentifikasi **{hot} lokasi dengan bukti hotspot statistik** pada ambang yang digunakan.")
    if "LISA_cluster" in spatial.columns:
        hh=int((spatial["LISA_cluster"]=="High-High").sum());parts.append(f"LISA mengidentifikasi **{hh} observasi High-High signifikan**.")
    parts.append("DBSCAN hanya menggambarkan kepadatan; status hotspot tidak ditetapkan tanpa bukti statistik spasial. Temuan spasial juga tidak membuktikan mekanisme transmisi tanpa bukti epidemiologis tambahan.")
    return " ".join(parts)


def ews_narrative(ews: dict | None, rt: dict | None = None) -> str:
    if not isinstance(ews,dict):return "Deret waktu belum memenuhi syarat untuk Early Warning."
    score=ews.get("ews_score",ews.get("score"));trend=ews.get("trend_pct",ews.get("trend"));parts=["EWS SI-HIS digunakan sebagai indikator kewaspadaan terhadap perubahan temporal, bukan probabilitas KLB terkalibrasi."]
    if isinstance(score,(int,float)):parts.append(f"Skor EWS **{float(score):.1f}**.")
    if isinstance(trend,(int,float)):parts.append(f"Perubahan tren **{float(trend):.1f}%**.")
    if isinstance(rt,dict) and isinstance(rt.get("rt_recent"),(int,float)):parts.append(f"Estimasi Rₜ terbaru **{float(rt['rt_recent']):.2f}** perlu dibaca bersama ketidakpastian dan asumsi estimator.")
    return " ".join(parts)
