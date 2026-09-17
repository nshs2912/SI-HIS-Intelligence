"""National synthetic patient-level data using real administrative references.

The observations remain entirely synthetic. Administrative labels/codes come
from a current Kemendagri-derived reference; spatial coordinates use district
administrative centroids when available. Missing geometry is kept missing and
is never replaced by random coordinates.
"""
from __future__ import annotations
from datetime import datetime,timedelta,timezone
from functools import lru_cache
import numpy as np
import pandas as pd
from .admin_reference import build_reference_locations,reference_metadata

DISEASES=["ISPA","Diare Akut","Demam Dengue","Leptospirosis","TBC","Pneumonia","Campak","Malaria","Hipertensi","Diabetes Melitus","Hepatitis","Tifoid","Penyakit Kulit","COVID-like Respiratory Illness","DBD"]
DISEASE_BASE_RATE={"ISPA":.24,"Diare Akut":.11,"Demam Dengue":.09,"Leptospirosis":.035,"TBC":.075,"Pneumonia":.06,"Campak":.035,"Malaria":.055,"Hipertensi":.075,"Diabetes Melitus":.055,"Hepatitis":.025,"Tifoid":.045,"Penyakit Kulit":.06,"COVID-like Respiratory Illness":.065,"DBD":.085}


def _island(province):
    p=str(province)
    if p in {"Dki Jakarta","Jawa Barat","Jawa Tengah","Di Yogyakarta","Jawa Timur","Banten"}:return "Java"
    if p.startswith("Sumatera") or p in {"Aceh","Kepulauan Bangka Belitung","Kepulauan Riau"}:return "Sumatra"
    if p.startswith("Kalimantan"):return "Kalimantan"
    if p.startswith("Sulawesi") or p=="Gorontalo":return "Sulawesi"
    if p in {"Bali","Nusa Tenggara Barat","Nusa Tenggara Timur"}:return "Bali-Nusa Tenggara"
    if p in {"Maluku","Maluku Utara"}:return "Maluku"
    return "Papua"


def _province_weight(province):
    return {"Jawa Barat":1.85,"Jawa Timur":1.55,"Jawa Tengah":1.45,"Dki Jakarta":1.30,"Banten":1.05,"Sumatera Utara":1.00,"Sulawesi Selatan":.78,"Sumatera Selatan":.74,"Lampung":.68,"Riau":.62,"Kalimantan Timur":.58,"Bali":.55}.get(province,.38)


def _disease_probability(disease,province,month,rng):
    p=DISEASE_BASE_RATE[disease]*{"Java":1.55,"Sumatra":1.15,"Kalimantan":.88,"Sulawesi":.82,"Bali-Nusa Tenggara":.78,"Maluku":.62,"Papua":.52}[_island(province)]
    if disease in {"DBD","Demam Dengue"}:p*=1+.55*np.sin((month-1)/12*2*np.pi)
    elif disease=="Leptospirosis":p*=1+.65*max(0,np.sin((month-2)/12*2*np.pi))
    elif disease in {"ISPA","Pneumonia","COVID-like Respiratory Illness"}:p*=1+.35*np.cos((month-1)/12*2*np.pi)
    elif disease=="Malaria":p*=2.4 if _island(province)=="Papua" else .55
    return max(.001,p*float(rng.uniform(.78,1.24)))


def _make_locations():
    loc=build_reference_locations().copy()
    if loc.empty:raise RuntimeError("Master wilayah administratif tidak tersedia.")
    loc["Province_Weight"]=loc["Nama_Provinsi"].map(_province_weight).fillna(.38)
    return loc


def _generate_cases(locations,days,target_rows,seed):
    rng=np.random.default_rng(seed);tz=timezone(timedelta(hours=7));end=datetime.now(tz).date();start=end-timedelta(days=days-1)
    loc=locations.copy();loc["_weight"]=loc["Province_Weight"]*rng.lognormal(0,.55,len(loc));loc["_weight"]/=loc["_weight"].sum();chosen=rng.choice(len(loc),size=target_rows,replace=True,p=loc["_weight"].to_numpy());selected=loc.iloc[chosen].reset_index(drop=True)
    dates=[start+timedelta(days=int(x)) for x in rng.integers(0,days,size=target_rows)];diseases=[]
    for i,row in selected.iterrows():
        w=np.array([_disease_probability(d,row["Nama_Provinsi"],dates[i].month,rng) for d in DISEASES]);w/=w.sum();diseases.append(rng.choice(DISEASES,p=w))
    diagnosis=np.asarray(diseases);age=rng.integers(1,86,target_rows);sex=rng.choice(["Laki-laki","Perempuan"],target_rows,p=[.51,.49]);occupation=rng.choice(["Petani","Nelayan","PNS/ASN","Wiraswasta","Ibu Rumah Tangga","Pelajar/Mahasiswa","Buruh","Pekerja Formal"],target_rows,p=[.12,.03,.12,.18,.17,.16,.10,.12]);comorbidity=np.where(age>=50,rng.choice(["Ada Komorbid","Tidak Ada"],target_rows,p=[.42,.58]),rng.choice(["Ada Komorbid","Tidak Ada"],target_rows,p=[.18,.82]));immunization=rng.choice(["Lengkap","Tidak Lengkap","Tidak Ada/Belum"],target_rows,p=[.62,.25,.13]);travel=rng.choice(["Ya","Tidak"],target_rows,p=[.18,.82]);other_risk=rng.choice(["Tidak Ada","Paparan Lingkungan","Kontak Erat","Air/Sanitasi","Paparan Vektor"],target_rows,p=[.45,.17,.15,.12,.11])
    confirmed=rng.random(target_rows)<np.clip(np.where(travel=="Ya",.62,.48)+np.where(comorbidity=="Ada Komorbid",.04,0),.2,.85)
    severe_prob=.015+np.where(age>=60,.08,0)+np.where(comorbidity=="Ada Komorbid",.07,0)+np.where(np.isin(diagnosis,["Malaria","Pneumonia","TBC"]),.05,0)+np.where(np.isin(diagnosis,["DBD","Demam Dengue"]),.02,0);severe=rng.random(target_rows)<np.clip(severe_prob,.005,.45)
    death_prob=.01+np.where(severe,.10,0)+np.where(age>=65,.035,0)+np.where(comorbidity=="Ada Komorbid",.025,0);deaths=rng.random(target_rows)<np.clip(death_prob,.002,.25);status=np.where(deaths,"Meninggal",np.where(severe,"Berat",np.where(confirmed,"Sembuh/Rawat","Dalam Pemantauan")))
    outbreak_mask=selected["Kode_Kabupaten"].astype(str).str.endswith(("03","07"),na=False).to_numpy()&np.isin(diagnosis,["DBD","Demam Dengue","Leptospirosis"]);pulse=rng.random(target_rows)<np.where(outbreak_mask,.22,.015)
    return pd.DataFrame({"Nama":[f"National_Dummy_{i:07d}" for i in range(1,target_rows+1)],"Umur":age,"Jenis Kelamin":sex,"Pekerjaan":occupation,"Status Imunisasi":immunization,"Status Komorbid":comorbidity,"Riwayat Perjalanan":travel,"Faktor Risiko Lain":other_risk,"Tanggal Sakit":pd.to_datetime(dates).strftime("%Y-%m-%d"),"Kode_Provinsi":selected["Kode_Provinsi"].to_numpy(),"Provinsi":selected["Nama_Provinsi"].to_numpy(),"Kode_Kabupaten":selected["Kode_Kabupaten"].to_numpy(),"Kabupaten":selected["Nama_Kabupaten"].to_numpy(),"Kode_Kecamatan":selected["Kode_Kecamatan"].to_numpy(),"Kecamatan":selected["Nama_Kecamatan"].to_numpy(),"Kode_Desa":selected["Kode_Desa"].to_numpy(),"Desa/Kelurahan":selected["Nama_Desa"].to_numpy(),"Puskesmas":[f"Puskesmas Sintetik {x}" for x in selected["Kode_Kecamatan"].astype(str)],"Latitude":pd.to_numeric(selected["Latitude"],errors="coerce").to_numpy(),"Longitude":pd.to_numeric(selected["Longitude"],errors="coerce").to_numpy(),"Spatial_Quality":selected["Spatial_Quality"].to_numpy(),"Diagnosis Suspek":diagnosis,"Diagnosis Probabel":np.where(confirmed,diagnosis,"Bukan"),"Diagnosis Konfirm":np.where(confirmed,diagnosis,"Bukan"),"Is_Konfirm":confirmed.astype(int),"Severity":np.where(severe,"Berat","Tidak Berat"),"Is_Severe":severe.astype(int),"Is_Meninggal":deaths.astype(int),"Status Penderita":status,"Synthetic_Outbreak_Pulse":pulse.astype(float)})


@lru_cache(maxsize=2)
def generate_national_dummy(days:int=365,target_rows:int=60000,seed:int=20260917)->pd.DataFrame:
    return _generate_cases(_make_locations(),int(days),int(target_rows),int(seed))


def national_metadata()->dict:
    return {**reference_metadata(),"dataset_type":"synthetic_national","provinces":38,"districts":514,"diseases":len(DISEASES),"default_days":365,"default_cases":60000,"source":"SI-HIS synthetic generator + administrative reference","real_patient_data":False}
