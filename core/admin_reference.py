"""Administrative reference loader for synthetic SI-HIS data.

Administrative labels are not invented by the synthetic generator. The loader
uses a current Kemendagri-derived reference and a district-centroid reference.
Production deployments should replace the URLs with governed master data.
"""
from __future__ import annotations
from io import StringIO
from functools import lru_cache
import json
import urllib.request
import pandas as pd

ADMIN_BASE="https://raw.githubusercontent.com/awanaprilino/wilayah-administrasi-kemendagri/main/csv"
PROVINCE_URL=f"{ADMIN_BASE}/provinces.csv"
REGENCY_URL=f"{ADMIN_BASE}/regencies.csv"
DISTRICT_URL=f"{ADMIN_BASE}/districts.csv"
VILLAGE_URL=f"{ADMIN_BASE}/villages.csv"
CENTROID_URL="https://raw.githubusercontent.com/quarcs-lab/indonesia514/main/maps/mapIndonesia514_new_points.geojson"


def _read_url(url: str, timeout: int = 30) -> str:
    req=urllib.request.Request(url,headers={"User-Agent":"SI-HIS-Intelligence/1.0"})
    with urllib.request.urlopen(req,timeout=timeout) as response:return response.read().decode("utf-8")


def _clean_name(series):
    return series.astype(str).str.replace(r"\s+"," ",regex=True).str.strip().str.title()


@lru_cache(maxsize=1)
def load_provinces() -> pd.DataFrame:
    raw=_read_url(PROVINCE_URL);d=pd.read_csv(StringIO(raw),header=None,names=["Kode_Provinsi","Nama_Provinsi"])
    d["Kode_Provinsi"]=d["Kode_Provinsi"].astype(str).str.zfill(2);d["Nama_Provinsi"]=_clean_name(d["Nama_Provinsi"])
    return d.drop_duplicates("Kode_Provinsi").reset_index(drop=True)


@lru_cache(maxsize=1)
def load_regencies() -> pd.DataFrame:
    raw=_read_url(REGENCY_URL);d=pd.read_csv(StringIO(raw),header=None,names=["Kode_Kabupaten","Kode_Provinsi","Nama_Kabupaten"])
    d["Kode_Kabupaten"]=d["Kode_Kabupaten"].astype(str).str.zfill(4);d["Kode_Provinsi"]=d["Kode_Provinsi"].astype(str).str.zfill(2)
    d["Nama_Kabupaten"]=_clean_name(d["Nama_Kabupaten"].str.replace(r"^KABUPATEN\s+|^KOTA\s+","",regex=True))
    return d.drop_duplicates("Kode_Kabupaten").reset_index(drop=True)


@lru_cache(maxsize=1)
def load_districts() -> pd.DataFrame:
    raw=_read_url(DISTRICT_URL);d=pd.read_csv(StringIO(raw),header=None,names=["Kode_Kecamatan","Kode_Kabupaten","Nama_Kecamatan"])
    d["Kode_Kecamatan"]=d["Kode_Kecamatan"].astype(str).str.zfill(6);d["Kode_Kabupaten"]=d["Kode_Kabupaten"].astype(str).str.zfill(4);d["Nama_Kecamatan"]=_clean_name(d["Nama_Kecamatan"])
    return d.drop_duplicates("Kode_Kecamatan").reset_index(drop=True)


@lru_cache(maxsize=1)
def load_villages() -> pd.DataFrame:
    raw=_read_url(VILLAGE_URL);d=pd.read_csv(StringIO(raw),header=None,names=["Kode_Desa","Kode_Kecamatan","Nama_Desa"])
    d["Kode_Desa"]=d["Kode_Desa"].astype(str).str.zfill(10);d["Kode_Kecamatan"]=d["Kode_Kecamatan"].astype(str).str.zfill(6);d["Nama_Desa"]=_clean_name(d["Nama_Desa"])
    return d.drop_duplicates("Kode_Desa").reset_index(drop=True)


@lru_cache(maxsize=1)
def load_district_centroids() -> pd.DataFrame:
    data=json.loads(_read_url(CENTROID_URL));rows=[]
    for feature in data.get("features",[]):
        p=feature.get("properties",{});rows.append({"Kode_Kabupaten":str(p.get("districtID","")).zfill(4),"Latitude":p.get("latitude"),"Longitude":p.get("longitude"),"Spatial_Quality":"administrative_centroid"})
    d=pd.DataFrame(rows);return d.drop_duplicates("Kode_Kabupaten") if not d.empty else d


def build_reference_locations() -> pd.DataFrame:
    provinces=load_provinces();regencies=load_regencies();districts=load_districts();villages=load_villages();centroids=load_district_centroids()
    d=districts.merge(regencies,on="Kode_Kabupaten",how="inner").merge(provinces,on="Kode_Provinsi",how="left").merge(villages,on="Kode_Kecamatan",how="inner").merge(centroids,on="Kode_Kabupaten",how="left")
    d["Spatial_Quality"]=d["Spatial_Quality"].fillna("unavailable")
    return d


def reference_metadata() -> dict:
    return {"administrative_source":"Kemendagri-derived current administrative reference","spatial_reference":"district administrative centroids where available","real_patient_data":False,"note":"Synthetic observations use real administrative names/codes. Missing spatial reference is not replaced with random coordinates."}
