"""Administrative reference loader for synthetic SI-HIS data.

The generator must not invent administrative names. This adapter uses a current
38-province/514-regency reference published from Kemendagri-derived data and a
separate district-centroid reference for spatial QA. Production deployments
should replace these URLs with the organization's governed master data.
"""
from __future__ import annotations
from io import StringIO
from functools import lru_cache
import json
import urllib.request
import pandas as pd

ADMIN_BASE="https://raw.githubusercontent.com/awanaprilino/wilayah-administrasi-kemendagri/main/csv"
REGENCY_URL=f"{ADMIN_BASE}/regencies.csv"
DISTRICT_URL=f"{ADMIN_BASE}/districts.csv"
VILLAGE_URL=f"{ADMIN_BASE}/villages.csv"
CENTROID_URL="https://raw.githubusercontent.com/quarcs-lab/indonesia514/main/maps/mapIndonesia514_new_points.geojson"


def _read_url(url: str, timeout: int = 30) -> str:
    req=urllib.request.Request(url,headers={"User-Agent":"SI-HIS-Intelligence/1.0"})
    with urllib.request.urlopen(req,timeout=timeout) as response:return response.read().decode("utf-8")


@lru_cache(maxsize=1)
def load_regencies() -> pd.DataFrame:
    raw=_read_url(REGISTRY_URL if False else REGENCY_URL)
    d=pd.read_csv(StringIO(raw),header=None,names=["Kode_Kabupaten","Kode_Provinsi","Nama_Kabupaten"])
    d["Kode_Kabupaten"]=d["Kode_Kabupaten"].astype(str).str.zfill(4);d["Kode_Provinsi"]=d["Kode_Provinsi"].astype(str).str.zfill(2)
    d["Nama_Kabupaten"]=d["Nama_Kabupaten"].astype(str).str.replace(r"^KABUPATEN\s+|^KOTA\s+","",regex=True).str.replace(r"\s+"," ",regex=True).str.strip().str.title()
    return d.drop_duplicates("Kode_Kabupaten").reset_index(drop=True)


@lru_cache(maxsize=1)
def load_districts() -> pd.DataFrame:
    raw=_read_url(DISTRICT_URL)
    d=pd.read_csv(StringIO(raw),header=None,names=["Kode_Kecamatan","Kode_Kabupaten","Nama_Kecamatan"])
    d["Kode_Kecamatan"]=d["Kode_Kecamatan"].astype(str).str.zfill(6);d["Kode_Kabupaten"]=d["Kode_Kabupaten"].astype(str).str.zfill(4)
    d["Nama_Kecamatan"]=d["Nama_Kecamatan"].astype(str).str.replace(r"\s+"," ",regex=True).str.strip().str.title()
    return d.drop_duplicates("Kode_Kecamatan").reset_index(drop=True)


@lru_cache(maxsize=1)
def load_villages() -> pd.DataFrame:
    raw=_read_url(VILLAGE_URL)
    d=pd.read_csv(StringIO(raw),header=None,names=["Kode_Desa","Kode_Kecamatan","Nama_Desa"])
    d["Kode_Desa"]=d["Kode_Desa"].astype(str).str.zfill(10);d["Kode_Kecamatan"]=d["Kode_Kecamatan"].astype(str).str.zfill(6)
    d["Nama_Desa"]=d["Nama_Desa"].astype(str).str.replace(r"\s+"," ",regex=True).str.strip().str.title()
    return d.drop_duplicates("Kode_Desa").reset_index(drop=True)


@lru_cache(maxsize=1)
def load_district_centroids() -> pd.DataFrame:
    data=json.loads(_read_url(CENTROID_URL));rows=[]
    for feature in data.get("features",[]):
        p=feature.get("properties",{});rows.append({"Kode_Kabupaten":str(p.get("districtID","")).zfill(4),"Latitude":p.get("latitude"),"Longitude":p.get("longitude"),"Spatial_Quality":"administrative_centroid"})
    d=pd.DataFrame(rows);return d.drop_duplicates("Kode_Kabupaten") if not d.empty else d


def reference_metadata() -> dict:
    return {"administrative_source":"Kemendagri-derived 2025 reference adapter","spatial_reference":"514 district administrative centroids","real_patient_data":False,"note":"Synthetic observations are joined to real administrative labels. Spatial inference requires validated geometry/centroid coverage; missing centroids are not replaced with random coordinates."}
