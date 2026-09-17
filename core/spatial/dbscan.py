"""Spatial clustering and statistical hotspot analysis.

DBSCAN describes geometric density. LISA and Getis-Ord Gi* add statistical
context. Synthetic observations with unavailable/random coordinates are never
allowed into inferential spatial analysis.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

EARTH_RADIUS_KM=6371.0088


def _spatial_valid(df):
    if not {"Latitude","Longitude"}.issubset(df.columns):return pd.Series(False,index=df.index)
    valid=df["Latitude"].notna()&df["Longitude"].notna()
    if "Spatial_Quality" in df.columns:valid &= df["Spatial_Quality"].astype(str).isin({"observed","administrative_centroid"})
    return valid


def run_dbscan(df: pd.DataFrame, eps_km: float=3.0, min_samples: int=4) -> pd.DataFrame:
    work=df.copy(deep=True);work["Cluster"]=-1
    valid=_spatial_valid(work)
    if valid.sum()<min_samples:return work
    coords_rad=np.radians(work.loc[valid,["Latitude","Longitude"]].astype(float));model=DBSCAN(eps=eps_km/EARTH_RADIUS_KM,min_samples=min_samples,metric="haversine").fit(coords_rad);work.loc[valid,"Cluster"]=model.labels_;return work


def _knn_weights(coords,k=8):
    n=len(coords)
    if n<3:return np.zeros((n,n),float)
    k=min(k,n-1);tree=cKDTree(coords);_,idx=tree.query(coords,k=k+1);w=np.zeros((n,n),float)
    for i in range(n):w[i,np.atleast_1d(idx[i])[1:]]=1.0
    w=np.maximum(w,w.T);np.fill_diagonal(w,0);return w


def _permutation_p(value,sim_values,two_sided=True):
    sims=np.asarray(sim_values,float);sims=sims[np.isfinite(sims)]
    if sims.size==0:return np.nan
    return float((1+np.sum(np.abs(sims)>=abs(value)))/(sims.size+1)) if two_sided else float((1+np.sum(sims>=value))/(sims.size+1))


def spatial_statistics(df: pd.DataFrame,value_col: str="Jumlah Kasus",k: int=8,permutations: int=199,seed: int=20260917)->pd.DataFrame:
    required={"Latitude","Longitude",value_col}
    if not required.issubset(df.columns):return pd.DataFrame()
    work=df.copy();valid_mask=_spatial_valid(work);valid=work.loc[valid_mask].dropna(subset=[value_col]).copy()
    if len(valid)<4:return pd.DataFrame()
    x=pd.to_numeric(valid[value_col],errors="coerce");valid=valid.loc[x.notna()].copy();x=x.loc[valid.index].to_numpy(float)
    if len(x)<4:return pd.DataFrame()
    coords=valid[["Latitude","Longitude"]].to_numpy(float);w=_knn_weights(coords,k);row_sum=w.sum(axis=1);n=len(x);mean=x.mean();z=x-mean;ss=np.sum(z*z);global_moran=(n/w.sum()*((z[:,None]*z[None,:]*w).sum())/ss) if w.sum()>0 and ss>0 else np.nan
    lisa=np.full(n,np.nan);lisa_p=np.full(n,np.nan);lisa_cluster=np.array(["Tidak tersedia"]*n,dtype=object);gi_z=np.full(n,np.nan);gi_p=np.full(n,np.nan);hot=np.zeros(n,dtype=bool)
    if ss>0:lisa=z*(w@z)/(ss/n)
    xsum=x.sum();den=np.sqrt((n*np.sum(x*x)-xsum*xsum)/(n-1)) if n>1 else np.nan
    if np.isfinite(den) and den>0:gi_z=(w@x-row_sum*xsum/n)/(den*np.sqrt(np.maximum((n*row_sum-row_sum*row_sum)/(n-1),1e-12)))
    rng=np.random.default_rng(seed);perm=max(0,int(permutations))
    if perm:
        lisa_sims=np.empty((perm,n));gi_sims=np.empty((perm,n))
        for b in range(perm):
            xp=rng.permutation(x);zp=xp-mean;lisa_sims[b]=zp*(w@zp)/(ss/n) if ss>0 else np.nan
            gi_sims[b]=(w@xp-row_sum*xsum/n)/(den*np.sqrt(np.maximum((n*row_sum-row_sum*row_sum)/(n-1),1e-12))) if np.isfinite(den) and den>0 else np.nan
        for i in range(n):
            lisa_p[i]=_permutation_p(lisa[i],lisa_sims[:,i],True) if np.isfinite(lisa[i]) else np.nan;gi_p[i]=_permutation_p(gi_z[i],gi_sims[:,i],True) if np.isfinite(gi_z[i]) else np.nan
    local_lag=w@z
    for i in range(n):
        if not np.isfinite(lisa_p[i]) or lisa_p[i]>0.05:continue
        if z[i]>0 and local_lag[i]>0:lisa_cluster[i]="High-High"
        elif z[i]<0 and local_lag[i]<0:lisa_cluster[i]="Low-Low"
        elif z[i]>0 and local_lag[i]<0:lisa_cluster[i]="High-Low"
        elif z[i]<0 and local_lag[i]>0:lisa_cluster[i]="Low-High"
    hot=np.isfinite(gi_z)&np.isfinite(gi_p)&(gi_z>0)&(gi_p<=0.05)
    valid["LISA_I"]=lisa;valid["LISA_p"]=lisa_p;valid["LISA_cluster"]=lisa_cluster;valid["GiZ"]=gi_z;valid["Gi_p"]=gi_p;valid["Gi_Hotspot"]=hot;valid["Global_Moran_I"]=global_moran
    out=work.copy();cols=["LISA_I","LISA_p","LISA_cluster","GiZ","Gi_p","Gi_Hotspot","Global_Moran_I"]
    out["LISA_I"]=np.nan;out["LISA_p"]=np.nan;out["LISA_cluster"]="Tidak tersedia";out["GiZ"]=np.nan;out["Gi_p"]=np.nan;out["Gi_Hotspot"]=False;out["Global_Moran_I"]=np.nan;out.loc[valid.index,cols]=valid[cols];return out


def analyze_spatial(df: pd.DataFrame,eps_km: float=3.0,min_samples: int=4,k: int=8,permutations: int=199)->pd.DataFrame:
    clustered=run_dbscan(df,eps_km,min_samples)
    valid=_spatial_valid(clustered)
    if valid.sum()==0:return clustered
    points=clustered.loc[valid].groupby(["Latitude","Longitude"],as_index=False).size().rename(columns={"size":"Jumlah Kasus"})
    stats_df=spatial_statistics(points,"Jumlah Kasus",k,permutations) if not points.empty else pd.DataFrame()
    if not stats_df.empty:clustered=clustered.merge(stats_df[["Latitude","Longitude","LISA_I","LISA_p","LISA_cluster","GiZ","Gi_p","Gi_Hotspot"]],on=["Latitude","Longitude"],how="left")
    return clustered
