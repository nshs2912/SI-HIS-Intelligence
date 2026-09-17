"""Spatial clustering and statistical hotspot analysis.

DBSCAN describes geometric density. LISA and Getis-Ord Gi* add statistical
context so a dense-looking cluster is not automatically called a significant
hotspot. Spatial weights are built from k-nearest neighbours using geographic
coordinates and permutation tests provide empirical p-values.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

EARTH_RADIUS_KM=6371.0088

def run_dbscan(df: pd.DataFrame, eps_km: float = 3.0, min_samples: int = 4) -> pd.DataFrame:
    work=df.copy(deep=True)
    if not {"Latitude","Longitude"}.issubset(work.columns):work["Cluster"]=-1;return work
    valid=work["Latitude"].notna() & work["Longitude"].notna()
    if valid.sum()<min_samples:work["Cluster"]=-1;return work
    coords_rad=np.radians(work.loc[valid,["Latitude","Longitude"]].astype(float))
    model=DBSCAN(eps=eps_km/EARTH_RADIUS_KM,min_samples=min_samples,metric="haversine").fit(coords_rad)
    work["Cluster"]=-1;work.loc[valid,"Cluster"]=model.labels_;return work

def _knn_weights(coords,k=8):
    n=len(coords)
    if n<3:return np.zeros((n,n),dtype=float)
    k=min(k,n-1);tree=cKDTree(coords);_,idx=tree.query(coords,k=k+1)
    w=np.zeros((n,n),dtype=float)
    for i in range(n):
        neighbours=np.atleast_1d(idx[i])[1:]
        w[i,neighbours]=1.0
    # Symmetric binary weights are easier to interpret for local statistics.
    w=np.maximum(w,w.T);np.fill_diagonal(w,0);return w

def _permutation_p(value,sim_values,two_sided=True):
    sims=np.asarray(sim_values,float);sims=sims[np.isfinite(sims)]
    if sims.size==0:return np.nan
    if two_sided:return float((1+np.sum(np.abs(sims)>=abs(value)))/(sims.size+1))
    return float((1+np.sum(sims>=value))/(sims.size+1))

def spatial_statistics(df: pd.DataFrame, value_col: str="Jumlah Kasus", k: int=8, permutations: int=199, seed: int=20260917) -> pd.DataFrame:
    """Return LISA and Getis-Ord Gi* statistics for each spatial observation.

    The input should contain one geographic observation per row and a numeric
    burden/rate column. If a rate is unavailable, the caller should pass case
    burden explicitly; the output is then interpreted as clustering of burden,
    not population-adjusted incidence.
    """
    required={"Latitude","Longitude",value_col}
    if not required.issubset(df.columns):return pd.DataFrame()
    out=df.copy(deep=True);valid=out.dropna(subset=["Latitude","Longitude",value_col]).copy()
    if len(valid)<4:return pd.DataFrame()
    vals=pd.to_numeric(valid[value_col],errors="coerce");valid=valid.loc[vals.notna()].copy();x=vals.loc[valid.index].to_numpy(float)
    if len(x)<4:return pd.DataFrame()
    coords=valid[["Latitude","Longitude"]].to_numpy(float);w=_knn_weights(coords,k);row_sum=w.sum(axis=1);n=len(x);mean=x.mean();z=x-mean;ss=np.sum(z*z)
    global_moran=(n/w.sum()*((z[:,None]*z[None,:]*w).sum())/ss) if w.sum()>0 and ss>0 else np.nan
    lisa=np.full(n,np.nan);lisa_z=np.full(n,np.nan);gi_z=np.full(n,np.nan);gi_p=np.full(n,np.nan);lisa_p=np.full(n,np.nan);lisa_cluster=np.array(["Tidak tersedia"]*n,dtype=object);hot=np.array([False]*n)
    if ss>0:
        lisa=(z*(w@z))/(ss/n)
        for i in range(n):
            if row_sum[i]>0:lisa_z[i]=lisa[i]
    # Local Getis-Ord Gi* uses row-standardised neighbourhood sums here.
    xsum=x.sum();x2sum=np.sum(x*x);den=np.sqrt((n*np.sum(x*x)-xsum*xsum)/(n-1)) if n>1 else np.nan
    if np.isfinite(den) and den>0:
        gi=(w@x);gi_z=(gi-row_sum*xsum/n)/(den*np.sqrt((n*row_sum-row_sum*row_sum)/(n-1)))
    rng=np.random.default_rng(seed);perm=max(0,int(permutations))
    if perm:
        lisa_sims=np.empty((perm,n));gi_sims=np.empty((perm,n))
        for b in range(perm):
            xp=rng.permutation(x);zp=xp-mean;lisa_sims[b]=zp*(w@zp)/(ss/n) if ss>0 else np.nan
            if np.isfinite(den) and den>0:gi_sims[b]=(w@xp-row_sum*xsum/n)/(den*np.sqrt(np.maximum((n*row_sum-row_sum*row_sum)/(n-1),1e-12)))
            else:gi_sims[b]=np.nan
        for i in range(n):
            lisa_p[i]=_permutation_p(lisa[i],lisa_sims[:,i],True) if np.isfinite(lisa[i]) else np.nan
            gi_p[i]=_permutation_p(gi_z[i],gi_sims[:,i],True) if np.isfinite(gi_z[i]) else np.nan
    # Moran quadrants: high-high, low-low, high-low, low-high. Only call it
    # statistically significant when empirical p <= 0.05.
    local_lag=w@z
    for i in range(n):
        if not np.isfinite(lisa[i]) or not np.isfinite(lisa_p[i]) or lisa_p[i]>0.05:continue
        if z[i]>0 and local_lag[i]>0:lisa_cluster[i]="High-High"
        elif z[i]<0 and local_lag[i]<0:lisa_cluster[i]="Low-Low"
        elif z[i]>0 and local_lag[i]<0:lisa_cluster[i]="High-Low"
        elif z[i]<0 and local_lag[i]>0:lisa_cluster[i]="Low-High"
    hot=(np.isfinite(gi_z)&np.isfinite(gi_p)&(gi_z>0)&(gi_p<=0.05))
    valid["LISA_I"]=lisa;valid["LISA_p"]=lisa_p;valid["LISA_cluster"]=lisa_cluster;valid["GiZ"]=gi_z;valid["Gi_p"]=gi_p;valid["Gi_Hotspot"]=hot;valid["Global_Moran_I"]=global_moran
    out=out.copy();cols=["LISA_I","LISA_p","LISA_cluster","GiZ","Gi_p","Gi_Hotspot","Global_Moran_I"]
    for c in cols:out[c]=np.nan if c!="LISA_cluster" and c!="Gi_Hotspot" else ("Tidak tersedia" if c=="LISA_cluster" else False)
    out.loc[valid.index,cols]=valid[cols]
    return out

def analyze_spatial(df: pd.DataFrame, eps_km: float=3.0, min_samples: int=4, k: int=8, permutations: int=199) -> pd.DataFrame:
    clustered=run_dbscan(df,eps_km,min_samples)
    if "Jumlah Kasus" not in clustered.columns:
        clustered["Jumlah Kasus"]=1
    # If rows are individual cases, aggregate to coordinates before statistical
    # hotspot testing; retain DBSCAN labels on the original case rows.
    if len(clustered)>0 and clustered[["Latitude","Longitude"]].dropna().duplicated().any():
        points=clustered.dropna(subset=["Latitude","Longitude"]).groupby(["Latitude","Longitude"],as_index=False).size().rename(columns={"size":"Jumlah Kasus"})
    else:
        points=clustered.dropna(subset=["Latitude","Longitude"])[["Latitude","Longitude","Jumlah Kasus"]].copy()
    if not points.empty:
        stats_df=spatial_statistics(points,"Jumlah Kasus",k,permutations)
        if not stats_df.empty:
            clustered=clustered.merge(stats_df[["Latitude","Longitude","LISA_I","LISA_p","LISA_cluster","GiZ","Gi_p","Gi_Hotspot"]],on=["Latitude","Longitude"],how="left")
    return clustered
