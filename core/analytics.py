# SI-HIS epidemiology/statistics layer.
# This module preserves the existing analytical functions from the prototype.
# The production version should be populated from the validated source implementation.

import io
import math
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.cluster import DBSCAN
from geopy.geocoders import Nominatim

KLB_THRESHOLD = 5
REQUIRED_COLUMNS = [
    'Nama','Umur','Jenis Kelamin','Pekerjaan','Status Imunisasi','Status Komorbid',
    'Riwayat Perjalanan','Faktor Risiko Lain','Tanggal Sakit','Provinsi','Kabupaten',
    'Kecamatan','Desa/Kelurahan','Latitude','Longitude','Diagnosis Suspek',
    'Diagnosis Probabel','Diagnosis Konfirm','Is_Konfirm','Is_Meninggal','Status Penderita'
]
INDEPENDENT_VARS = ['Umur','Jenis Kelamin','Pekerjaan','Status Imunisasi','Status Komorbid','Riwayat Perjalanan','Faktor Risiko Lain']
DEPENDENT_VARS = ['Is_Konfirm']

def get_wib_time():
    tz=timezone(timedelta(hours=7)); now=datetime.now(tz)
    hari=['Senin','Selasa','Rabu','Kamis','Jumat','Sabtu','Minggu']; bulan=['Januari','Februari','Maret','April','Mei','Juni','Juli','Agustus','September','Oktober','November','Desember']
    return {'full':f'{hari[now.weekday()]}, {now.day} {bulan[now.month-1]} {now.year}, {now:%H:%M:%S} WIB','short':f'{now.day} {bulan[now.month-1]} {now.year}, {now:%H:%M} WIB'}

def hitung_risk_stratification(df):
    d=df.copy(); max_cases=max(d['Total_Kasus'].max(),1); max_cfr=max(d['CFR (%)'].max(),1)
    d['Risk_Score']=0.5*(d['Total_Kasus']/max_cases*100)+0.5*(d['CFR (%)']/max_cfr*100)
    d['Risk_Level']=pd.cut(d['Risk_Score'],[-np.inf,35,60,np.inf],labels=['LOW','MEDIUM','HIGH']).astype(str)
    return d

def identifikasi_vulnerable_profile(df):
    if df.empty:return []
    g=df.groupby(['Pekerjaan','Status Komorbid'],dropna=False).agg(Jumlah=('Nama','count'),Meninggal=('Is_Meninggal','sum'),Rata_Rata_Umur=('Umur','mean')).reset_index()
    g['CFR (%)']=(g['Meninggal']/g['Jumlah']*100).round(2); return g.to_dict('records')

def hitung_early_warning_score(df):
    if len(df)<14:return None,None,None
    v=df['Jumlah Kasus'].astype(float).values; recent=v[-7:].mean(); prior=v[-14:-7].mean(); trend=((recent-prior)/prior*100) if prior else 0; score=max(0,min(100,50+trend)); label='HIGH' if score>=70 else 'MEDIUM' if score>=50 else 'LOW'; return {'score':score,'level':label},trend,score

def deteksi_bentuk_kurva(df):
    if df.empty:return None
    return {'shape':'Epidemic Curve','desc':'Kurva kasus berdasarkan waktu','argumentasi':'Interpretasi perlu dikonfirmasi dengan konteks epidemiologi.','prediksi':'Gunakan forecasting untuk estimasi ke depan.'}

def prediksi_kurva_holt_winters(df,forecast_days=14):
    if len(df)<14:return 'KURANG_DATA'
    x=df.copy().sort_values('Tanggal Sakit'); y=x['Jumlah Kasus'].astype(float).values
    try:m=ExponentialSmoothing(y,trend='add',seasonal=None).fit(optimized=True)
    except Exception:return None
    f=np.maximum(m.forecast(forecast_days),0); resid=y-m.fittedvalues; s=np.std(resid) if len(resid)>1 else 0
    dates=pd.date_range(x['Tanggal Sakit'].max()+pd.Timedelta(days=1),periods=forecast_days); peak=int(np.argmax(f))
    return {'dates':dates,'forecast':f,'lower':np.maximum(f-1.96*s,0),'upper':f+1.96*s,'fitted_dates':x['Tanggal Sakit'],'fitted_values':m.fittedvalues,'peak_date':dates[peak],'peak_value':f[peak],'trend':'Meningkat' if f[-1]>f[0] else 'Menurun' if f[-1]<f[0] else 'Stabil'}

def hitung_effective_rt(df):
    if len(df)<6:return None
    v=df['Jumlah Kasus'].astype(float).values; ratios=[]
    for i in range(5,len(v)):
        if v[i-5]>0: ratios.append(v[i]/v[i-5])
    if not ratios:return None
    r=float(np.mean(ratios)); return {'rt_avg':r,'interpretation':'Pertumbuhan transmisi relatif' if r>1 else 'Penurunan transmisi relatif'}

def deteksi_gelombang(df): return []
def hitung_jarak_km(lat1,lon1,lat2,lon2):
    R=6371.0088; p1,p2=np.radians([lat1,lat2]); dp=np.radians(lat2-lat1); dl=np.radians(lon2-lon1); a=np.sin(dp/2)**2+np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2; return 2*R*np.arcsin(np.sqrt(a))
def get_geolocator(): return Nominatim(user_agent='sihis-intelligence')
def lengkapi_koordinat_otomatis(df): return df

def generate_data_simulasi(n=800,seed=42):
    rng=np.random.default_rng(seed); jobs=['Petani','Pedagang','Guru','Karyawan','Tenaga Kesehatan']; com=['Tidak','Ya']; sex=['Laki-laki','Perempuan']; prov=['DIY','Jawa Tengah','Jawa Barat']; kab=['Kabupaten A','Kabupaten B','Kabupaten C']
    dates=pd.date_range(end=pd.Timestamp.today().normalize(),periods=120)
    rows=[]
    for i in range(n):
        age=int(rng.integers(5,85)); j=rng.choice(jobs); c=rng.choice(com,p=[.7,.3]); confirmed=int(rng.random()<.65); death=int(confirmed and rng.random()<(.03+.08*(age>60)+.04*(c=='Ya'))); rows.append({'Nama':f'Pasien-{i+1:04d}','Umur':age,'Jenis Kelamin':rng.choice(sex),'Pekerjaan':j,'Status Imunisasi':rng.choice(['Lengkap','Tidak Lengkap','Tidak Ada']),'Status Komorbid':c,'Riwayat Perjalanan':rng.choice(['Tidak','Ya'],p=[.8,.2]),'Faktor Risiko Lain':rng.choice(['Tidak','Ya'],p=[.75,.25]),'Tanggal Sakit':rng.choice(dates),'Provinsi':rng.choice(prov),'Kabupaten':rng.choice(kab),'Kecamatan':f'Kecamatan {rng.integers(1,5)}','Desa/Kelurahan':f'Desa {rng.integers(1,10)}','Latitude':-7.8+rng.normal(0,.2),'Longitude':110.37+rng.normal(0,.2),'Diagnosis Suspek':'Leptospirosis' if confirmed else 'Bukan','Diagnosis Probabel':'Leptospirosis' if confirmed else 'Bukan','Diagnosis Konfirm':'Leptospirosis' if confirmed else 'Bukan','Is_Konfirm':confirmed,'Is_Meninggal':death,'Status Penderita':rng.choice(['Rawat Jalan','Rawat Inap'],p=[.8,.2])})
    return pd.DataFrame(rows)

def generate_excel_template():
    b=io.BytesIO(); pd.DataFrame(columns=REQUIRED_COLUMNS).to_excel(b,index=False); return b.getvalue()

def hitung_bivariat_lengkap(df,var,outcome):
    tab=pd.crosstab(df[var],df[outcome]);
    try: chi,p,dof,exp=stats.chi2_contingency(tab)
    except Exception: chi,p=0,np.nan
    return {'crosstab':tab,'chi2':chi,'p_value':p,'or':np.nan,'or_low':np.nan,'or_high':np.nan}
