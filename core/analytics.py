# SI-HIS epidemiology and statistical analytics layer.
# Restored from the validated prototype while preserving the API used by app.py.

import io
import math
import time
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.cluster import DBSCAN
from geopy.geocoders import Nominatim
try:
    import streamlit as st
except Exception:
    st = None

KLB_THRESHOLD = 5
REQUIRED_COLUMNS = ['Nama','Umur','Jenis Kelamin','Pekerjaan','Status Imunisasi','Status Komorbid','Riwayat Perjalanan','Faktor Risiko Lain','Tanggal Sakit','Provinsi','Kabupaten','Kecamatan','Desa/Kelurahan','Latitude','Longitude','Diagnosis Suspek','Diagnosis Probabel','Diagnosis Konfirm','Is_Konfirm','Is_Meninggal','Status Penderita']
INDEPENDENT_VARS = ['Umur','Jenis Kelamin','Pekerjaan','Status Imunisasi','Status Komorbid','Riwayat Perjalanan','Faktor Risiko Lain']
DEPENDENT_VARS = ['Is_Konfirm']

def get_wib_time():
    wib_tz=timezone(timedelta(hours=7)); now=datetime.now(wib_tz); hari=['Senin','Selasa','Rabu','Kamis','Jumat','Sabtu','Minggu']; bulan=['Januari','Februari','Maret','April','Mei','Juni','Juli','Agustus','September','Oktober','November','Desember']; return {'full':f'{hari[now.weekday()]}, {now.day} {bulan[now.month-1]} {now:%Y}, {now:%H:%M:%S} WIB','short':f'{now.day} {bulan[now.month-1]} {now:%Y}, {now:%H:%M} WIB'}

def hitung_risk_stratification(rekap_desa):
    if rekap_desa.empty:return rekap_desa
    x=rekap_desa.copy(); max_k=max(x['Total_Kasus'].max(),1); max_c=max(x['CFR (%)'].max(),1); x['Skoring_Angka_Kasus']=(x['Total_Kasus']/max_k*100).round(1); x['Skoring_Angka_Kematian']=(x['CFR (%)']/max_c*100).round(1); x['Risk_Score']=(.5*x['Skoring_Angka_Kasus']+.5*x['Skoring_Angka_Kematian']).round(1); x['Risk_Level']=x['Risk_Score'].apply(lambda s:'HIGH' if s>=60 else ('MEDIUM' if s>=35 else 'LOW')); return x

def identifikasi_vulnerable_profile(df):
    if len(df)<20:return []
    d=df.copy(); d['Age_Group']=pd.cut(d['Umur'],bins=[0,18,45,60,100],labels=['<=18','19-45','46-60','>60']); d['Fatal']=(d['Status Penderita'].astype(str).str.strip()=='Meninggal').astype(int); g=d.groupby(['Age_Group','Pekerjaan','Status Komorbid'],observed=False).agg(Total=('Fatal','count'),Meninggal=('Fatal','sum')).reset_index(); g['CFR (%)']=(g['Meninggal']/g['Total']*100).round(2); g=g[g['Total']>=5]; baseline=d['Fatal'].mean()*100; g['Risk_Multiplier']=(g['CFR (%)']/baseline).round(2) if baseline>0 else 1.0; return g.sort_values('Risk_Multiplier',ascending=False).head(5).to_dict('records')

def hitung_early_warning_score(df_epi):
    if len(df_epi)<14:return None,None,None
    d=df_epi.sort_values('Tanggal Sakit'); last7=d.tail(7)['Jumlah Kasus'].sum(); prev7=d.iloc[-14:-7]['Jumlah Kasus'].sum(); trend_pct=100.0 if prev7==0 and last7>0 else (0.0 if prev7==0 else ((last7-prev7)/prev7)*100); ews=min(100,70+(trend_pct-100)/10) if trend_pct>100 else (50+(trend_pct-50)/2.5 if trend_pct>50 else (30+trend_pct/2.5 if trend_pct>0 else max(0,30+trend_pct/2))); prob='SANGAT TINGGI (>85%)' if ews>=80 else ('TINGGI (60-85%)' if ews>=60 else ('SEDANG (30-60%)' if ews>=40 else 'RENDAH (<30%)')); return ews,trend_pct,prob

def deteksi_bentuk_kurva(df_epi,disease_name):
    if len(df_epi)<7:return 'DATA_TIDAK_CUKUP','Data terlalu sedikit.','Kumpulkan data lebih banyak.','Data belum cukup untuk analisis tren.',{}
    values=df_epi['Jumlah Kasus'].values; peak_idx=int(np.argmax(values)); mean_val=np.mean(values); skewness=stats.skew(values); kurtosis=stats.kurtosis(values); peaks=[i for i in range(1,len(values)-1) if values[i]>values[i-1] and values[i]>values[i+1] and values[i]>mean_val*1.5]; n=len(peaks); pos=peak_idx/(len(values)-1) if len(values)>1 else .5
    if n==1 and pos<.5 and skewness>.5:return 'POINT SOURCE (LOG-NORMAL)','Pola klasik sumber paparan tunggal. Puncak di awal dengan ekor penurunan yang landai.','Kurva point-source biasanya muncul ketika banyak orang terpapar sumber yang sama dalam waktu relatif singkat. Bentuk kurva harus dibaca bersama periode inkubasi dan riwayat paparan.','Jika sumber paparan dihentikan, kasus dapat bergerak turun menuju baseline; paparan ulang dapat menghasilkan puncak baru.',{'num_peaks':n,'skewness':skewness,'kurtosis':kurtosis,'peak_position_ratio':pos}
    if n==1 and .4<=pos<=.7:return 'COMMON SOURCE (CONTINUOUS)','Pola sumber paparan berkelanjutan. Puncak berada di bagian tengah periode observasi.','Pola ini dapat konsisten dengan paparan yang berlangsung terus-menerus dari satu sumber atau lingkungan yang belum dikendalikan.','Kasus dapat bertahan selama sumber paparan masih aktif; investigasi sumber dan intervensi lingkungan penting.',{'num_peaks':n,'skewness':skewness,'kurtosis':kurtosis,'peak_position_ratio':pos}
    if n>=2:return 'MULTI-PEAK / PROPAGATED-LIKE','Terdapat beberapa puncak temporal. Bentuk ini dapat konsisten dengan gelombang transmisi, introduksi berulang, atau paparan berulang.','Beberapa puncak tidak cukup untuk membuktikan transmisi antar-manusia. Interpretasi harus mempertimbangkan karakteristik penyakit, masa inkubasi, intervensi, dan kemungkinan paparan berulang.','Pantau jarak antar-puncak, perubahan intervensi, serta data kontak atau paparan untuk membedakan transmisi berantai dari paparan berulang.',{'num_peaks':n,'skewness':skewness,'kurtosis':kurtosis,'peak_position_ratio':pos}
    if n==1 and abs(skewness)<.5:return 'INTERMITTENT SOURCE','Pola paparan intermiten dengan satu puncak yang relatif simetris.','Pola dapat menunjukkan kejadian paparan yang terputus-putus atau periodik; bentuk kurva saja tidak menentukan sumbernya.','Periksa hubungan puncak dengan aktivitas, tempat, musim, atau kejadian tertentu.',{'num_peaks':n,'skewness':skewness,'kurtosis':kurtosis,'peak_position_ratio':pos}
    return 'MIXED / UNCLASSIFIED','Pola tidak cukup khas untuk satu bentuk kurva.','Kurva dapat merupakan kombinasi beberapa mekanisme atau dipengaruhi variasi pelaporan.','Gunakan investigasi epidemiologi lapangan dan data yang lebih lengkap sebelum menetapkan hipotesis sumber.',{'num_peaks':n,'skewness':skewness,'kurtosis':kurtosis,'peak_position_ratio':pos}

def prediksi_kurva_holt_winters(df_epi,forecast_days=14):
    if len(df_epi)<14:return None
    d=df_epi.copy(); d['Tanggal Sakit']=pd.to_datetime(d['Tanggal Sakit'],errors='coerce'); d=d.dropna(subset=['Tanggal Sakit']).sort_values('Tanggal Sakit'); daily=d.set_index('Tanggal Sakit')['Jumlah Kasus'].asfreq('D',fill_value=0)
    if len(daily)<14:return 'KURANG_DATA'
    try:m=ExponentialSmoothing(daily.values,trend='add',seasonal=None).fit(optimized=True)
    except Exception:return None
    f=np.asarray(m.forecast(forecast_days),dtype=float); resid=daily.values-np.asarray(m.fittedvalues,dtype=float); s=float(np.std(resid)); dates=pd.date_range(daily.index[-1]+timedelta(days=1),periods=forecast_days); idx=int(np.argmax(f)); return {'forecast':f.tolist(),'lower':(f-1.96*s).tolist(),'upper':(f+1.96*s).tolist(),'dates':dates.strftime('%Y-%m-%d').tolist(),'fitted_dates':daily.index.strftime('%Y-%m-%d').tolist(),'fitted_values':np.asarray(m.fittedvalues,dtype=float).tolist(),'peak_date':dates[idx].strftime('%Y-%m-%d'),'peak_value':float(f[idx]),'trend':'NAIK 📈' if f[-1]>daily.values[-1] else 'TURUN 📉','model':'Holt-Winters / Exponential Smoothing dengan tren aditif','confidence_method':'interval pendekatan 95% berbasis 1.96 × simpangan baku residual'}

def hitung_effective_rt(df_epi):
    if len(df_epi)<7:return None
    v=df_epi.sort_values('Tanggal Sakit')['Jumlah Kasus'].values; si=5; recent=v[-1]/v[-si] if len(v)>si and v[-si]>0 else 1.; vals=[v[i]/v[i-si] for i in range(si,len(v)) if v[i-si]>0]; avg=float(np.mean(vals)) if vals else 1.; status,color,interp=('KRITIS','#dc2626','Penularan relatif sangat aktif.') if recent>1.5 else (('MENINGKAT','#d97706','Penularan relatif masih tumbuh.') if recent>1 else (('STABIL','#65a30d','Penularan berada di sekitar titik ekuilibrium.') if recent==1 else ('MELANDAI','#16a34a','Kurva menunjukkan penurunan relatif.'))); return {'rt_recent':recent,'rt_avg':avg,'status':status,'color':color,'interpretation':interp}

def deteksi_gelombang(df_epi):
    if len(df_epi)<14:return []
    d=df_epi.sort_values('Tanggal Sakit').reset_index(drop=True); values=d['Jumlah Kasus'].values; threshold=np.mean(values)*1.3; waves=[]; in_wave=False
    for i,val in enumerate(values):
        if val>threshold and not in_wave:in_wave=True;start=i
        elif val<=threshold and in_wave:
            end=i;in_wave=False;p=start+int(np.argmax(values[start:end]));waves.append({'start_date':str(d.iloc[start]['Tanggal Sakit'].date()),'end_date':str(d.iloc[end-1]['Tanggal Sakit'].date()),'peak_date':str(d.iloc[p]['Tanggal Sakit'].date()),'peak_value':int(values[p]),'duration_days':int(end-start),'total_cases':int(np.sum(values[start:end]))})
    if in_wave:
        p=start+int(np.argmax(values[start:]));waves.append({'start_date':str(d.iloc[start]['Tanggal Sakit'].date()),'end_date':str(d.iloc[-1]['Tanggal Sakit'].date()),'peak_date':str(d.iloc[p]['Tanggal Sakit'].date()),'peak_value':int(values[p]),'duration_days':int(len(values)-start),'total_cases':int(np.sum(values[start:])),'ongoing':True})
    return waves

def hitung_jarak_km(lat1,lon1,lat2,lon2):
    R=6371.;p1,p2=math.radians(lat1),math.radians(lat2);dlat,dlon=math.radians(lat2-lat1),math.radians(lon2-lon1);a=math.sin(dlat/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dlon/2)**2;return R*(2*math.atan2(math.sqrt(a),math.sqrt(1-a)))
def get_geolocator():return Nominatim(user_agent='ai_epi_surveillance_final_v20')
def lengkapi_koordinat_otomatis(df):return df

def generate_data_simulasi():
    np.random.seed(42); wilayah=[{'desa':'Sinduadi','kec':'Mlati','kab':'Sleman','prov':'D.I. Yogyakarta','lat':-7.7583,'lon':110.3667},{'desa':'Caturtunggal','kec':'Depok','kab':'Sleman','prov':'D.I. Yogyakarta','lat':-7.7712,'lon':110.3920},{'desa':'Dago','kec':'Coblong','kab':'Bandung','prov':'Jawa Barat','lat':-6.8833,'lon':107.6167},{'desa':'Kedungdoro','kec':'Tegalsari','kab':'Surabaya','prov':'Jawa Timur','lat':-7.2620,'lon':112.7380}]; rows=[];tz=timezone(timedelta(hours=7));end=datetime.now(tz);start=end-timedelta(days=90)
    for i in range(1,801):
        w=wilayah[np.random.randint(len(wilayah))];disease=np.random.choice(['Demam Dengue','Leptospirosis','ISPA Berat','Diare Akut']);dt=start+timedelta(days=int(np.random.randint(0,90)));age=int(np.random.randint(1,80));com=np.random.choice(['Ada Komorbid','Tidak Ada'],p=[.3,.7]);travel=np.random.choice(['Ya','Tidak'],p=[.35,.65]);fatality=.02+(.13 if com=='Ada Komorbid' else 0)+(.08 if age>60 else 0);diag=np.random.choice(['Suspek','Probabel','Konfirm'],p=[.3,.3,.4] if travel=='Ya' else [.6,.3,.1]); rows.append({'Nama':f'Pasien_{i:04d}','Umur':age,'Jenis Kelamin':np.random.choice(['Laki-laki','Perempuan']),'Pekerjaan':np.random.choice(['Petani','PNS/ASN','Wiraswasta','Ibu Rumah Tangga','Pelajar/Mahasiswa']),'Tanggal Sakit':dt.strftime('%Y-%m-%d'),'Diagnosis Suspek':disease,'Diagnosis Probabel':disease if diag in ['Probabel','Konfirm'] else 'Bukan','Diagnosis Konfirm':disease if diag=='Konfirm' else 'Bukan','Status Imunisasi':np.random.choice(['Lengkap','Tidak Lengkap','Tidak Diketahui']),'Status Komorbid':com,'Riwayat Perjalanan':travel,'Faktor Risiko Lain':np.random.choice(['Kontak erat','Air tercemar','Lingkungan padat','Tidak Ada']),'Provinsi':w['prov'],'Kabupaten':w['kab'],'Kecamatan':w['kec'],'Desa/Kelurahan':w['desa'],'Puskesmas':f"Puskesmas {w['desa']}",'Latitude':w['lat']+np.random.normal(0,.01),'Longitude':w['lon']+np.random.normal(0,.01),'Is_Konfirm':1 if diag=='Konfirm' else 0,'Is_Meninggal':1 if np.random.rand()<fatality else 0,'Status Penderita':'Meninggal' if np.random.rand()<fatality else 'Hidup'})
    return pd.DataFrame(rows)

def generate_excel_template():
    out=io.BytesIO(); pd.DataFrame(columns=REQUIRED_COLUMNS).to_excel(out,index=False); out.seek(0); return out.getvalue()
