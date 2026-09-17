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
REQUIRED_COLUMNS = [
    'Nama','Umur','Jenis Kelamin','Pekerjaan','Status Imunisasi','Status Komorbid',
    'Riwayat Perjalanan','Faktor Risiko Lain','Tanggal Sakit','Provinsi','Kabupaten',
    'Kecamatan','Desa/Kelurahan','Latitude','Longitude','Diagnosis Suspek',
    'Diagnosis Probabel','Diagnosis Konfirm','Is_Konfirm','Is_Meninggal','Status Penderita'
]
INDEPENDENT_VARS = ['Umur','Jenis Kelamin','Pekerjaan','Status Imunisasi','Status Komorbid','Riwayat Perjalanan','Faktor Risiko Lain']
DEPENDENT_VARS = ['Is_Konfirm']

def get_wib_time():
    wib_tz = timezone(timedelta(hours=7)); now = datetime.now(wib_tz)
    hari_indo = ['Senin','Selasa','Rabu','Kamis','Jumat','Sabtu','Minggu']
    bulan_indo = ['Januari','Februari','Maret','April','Mei','Juni','Juli','Agustus','September','Oktober','November','Desember']
    return {'full': f'{hari_indo[now.weekday()]}, {now.day} {bulan_indo[now.month-1]} {now:%Y}, {now:%H:%M:%S} WIB',
            'short': f'{now.day} {bulan_indo[now.month-1]} {now:%Y}, {now:%H:%M} WIB'}

def hitung_risk_stratification(rekap_desa):
    if rekap_desa.empty: return rekap_desa
    max_kasus = rekap_desa['Total_Kasus'].max() if rekap_desa['Total_Kasus'].max() > 0 else 1
    max_cfr = rekap_desa['CFR (%)'].max() if rekap_desa['CFR (%)'].max() > 0 else 1
    rekap_desa = rekap_desa.copy()
    rekap_desa['Skoring_Angka_Kasus'] = (rekap_desa['Total_Kasus'] / max_kasus * 100).round(1)
    rekap_desa['Skoring_Angka_Kematian'] = (rekap_desa['CFR (%)'] / max_cfr * 100).round(1)
    rekap_desa['Risk_Score'] = (0.5 * rekap_desa['Skoring_Angka_Kasus'] + 0.5 * rekap_desa['Skoring_Angka_Kematian']).round(1)
    rekap_desa['Risk_Level'] = rekap_desa['Risk_Score'].apply(lambda s: 'HIGH' if s >= 60 else ('MEDIUM' if s >= 35 else 'LOW'))
    return rekap_desa

def identifikasi_vulnerable_profile(df):
    if len(df) < 20: return []
    d = df.copy()
    d['Age_Group'] = pd.cut(d['Umur'], bins=[0,18,45,60,100], labels=['<=18','19-45','46-60','>60'])
    d['Fatal'] = (d['Status Penderita'] == 'Meninggal').astype(int)
    g = d.groupby(['Age_Group','Pekerjaan','Status Komorbid'], observed=False).agg(Total=('Fatal','count'), Meninggal=('Fatal','sum')).reset_index()
    g['CFR (%)'] = (g['Meninggal']/g['Total']*100).round(2); g = g[g['Total'] >= 5]
    baseline = d['Fatal'].mean()*100
    g['Risk_Multiplier'] = (g['CFR (%)']/baseline).round(2) if baseline > 0 else 1.0
    return g.sort_values('Risk_Multiplier',ascending=False).head(5).to_dict('records')

def hitung_early_warning_score(df_epi):
    if len(df_epi) < 14: return None,None,None
    d = df_epi.sort_values('Tanggal Sakit'); last7=d.tail(7)['Jumlah Kasus'].sum(); prev7=d.iloc[-14:-7]['Jumlah Kasus'].sum()
    trend_pct = 100.0 if prev7 == 0 and last7 > 0 else (0.0 if prev7 == 0 else ((last7-prev7)/prev7)*100)
    if trend_pct > 100: ews=min(100,70+(trend_pct-100)/10)
    elif trend_pct > 50: ews=50+(trend_pct-50)/2.5
    elif trend_pct > 0: ews=30+trend_pct/2.5
    else: ews=max(0,30+trend_pct/2)
    prob='SANGAT TINGGI (>85%)' if ews>=80 else ('TINGGI (60-85%)' if ews>=60 else ('SEDANG (30-60%)' if ews>=40 else 'RENDAH (<30%)'))
    return ews,trend_pct,prob

def deteksi_bentuk_kurva(df_epi,disease_name):
    if len(df_epi)<7: return 'DATA_TIDAK_CUKUP','Data terlalu sedikit.','Kumpulkan data lebih banyak.','Data belum cukup untuk analisis tren.',{}
    values=df_epi['Jumlah Kasus'].values; peak_idx=int(np.argmax(values)); mean_val=np.mean(values); skewness=stats.skew(values); kurtosis=stats.kurtosis(values)
    peaks=[i for i in range(1,len(values)-1) if values[i]>values[i-1] and values[i]>values[i+1] and values[i]>mean_val*1.5]; n=len(peaks); pos=peak_idx/(len(values)-1) if len(values)>1 else .5
    if n==1 and pos<.5 and skewness>.5:
        return 'POINT SOURCE (LOG-NORMAL)','Pola klasik sumber paparan tunggal. Puncak di awal dengan ekor penurunan yang landai.','Kurva Point Source menunjukkan sekelompok orang terpapar sumber penyakit yang sama dalam waktu singkat.','Tren kasus diprediksi menurun menuju baseline kecuali terjadi paparan ulang.',{'num_peaks':n,'skewness':skewness,'kurtosis':kurtosis,'peak_position_ratio':pos}
    if n==1 and .4<=pos<=.7:
        return 'COMMON SOURCE (CONTINUOUS)','Pola sumber paparan berkelanjutan. Puncak di tengah periode dengan distribusi relatif simetris.','Pola ini konsisten dengan paparan berkelanjutan dari sumber yang belum dieliminasi.','Kasus dapat tetap tinggi selama sumber paparan belum diidentifikasi dan dihilangkan.',{'num_peaks':n,'skewness':skewness,'kurtosis':kurtosis,'peak_position_ratio':pos}
    if n>=2:
        return 'PROPAGATED (MULTI-WAVE)',f'Pola {n} gelombang penularan berantai antar-manusia.','Pola propagated dapat terjadi ketika transmisi berlangsung berantai atau terdapat introduksi berulang.','Gelombang berikutnya perlu dinilai bersama data intervensi dan masa inkubasi penyakit.',{'num_peaks':n,'skewness':skewness,'kurtosis':kurtosis,'peak_position_ratio':pos}
    if n==1 and abs(skewness)<.5:
        return 'INTERMITTENT SOURCE','Pola paparan intermiten dengan satu puncak simetris.','Pola ini dapat menunjukkan paparan periodik atau terputus-putus.','Surveilans dapat ditingkatkan sebelum periode paparan yang berulang.',{'num_peaks':n,'skewness':skewness,'kurtosis':kurtosis,'peak_position_ratio':pos}
    return 'MIXED / UNCLASSIFIED','Pola tidak jelas atau campuran dari beberapa mekanisme penularan.','Kurva dapat merupakan kombinasi beberapa mekanisme dan perlu investigasi epidemiologi.','Prediksi jangka pendek perlu dilengkapi investigasi lapangan.',{'num_peaks':n,'skewness':skewness,'kurtosis':kurtosis,'peak_position_ratio':pos}

def prediksi_kurva_holt_winters(df_epi,forecast_days=14):
    if len(df_epi)<14:return None
    d=df_epi.copy(); d['Tanggal Sakit']=pd.to_datetime(d['Tanggal Sakit'],errors='coerce'); d=d.dropna(subset=['Tanggal Sakit']).sort_values('Tanggal Sakit')
    daily=d.set_index('Tanggal Sakit')['Jumlah Kasus'].asfreq('D',fill_value=0)
    if len(daily)<14:return 'KURANG_DATA'
    try:m=ExponentialSmoothing(daily.values,trend='add',seasonal=None).fit(optimized=True)
    except Exception:return None
    f=m.forecast(forecast_days); resid=daily.values-m.fittedvalues; s=np.std(resid); dates=pd.date_range(daily.index[-1]+timedelta(days=1),periods=forecast_days); idx=int(np.argmax(f))
    return {'forecast':f,'lower':f-1.96*s,'upper':f+1.96*s,'dates':dates,'fitted_dates':daily.index,'fitted_values':m.fittedvalues,'peak_date':dates[idx],'peak_value':f[idx],'trend':'NAIK 📈' if f[-1]>daily.values[-1] else 'TURUN 📉'}

def hitung_effective_rt(df_epi):
    if len(df_epi)<7:return None
    v=df_epi.sort_values('Tanggal Sakit')['Jumlah Kasus'].values; si=5; recent=v[-1]/v[-si] if len(v)>si and v[-si]>0 else 1.0; vals=[v[i]/v[i-si] for i in range(si,len(v)) if v[i-si]>0]; avg=float(np.mean(vals)) if vals else 1.0
    if recent>1.5: status, color, interp='KRITIS','#dc2626','Penularan relatif sangat aktif.'
    elif recent>1: status,color,interp='MENINGKAT','#d97706','Penularan relatif masih tumbuh.'
    elif recent==1: status,color,interp='STABIL','#65a30d','Penularan berada di sekitar titik ekuilibrium.'
    else: status,color,interp='MELANDAI','#16a34a','Kurva menunjukkan penurunan relatif.'
    return {'rt_recent':recent,'rt_avg':avg,'status':status,'color':color,'interpretation':interp}

def deteksi_gelombang(df_epi):
    if len(df_epi)<14:return []
    d=df_epi.sort_values('Tanggal Sakit').reset_index(drop=True); values=d['Jumlah Kasus'].values; threshold=np.mean(values)*1.3; waves=[]; in_wave=False
    for i,val in enumerate(values):
        if val>threshold and not in_wave: in_wave=True; start=i
        elif val<=threshold and in_wave:
            end=i; in_wave=False; p=start+int(np.argmax(values[start:end])); waves.append({'start_date':d.iloc[start]['Tanggal Sakit'],'end_date':d.iloc[end-1]['Tanggal Sakit'],'peak_date':d.iloc[p]['Tanggal Sakit'],'peak_value':values[p],'duration_days':end-start,'total_cases':int(np.sum(values[start:end]))})
    if in_wave:
        p=start+int(np.argmax(values[start:])); waves.append({'start_date':d.iloc[start]['Tanggal Sakit'],'end_date':d.iloc[-1]['Tanggal Sakit'],'peak_date':d.iloc[p]['Tanggal Sakit'],'peak_value':values[p],'duration_days':len(values)-start,'total_cases':int(np.sum(values[start:])),'ongoing':True})
    return waves

def hitung_jarak_km(lat1,lon1,lat2,lon2):
    R=6371.; p1,p2=math.radians(lat1),math.radians(lat2); dlat,dlon=math.radians(lat2-lat1),math.radians(lon2-lon1); a=math.sin(dlat/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dlon/2)**2; return R*(2*math.atan2(math.sqrt(a),math.sqrt(1-a)))

def get_geolocator(): return Nominatim(user_agent='ai_epi_surveillance_final_v20')

def lengkapi_koordinat_otomatis(df):
    if st is None:return df
    if 'geocode_cache' not in st.session_state: st.session_state.geocode_cache={}
    cache=st.session_state.geocode_cache; geolocator=get_geolocator(); out=df.copy(); missing=out['Latitude'].isna()|out['Longitude'].isna(); dm=out[missing].copy()
    if dm.empty:return out
    dm['alamat_key']=dm.apply(lambda r:f"{r['Desa/Kelurahan']}|{r['Kecamatan']}|{r['Kabupaten']}|{r['Provinsi']}",axis=1); todo=[a for a in dm['alamat_key'].unique() if a not in cache]
    if todo:
        bar=st.progress(0)
        for i,key in enumerate(todo):
            parts=key.split('|')
            try:
                loc=geolocator.geocode(f'{parts[0]}, {parts[1]}, {parts[2]}, {parts[3]}, Indonesia',timeout=3); cache[key]=(round(loc.latitude,6),round(loc.longitude,6)) if loc else (np.nan,np.nan); time.sleep(1.2)
            except Exception: cache[key]=(np.nan,np.nan)
            bar.progress((i+1)/len(todo))
        bar.empty()
    for idx,row in dm.iterrows(): out.at[idx,'Latitude'],out.at[idx,'Longitude']=cache.get(row['alamat_key'],(np.nan,np.nan))
    return out

def generate_data_simulasi():
    np.random.seed(42); wilayah=[{'desa':'Sinduadi','kec':'Mlati','kab':'Sleman','prov':'D.I. Yogyakarta','lat':-7.7583,'lon':110.3667},{'desa':'Caturtunggal','kec':'Depok','kab':'Sleman','prov':'D.I. Yogyakarta','lat':-7.7712,'lon':110.3920},{'desa':'Dago','kec':'Coblong','kab':'Bandung','prov':'Jawa Barat','lat':-6.8833,'lon':107.6167},{'desa':'Kedungdoro','kec':'Tegalsari','kab':'Surabaya','prov':'Jawa Timur','lat':-7.2620,'lon':112.7380}]
    rows=[]; tz=timezone(timedelta(hours=7)); end=datetime.now(tz); start=end-timedelta(days=90)
    for i in range(1,801):
        w=wilayah[np.random.randint(len(wilayah))]; disease=np.random.choice(['Demam Dengue','Leptospirosis','ISPA Berat','Diare Akut']); dt=start+timedelta(days=int(np.random.randint(0,90))); age=int(np.random.randint(1,80)); com=np.random.choice(['Ada Komorbid','Tidak Ada'],p=[.3,.7]); travel=np.random.choice(['Ya','Tidak'],p=[.35,.65]); fatality=.02+(.13 if com=='Ada Komorbid' else 0)+(.08 if age>60 else 0); probs=np.array([max(.3,.85-fatality*3),.20,.10+fatality,min(.25,fatality)]); probs=probs/probs.sum(); diag=np.random.choice(['Suspek','Probabel','Konfirm'],p=[.3,.3,.4] if travel=='Ya' else [.6,.3,.1])
        rows.append({'Nama':f'Pasien_{i:04d}','Umur':age,'Jenis Kelamin':np.random.choice(['Laki-laki','Perempuan']),'Pekerjaan':np.random.choice(['Petani','PNS/ASN','Wiraswasta','Ibu Rumah Tangga','Pelajar/Mahasiswa']),'Tanggal Sakit':dt.strftime('%Y-%m-%d'),'Diagnosis Suspek':disease if diag in ['Suspek','Probabel','Konfirm'] else 'Bukan','Diagnosis Probabel':disease if diag in ['Probabel','Konfirm'] else 'Bukan','Diagnosis Konfirm':disease if diag=='Konfirm' else 'Bukan','Riwayat Perjalanan':travel,'Status Komorbid':com,'Status Imunisasi':np.random.choice(['Lengkap','Tidak Lengkap'],p=[.65,.35]),'Faktor Risiko Lain':np.random.choice(['Paparan Genangan Air','Kontak Ternak/Vektor','Konsumsi Air Tak Dimasak','Kerumunan Padat','Tidak Ada']),'Status Penderita':np.random.choice(['Sembuh','Rawat Jalan','Rawat Inap','Meninggal'],p=probs),'Desa/Kelurahan':w['desa'],'Kecamatan':w['kec'],'Kabupaten':w['kab'],'Provinsi':w['prov'],'Latitude':round(w['lat']+np.random.uniform(-.01,.01),6),'Longitude':round(w['lon']+np.random.uniform(-.01,.01),6)})
    d=pd.DataFrame(rows); d.loc[d.sample(frac=.05,random_state=42).index,['Latitude','Longitude']]=np.nan; d['Is_Konfirm']=np.nan; d['Is_Meninggal']=np.nan; return d[REQUIRED_COLUMNS]

def generate_excel_template():
    out=io.BytesIO();
    with pd.ExcelWriter(out,engine='openpyxl') as writer: generate_data_simulasi().head(15).to_excel(writer,index=False,sheet_name='Template_Data')
    return out.getvalue()

def hitung_bivariat_lengkap(df,var_indep,var_dep_binary):
    ct=pd.crosstab(df[var_indep],df[var_dep_binary])
    try: chi2,p_val,_,_=stats.chi2_contingency(ct)
    except Exception: chi2,p_val=np.nan,np.nan
    or_val=or_low=or_high=np.nan
    if ct.shape==(2,2):
        a,b,c,d=ct.iloc[1,1],ct.iloc[1,0],ct.iloc[0,1],ct.iloc[0,0]
        if min(a,b,c,d)>0:
            or_val=(a*d)/(b*c); se=math.sqrt(1/a+1/b+1/c+1/d); or_low,or_high=math.exp(math.log(or_val)-1.96*se),math.exp(math.log(or_val)+1.96*se)
    return {'crosstab':ct,'chi2':chi2,'p_value':p_val,'or':or_val,'or_low':or_low,'or_high':or_high}
