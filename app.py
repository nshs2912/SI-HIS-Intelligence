import warnings
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
from core.analytics import REQUIRED_COLUMNS, generate_excel_template, generate_data_simulasi, get_wib_time
from core.engine import SIHISIntelligenceEngine
from core.scope import QueryScope
warnings.filterwarnings('ignore')
st.set_page_config(page_title='SI-HIS Intelligence', page_icon='🧠', layout='wide')
st.markdown('''<div style="background:#f0e68c;padding:18px 22px;border-radius:12px;border:2px solid #d4c886;"><h1 style="margin:0;color:#1e3a8a;font-size:1.7rem;">SI-HIS — Smart Integrated Health Intelligence System</h1><p style="margin:6px 0 0;color:#475569;font-weight:600;">Early Detection, Smarter Intervention</p></div>''', unsafe_allow_html=True)
st.caption(f"🕒 Waktu Sistem: {get_wib_time()['full']}")
engine=SIHISIntelligenceEngine()

def render_value(value,title=None):
    if title: st.markdown(f'#### {title}')
    if isinstance(value,pd.DataFrame):
        if value.empty: st.info('Belum ada data untuk ditampilkan.')
        else: st.dataframe(value,use_container_width=True,hide_index=True)
    elif isinstance(value,dict):
        if not value: st.info('Belum ada hasil.'); return
        for k,v in value.items():
            if isinstance(v,pd.DataFrame): st.markdown(f'**{k}**'); st.dataframe(v,use_container_width=True,hide_index=True)
            elif isinstance(v,(dict,list)): st.markdown(f'**{k}**'); st.json(v)
            else: st.markdown(f'**{k}:** {v}')
    elif isinstance(value,list):
        if not value: st.info('Tidak ada temuan pada bagian ini.')
        elif all(isinstance(x,dict) for x in value): st.dataframe(pd.DataFrame(value),use_container_width=True,hide_index=True)
        else: st.write(value)
    elif value is None: st.info('Belum tersedia untuk scope/data ini.')
    else: st.write(value)

def risk_summary(risk):
    if not isinstance(risk,dict): return pd.DataFrame()
    rows=[]
    for var,obj in risk.items():
        if var.startswith('MULTIVARIAT') or not isinstance(obj,dict): continue
        p=obj.get('p_value'); sig=bool(pd.notna(p) and float(p)<.05) if p is not None else False
        rows.append({'Faktor':var,'p-value':round(float(p),4) if p is not None and pd.notna(p) else None,'Status':'Sinyal asosiasi (p<0,05)' if sig else 'Belum signifikan','Metode':'Chi-square / tabulasi silang'})
    return pd.DataFrame(rows)

def risk_narrative(risk):
    s=risk_summary(risk)
    if s.empty: return 'Tidak ditemukan faktor risiko spesifik yang dapat disimpulkan dari data yang tersedia. Hasil analisis bivariat dan multivariat selengkapnya bisa dilihat di bawah ini.'
    sig=s[s['p-value'].notna() & (s['p-value']<.05)]
    if sig.empty: return 'Analisis bivariat belum menemukan faktor dengan asosiasi statistik yang signifikan pada ambang p<0,05. Ini bukan bukti bahwa faktor tersebut tidak berpengaruh; hasil harus dibaca bersama ukuran efek, interval kepercayaan, confounding, bias, dan kualitas data. Hasil analisis bivariat dan multivariat selengkapnya bisa dilihat di bawah ini.'
    return f"Analisis bivariat menemukan sinyal asosiasi statistik pada **{', '.join(sig['Faktor'].astype(str))}** (p<0,05). Temuan ini menunjukkan perbedaan proporsi antar-kelompok dalam dataset, bukan bukti hubungan kausal. Besar efek dan hasil multivariat perlu diperiksa untuk melihat apakah sinyal tetap setelah faktor lain diperhitungkan. Hasil analisis bivariat dan multivariat selengkapnya bisa dilihat di bawah ini."

def ews_narrative(ews,rt):
    if not isinstance(ews,dict): return 'Data temporal belum cukup untuk membentuk Early Warning Score. Secara epidemiologis, sistem peringatan harus membaca perubahan insidens terhadap baseline waktu, bukan sekadar jumlah absolut kasus.'
    score=ews.get('ews_score',ews.get('score',ews.get('EWS'))); trend=ews.get('trend_pct',ews.get('trend'))
    text='SI-HIS membaca Early Warning sebagai sinyal perubahan kasus: tujuh hari terakhir dibandingkan tujuh hari sebelumnya. Skor ini merupakan indikator kewaspadaan internal dan bukan probabilitas KLB terkalibrasi.'
    if isinstance(score,(int,float)): text+=f' Skor saat ini **{score:.1f}**.'
    if isinstance(trend,(int,float)): text+=f' Perubahan tren 7 hari **{trend:.1f}%**.'
    if isinstance(rt,dict): text+=f" Rₜ relatif terbaru **{float(rt.get('rt_recent',1)):.2f}**, dengan interpretasi: {rt.get('interpretation','tersedia')}"
    text+=' Keputusan KLB tetap memerlukan verifikasi definisi kasus, kelengkapan pelaporan, baseline historis, investigasi lapangan, dan penilaian sumber paparan/transmisi.'
    return text

def spatial_narrative(spatial):
    if not isinstance(spatial,pd.DataFrame) or spatial.empty: return 'Belum terdapat cukup koordinat untuk analisis kepadatan spasial.'
    counts=spatial['Cluster'].value_counts() if 'Cluster' in spatial.columns else pd.Series(dtype=int); non_noise=counts.drop(index=-1,errors='ignore'); noise=int(counts.get(-1,0))
    return f"SI-HIS menggunakan **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** untuk menemukan konsentrasi kasus berdasarkan kepadatan geografis. Terbentuk **{len(non_noise)} cluster non-noise** dan **{noise} titik noise/outlier**. DBSCAN tidak mengharuskan jumlah cluster ditentukan sejak awal dan dapat mengenali bentuk cluster yang tidak beraturan serta memisahkan titik terpencil. K-Means memerlukan K terlebih dahulu dan mengelompokkan titik berdasarkan centroid sehingga kurang sesuai bila tujuan utama adalah menemukan hotspot tidak beraturan dan outlier. Cluster bukan otomatis episentrum penularan; hasil harus dikonfirmasi dengan TIME, riwayat paparan, kepadatan penduduk, kualitas koordinat, dan investigasi lapangan."

def curve_narrative(curve,disease):
    if not isinstance(curve,(tuple,list)) or len(curve)<4: return 'Klasifikasi kurva belum tersedia.'
    label,short,meaning,implication=curve[:4]
    return f"**Klasifikasi:** {label}. {short} **Makna:** {meaning} **Implikasi:** {implication} Untuk **{disease}**, bentuk kurva tidak boleh dipakai sendirian untuk menyimpulkan mekanisme penularan."

def forecast_render(fc):
    if not isinstance(fc,dict): render_value(fc); return
    dates=pd.to_datetime(fc.get('dates',[]),errors='coerce'); f=fc.get('forecast',[]); lo=fc.get('lower',[]); hi=fc.get('upper',[])
    table=pd.DataFrame({'Tanggal':dates,'Forecast':f,'Lower 95%':lo,'Upper 95%':hi})
    st.info(f"Model **{fc.get('model','Holt-Winters')}** memperkirakan arah **{fc.get('trend','-')}**. Nilai puncak forecast sekitar **{float(fc.get('peak_value',0)):.2f} kasus/hari** pada **{fc.get('peak_date','-')}**. Interval Lower–Upper adalah rentang ketidakpastian pendekatan model, bukan kepastian kasus aktual.")
    if not table.empty: st.line_chart(table.set_index('Tanggal')[['Forecast','Lower 95%','Upper 95%']]); st.dataframe(table,use_container_width=True,hide_index=True)
    st.caption(fc.get('confidence_method',''))

def vulnerable_narrative(v):
    if not isinstance(v,list) or not v: return 'Belum ditemukan profil populasi rentan yang memenuhi batas minimal analisis.'
    d=pd.DataFrame(v); top=d.iloc[0]
    return f"Analisis menggunakan **stratifikasi Kelompok Umur × Pekerjaan × Status Komorbid**. Setiap strata dihitung jumlah kasus, kematian, CFR, kemudian CFR dibandingkan dengan CFR baseline dataset melalui Risk Multiplier. Profil teratas pada hasil ini: **{top.get('Age_Group','-')} × {top.get('Pekerjaan','-')} × {top.get('Status Komorbid','-')}**, n={int(top.get('Total',0))}, kematian={int(top.get('Meninggal',0))}, CFR={float(top.get('CFR (%)',0)):.2f}%, multiplier={float(top.get('Risk_Multiplier',0)):.2f}×. Ini adalah sinyal stratifikasi; profil dengan n kecil memerlukan kehati-hatian dan validasi eksternal."

def ml_narrative(ml):
    if not isinstance(ml,dict) or not ml:return 'ML belum dijalankan.'
    names={'case_severity':'Case Severity','klb':'Prediksi KLB','spatial':'Spatial Outbreak','vulnerable':'Vulnerable Population'}; active=[names.get(k,k) for k in ml]
    return f"ML layer menjalankan modul **{', '.join(active)}**. ML digunakan sebagai decision-support untuk menemukan pola yang sulit dibaca secara manual. Angka prediksi tidak boleh dibaca sebagai kepastian; sebelum produksi model perlu validasi silang/eksternal, evaluasi sensitivitas-spesifisitas, AUC atau metrik yang sesuai, kalibrasi, class imbalance, explainability, data drift, dan human review."

st.sidebar.header('⚙️ Panel Kontrol & Filter')
source=st.sidebar.radio('Sumber Data',['Gunakan Data Simulasi (AI-Ready)','Upload File Excel/CSV Custom'])
if source.startswith('Gunakan'): df_raw=generate_data_simulasi().copy(); st.sidebar.success(f'✅ {len(df_raw):,} data dimuat.')
else:
    uploaded=st.sidebar.file_uploader('Upload File Kasus',type=['xlsx','csv'])
    if uploaded is None: st.info('Upload file kasus untuk memulai.'); st.stop()
    try: df_raw=pd.read_csv(uploaded) if uploaded.name.lower().endswith('.csv') else pd.read_excel(uploaded)
    except Exception as exc: st.error(f'Error membaca file: {exc}'); st.stop()
    missing=[c for c in REQUIRED_COLUMNS if c not in df_raw.columns]
    if missing: st.error(f'Kolom wajib kurang: {missing}'); st.stop()
try: st.sidebar.download_button('📥 Download Template Excel Standard',generate_excel_template(),'Template_Data_Surveilans.xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',use_container_width=True)
except Exception: pass

st.sidebar.markdown('---'); st.sidebar.subheader('📍 Filter Analisis Epidemiologi')
provinces=sorted(df_raw['Provinsi'].dropna().astype(str).unique()) if 'Provinsi' in df_raw else []; sel_prov=st.sidebar.selectbox('1. Provinsi',['Semua Provinsi']+provinces); dfp=df_raw if sel_prov=='Semua Provinsi' else df_raw[df_raw['Provinsi'].astype(str).eq(sel_prov)]
districts=sorted(dfp['Kabupaten'].dropna().astype(str).unique()) if 'Kabupaten' in dfp else []; sel_kab=st.sidebar.selectbox('2. Kabupaten/Kota',['Semua Kabupaten/Kota']+districts); dfk=dfp if sel_kab=='Semua Kabupaten/Kota' else dfp[dfp['Kabupaten'].astype(str).eq(sel_kab)]
disease_values=set()
for col in ['Diagnosis Konfirm','Diagnosis Probabel','Diagnosis Suspek']:
    if col in df_raw.columns:disease_values.update(str(x).strip() for x in df_raw[col].dropna().unique() if str(x).strip().lower() not in {'','nan','bukan','none','tidak ada','-'})
sel_disease=st.sidebar.selectbox('3. Diagnosis Penyakit',['Semua Penyakit']+sorted(disease_values))
with st.sidebar.expander('Drill-down wilayah (opsional)'):
    kecs=sorted(dfk['Kecamatan'].dropna().astype(str).unique()) if 'Kecamatan' in dfk else []; sel_kec=st.selectbox('Kecamatan',['Semua Kecamatan']+kecs); dbase=dfk if sel_kec=='Semua Kecamatan' else dfk[dfk['Kecamatan'].astype(str).eq(sel_kec)]; villages=sorted(dbase['Desa/Kelurahan'].dropna().astype(str).unique()) if 'Desa/Kelurahan' in dbase else []; sel_desa=st.selectbox('Desa/Kelurahan',['Semua Desa/Kelurahan']+villages); vbase=dbase if sel_desa=='Semua Desa/Kelurahan' else dbase[dbase['Desa/Kelurahan'].astype(str).eq(sel_desa)]; pusk=sorted(vbase['Puskesmas'].dropna().astype(str).unique()) if 'Puskesmas' in vbase else []; sel_pusk=st.selectbox('Puskesmas',['Semua Puskesmas']+pusk)
include_ml=st.sidebar.checkbox('Aktifkan ML layer',False)
scope=QueryScope(province=None if sel_prov=='Semua Provinsi' else sel_prov,district=None if sel_kab=='Semua Kabupaten/Kota' else sel_kab,kecamatan=None if sel_kec=='Semua Kecamatan' else sel_kec,village=None if sel_desa=='Semua Desa/Kelurahan' else sel_desa,puskesmas=None if sel_pusk=='Semua Puskesmas' else sel_pusk,disease=None if sel_disease=='Semua Penyakit' else sel_disease,period_days=3650)

if sel_disease=='Semua Penyakit':
    result=engine.descriptive(dfk); label=sel_kab if sel_kab!='Semua Kabupaten/Kota' else (sel_prov if sel_prov!='Semua Provinsi' else 'Indonesia'); st.markdown(f'## 📊 Analisis Deskriptif — {label}'); st.caption('Semua Penyakit → distribusi deskriptif. Kematian gabungan tidak disebut CFR.')
    ov=result['overview']; a,b=st.columns(2); a.metric('Total Kunjungan Pasien',f"{ov['total_cases']:,}"); b.metric('Kasus Meninggal',f"{ov['deaths']:,}"); st.markdown('### 🏆 10 Besar Penyakit'); st.dataframe(result['top10_diseases'],use_container_width=True,hide_index=True); st.markdown('### 🦠 Distribusi Penyakit'); st.dataframe(result['disease_distribution'],use_container_width=True,hide_index=True)
    a,b=st.columns(2)
    with a: st.markdown('### 👤 Jenis Kelamin'); st.dataframe(result['sex_distribution'],use_container_width=True,hide_index=True)
    with b: st.markdown('### 🎂 Kelompok Umur'); st.dataframe(result['age_distribution'],use_container_width=True,hide_index=True)
    a,b=st.columns(2)
    with a: st.markdown('### 🗺️ Provinsi'); st.dataframe(result['province_distribution'],use_container_width=True,hide_index=True)
    with b: st.markdown('### 🏘️ Kabupaten/Kota'); st.dataframe(result['district_distribution'].head(50),use_container_width=True,hide_index=True)
    st.info('Mode deskriptif tidak menjalankan analisis faktor risiko, EWS/KLB, DBSCAN, episentrum, kurva epidemik, forecast, atau prediksi.')
else:
    result=engine.analyze(df_raw,scope=scope,include_ml=include_ml,mode='epidemiology'); label=sel_kab if sel_kab!='Semua Kabupaten/Kota' else (sel_prov if sel_prov!='Semua Provinsi' else 'Indonesia'); st.markdown(f'## 🧬 Analisis Epidemiologi — {sel_disease}'); st.caption(f'Scope: **{sel_disease} — {label}** | TIME + PERSON + PLACE')
    if not result.get('eligible',False): st.warning('Analisis epidemiologi belum dapat dijalankan.'); [st.write('• '+x) for x in result.get('eligibility',{}).get('reasons',[])]; st.stop()
    total=int(result['overview']['total_cases']); mortality=result.get('mortality'); deaths=int(mortality.get('deaths',mortality.get('meninggal',0)) or 0) if isinstance(mortality,dict) else int(mortality['Meninggal'].sum()) if isinstance(mortality,pd.DataFrame) and 'Meninggal' in mortality else 0; cfr=deaths/total*100 if total else 0; a,b,c=st.columns(3); a.metric(f'Total {sel_disease}',f'{total:,}'); b.metric(f'Meninggal {sel_disease}',f'{deaths:,}'); c.metric('CFR',f'{cfr:.2f}%')
    tabs=st.tabs(['📊 TIME + PERSON + PLACE','🧪 Faktor Risiko','🚨 Early Warning / KLB','🗺️ Spatial / DBSCAN','📈 Kurva Epidemik','👥 Vulnerable Population','🧠 ML'])
    with tabs[0]:
        st.info('TIME + PERSON + PLACE adalah kerangka epidemiologi untuk menjawab kapan kasus terjadi, siapa yang terkena, dan di mana kasus terkonsentrasi. Kombinasi ketiganya membantu membentuk hipotesis epidemiologi sebelum investigasi lapangan dan analisis kausal.'); render_value(result.get('place'),'PLACE'); render_value(result.get('person'),'PERSON'); t=result.get('time');
        if isinstance(t,pd.DataFrame) and not t.empty and 'Tanggal Sakit' in t.columns:
            chart=t.copy(); chart['Tanggal Sakit']=pd.to_datetime(chart['Tanggal Sakit'],errors='coerce'); chart=chart.dropna(subset=['Tanggal Sakit']).set_index('Tanggal Sakit'); st.line_chart(chart['Jumlah Kasus']); st.dataframe(t,use_container_width=True,hide_index=True)
        render_value(mortality,'MORTALITY / CFR')
    with tabs[1]:
        st.markdown('### Resume Faktor Risiko'); st.info(risk_narrative(result.get('risk_factors'))); s=risk_summary(result.get('risk_factors')); 
        if not s.empty: st.dataframe(s,use_container_width=True,hide_index=True)
        st.markdown('### Hasil Analisis Bivariat & Multivariat Selengkapnya'); render_value(result.get('risk_factors')); st.caption('Hasil OR, CI, p-value, dan adjusted OR harus dibaca dengan mempertimbangkan confounding, bias, ukuran sampel, serta desain penelitian.')
    with tabs[2]:
        st.markdown('### Interpretasi Epidemiologi'); st.info(ews_narrative(result.get('ews'),result.get('rt'))); render_value(result.get('ews'),'Indikator EWS'); render_value(result.get('rt'),'Rₜ'); st.warning('Skor EWS internal bukan probabilitas KLB terkalibrasi. Status KLB memerlukan verifikasi epidemiologis.')
    with tabs[3]:
        st.markdown('### Interpretasi Spatial'); st.info(spatial_narrative(result.get('spatial'))); render_value(result.get('spatial'),'Hasil DBSCAN'); spatial=result.get('spatial')
        if isinstance(spatial,pd.DataFrame) and not spatial.empty and {'Latitude','Longitude'}.issubset(spatial.columns):
            geo=spatial.dropna(subset=['Latitude','Longitude']);
            if not geo.empty:
                m=folium.Map(location=[float(geo.Latitude.mean()),float(geo.Longitude.mean())],zoom_start=9)
                for _,row in geo.head(500).iterrows(): folium.CircleMarker([float(row.Latitude),float(row.Longitude)],radius=4,popup=f"{row.get('Desa/Kelurahan','')} | Cluster {row.get('Cluster','')}").add_to(m)
                st_folium(m,width=None,height=500)
        render_value(result.get('epicenters'),'Episentrum / Titik Prioritas')
    with tabs[4]:
        curve=result.get('epidemic_curve_classification'); st.markdown('### Interpretasi Kurva Epidemik'); st.info(curve_narrative(curve,sel_disease)); st.markdown('**Jenis kurva yang umum:** Point Source (paparan singkat), Common Source Continuous (paparan berkelanjutan), Intermittent Source (paparan berulang/terputus), Propagated/Multi-Wave (beberapa gelombang yang dapat konsisten dengan transmisi berantai tetapi tidak membuktikannya), dan Mixed/Unclassified.'); t=result.get('time')
        if isinstance(t,pd.DataFrame) and not t.empty: chart=t.copy(); chart['Tanggal Sakit']=pd.to_datetime(chart['Tanggal Sakit'],errors='coerce'); st.line_chart(chart.dropna(subset=['Tanggal Sakit']).set_index('Tanggal Sakit')['Jumlah Kasus'])
        st.markdown('### Forecast 14 Hari'); forecast_render(result.get('forecast')); waves=result.get('waves',[])
        if waves: render_value(waves,'Gelombang Temporal')
    with tabs[5]:
        st.markdown('### Interpretasi Vulnerable Population'); v=result.get('vulnerable'); st.info(vulnerable_narrative(v)); render_value(v,'Profil Rentan'); st.markdown('**Metode:** stratifikasi Kelompok Umur × Pekerjaan × Status Komorbid → hitung kasus, kematian, CFR → bandingkan CFR strata terhadap baseline melalui Risk Multiplier. Hasil merupakan sinyal stratifikasi, bukan bukti kausalitas.')
    with tabs[6]:
        if include_ml: st.markdown('### Interpretasi ML'); st.info(ml_narrative(result.get('ml'))); render_value(result.get('ml')); st.caption('Sebelum penggunaan operasional, model perlu validasi internal/eksternal, kalibrasi, evaluasi bias dan drift, explainability, serta human review.')
        else: st.info('ML layer tidak diaktifkan. Aktifkan checkbox ML di sidebar.')

st.caption('SI-HIS Intelligence — disease-specific intelligence is interpreted through TIME + PERSON + PLACE; descriptive mode remains available for all-disease distribution.')
