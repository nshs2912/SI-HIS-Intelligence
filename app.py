import warnings
import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
from core.analytics import REQUIRED_COLUMNS, generate_excel_template, generate_data_simulasi, get_wib_time
from core.engine import SIHISIntelligenceEngine
from core.scope import QueryScope
warnings.filterwarnings('ignore')
st.set_page_config(page_title='SI-HIS Intelligence',page_icon='🧠',layout='wide')
st.markdown('''<div style="background:#f0e68c;padding:18px 22px;border-radius:12px;border:2px solid #d4c886;"><h1 style="margin:0;color:#1e3a8a;font-size:1.7rem;">SI-HIS — Smart Integrated Health Intelligence System</h1><p style="margin:6px 0 0;color:#475569;font-weight:600;">Early Detection, Smarter Intervention</p></div>''',unsafe_allow_html=True)
st.caption(f"🕒 Waktu Sistem: {get_wib_time()['full']}")
engine=SIHISIntelligenceEngine()

def _fmt_number(v,digits=4):
    if v is None or (isinstance(v,(float,np.floating)) and np.isnan(v)):return 'Tidak dapat dihitung'
    if isinstance(v,(float,np.floating)):return f'{float(v):.{digits}f}'
    return str(v)

def show_resume(text,detail='tabel'):
    st.info(text or 'Belum tersedia interpretasi untuk scope/data ini.')
    st.caption(f"Untuk lebih detail bisa dilihat pada {detail} di bawah ini.")

def render_value(value,title=None):
    if title:st.markdown(f'#### {title}')
    if isinstance(value,pd.DataFrame):
        if value.empty:st.info('Belum ada data untuk ditampilkan.')
        else:st.dataframe(value,use_container_width=True,hide_index=True)
        return
    if isinstance(value,dict):
        if not value:st.info('Belum ada hasil.');return
        for key,item in value.items():
            st.markdown(f'**{key}**')
            if isinstance(item,pd.DataFrame):
                if item.empty:st.caption('Tidak ada hasil yang dapat dihitung.')
                else:st.dataframe(item,use_container_width=True,hide_index=True)
            elif isinstance(item,dict):render_value(item)
            elif isinstance(item,list) and item and all(isinstance(x,dict) for x in item):st.dataframe(pd.DataFrame(item),use_container_width=True,hide_index=True)
            elif item is None or (isinstance(item,(float,np.floating)) and np.isnan(item)):st.caption('Tidak dapat dihitung dari data yang tersedia.')
            else:st.write(item)
        return
    if isinstance(value,list):
        if not value:st.info('Tidak ada temuan pada bagian ini.')
        elif all(isinstance(x,dict) for x in value):st.dataframe(pd.DataFrame(value),use_container_width=True,hide_index=True)
        else:st.write(value)
        return
    if value is None:st.info('Belum tersedia untuk scope/data ini.')
    else:st.write(value)

def risk_summary(risk):
    if not isinstance(risk,dict):return pd.DataFrame()
    rows=[]
    for var,obj in risk.items():
        if var.startswith('MULTIVARIAT') or not isinstance(obj,dict):continue
        p=obj.get('p_value');valid=p is not None and pd.notna(p);sig=bool(valid and float(p)<.05)
        rows.append({'Faktor':var,'p-value':round(float(p),4) if valid else None,'Interpretasi':'Ada asosiasi statistik (p<0,05)' if sig else ('Tidak ada bukti asosiasi statistik pada α=0,05' if valid else 'Uji tidak dapat dihitung')})
    return pd.DataFrame(rows)

def risk_narrative(risk):
    s=risk_summary(risk)
    if s.empty:return 'Belum ada faktor yang dapat dievaluasi.'
    sig=s[s['p-value'].notna() & (s['p-value']<.05)]
    if sig.empty:return 'Pada data dan outcome yang dianalisis, belum ditemukan faktor dengan asosiasi statistik pada ambang p<0,05. Hal ini tidak membuktikan tidak adanya faktor risiko. Interpretasi harus mempertimbangkan OR, interval kepercayaan, ukuran sampel, confounding, bias, dan definisi outcome.'
    return f"Ditemukan sinyal asosiasi statistik pada **{', '.join(sig['Faktor'].astype(str))}**. Ini adalah asosiasi pada dataset, bukan bukti kausal. OR/CI 95% dan model multivariat perlu dibaca untuk menilai besar, arah, dan kestabilan asosiasi setelah penyesuaian."

def render_risk_factors(risk):
    if not isinstance(risk,dict):render_value(risk);return
    for factor,obj in risk.items():
        st.markdown(f'#### {factor}')
        if isinstance(obj,pd.DataFrame):render_value(obj);continue
        if not isinstance(obj,dict):st.write(obj);continue
        p=obj.get('p_value');chi=obj.get('chi2');c1,c2=st.columns(2);c1.metric('Chi-square',_fmt_number(chi));c2.metric('p-value',_fmt_number(p))
        if p is not None and pd.notna(p):
            if float(p)<.05:st.info('Terdapat asosiasi statistik pada α=0,05. Besar dan arah asosiasi tetap harus dinilai dari OR dan CI 95%.')
            else:st.info('Belum terdapat bukti asosiasi statistik pada α=0,05. Ini bukan bukti bahwa faktor tersebut tidak berpengaruh.')
        else:st.warning('Uji Chi-square tidak dapat dihitung secara valid pada tabel ini; periksa kategori kosong, variasi outcome, dan kecukupan frekuensi sel.')
        ct=obj.get('crosstab')
        if isinstance(ct,pd.DataFrame):st.markdown('**Tabel silang**');st.dataframe(ct,use_container_width=True)
        ors=obj.get('or_by_group')
        if isinstance(ors,pd.DataFrame) and not ors.empty:st.markdown('**Odds Ratio menurut kelompok**');st.dataframe(ors,use_container_width=True,hide_index=True)

def _distribution_narrative(df,label):
    if not isinstance(df,pd.DataFrame) or df.empty:return ''
    total=len(df);parts=[]
    if 'Jenis Kelamin' in df.columns:
        sex=df['Jenis Kelamin'].astype(str).replace({'nan':'Tidak diketahui'}).value_counts(dropna=False)
        if not sex.empty:
            top=sex.index[0];n=int(sex.iloc[0]);pct=n/total*100
            parts.append(f"Menurut **jenis kelamin**, distribusi seluruh kasus didominasi oleh **{top}**, sebanyak **{n:,} kasus ({pct:.2f}%)** dari seluruh kasus pada scope **{label}**.")
    if 'Umur' in df.columns:
        age=pd.to_numeric(df['Umur'],errors='coerce')
        bins=pd.cut(age,bins=[-np.inf,1,5,10,15,20,25,35,45,55,65,75,85,np.inf],labels=['<1 tahun','1-4 tahun','5-9 tahun','10-14 tahun','15-19 tahun','20-24 tahun','25-34 tahun','35-44 tahun','45-54 tahun','55-64 tahun','65-74 tahun','75-84 tahun','≥85 tahun'],right=False)
        age_counts=bins.value_counts(sort=False,dropna=True)
        if not age_counts.empty:
            top_age=age_counts.idxmax();n=int(age_counts.max());pct=n/total*100
            parts.append(f"Menurut **kelompok umur**, distribusi seluruh kasus terbesar berada pada kelompok **{top_age}**, sebanyak **{n:,} kasus ({pct:.2f}%)** dari seluruh kasus pada scope **{label}**.")
    return ' '.join(parts)

def descriptive_narrative(result,label):
    ov=result.get('overview',{}) if isinstance(result,dict) else {};total=int(ov.get('total_cases',0) or 0);deaths=int(ov.get('deaths',0) or 0);parts=[f'Scope **{label}** mencakup **{total:,} kasus/kunjungan** dengan **{deaths:,} kasus meninggal**.']
    top=result.get('top10_diseases') if isinstance(result,dict) else None
    if isinstance(top,pd.DataFrame) and not top.empty:
        r=top.iloc[0];parts.append(f"Penyakit dengan jumlah kasus terbesar adalah **{r.get('Nama Penyakit','-')}**, sebanyak **{int(r.get('Jumlah Kasus',0)):,} kasus**. Tingkat kematian akibat penyakit tersebut (**CFR**) adalah **{float(r.get('CFR',0)):.2f}%**, yaitu proporsi kasus penyakit tersebut yang meninggal akibat penyakitnya; CFR bukan mortality rate populasi.")
    if isinstance(result.get('province_distribution'),pd.DataFrame) and not result['province_distribution'].empty:
        p=result['province_distribution'].iloc[0];parts.append(f"Distribusi wilayah menunjukkan konsentrasi kasus terbesar pada **{p.get('Provinsi','-')}** sebanyak **{int(p.get('Jumlah Kasus',0)):,} kasus**.")
    dist=_distribution_narrative(result.get('source_dataframe') if isinstance(result,dict) else None,label)
    if not dist:
        sex=result.get('sex_distribution') if isinstance(result,dict) else None
        age=result.get('age_distribution') if isinstance(result,dict) else None
        if isinstance(sex,pd.DataFrame) and not sex.empty:
            row=sex.iloc[0];n=int(row.get('Jumlah Kasus',0));pct=n/total*100 if total else 0;parts.append(f"Menurut **jenis kelamin**, distribusi seluruh kasus didominasi oleh **{row.get('Jenis Kelamin','Tidak diketahui')}**, sebanyak **{n:,} kasus ({pct:.2f}%)** dari seluruh kasus pada scope **{label}**.")
        if isinstance(age,pd.DataFrame) and not age.empty:
            valid=age[age['Kelompok Umur'].astype(str).ne('nan')].copy();
            if not valid.empty:
                row=valid.sort_values('Jumlah Kasus',ascending=False).iloc[0];n=int(row.get('Jumlah Kasus',0));pct=n/total*100 if total else 0;parts.append(f"Menurut **kelompok umur**, distribusi seluruh kasus terbesar berada pada kelompok **{row.get('Kelompok Umur','Tidak diketahui')}**, sebanyak **{n:,} kasus ({pct:.2f}%)** dari seluruh kasus pada scope **{label}**.")
    parts.append('Temuan ini bersifat deskriptif; interpretasi risiko membutuhkan denominator populasi, definisi outcome, dan analisis epidemiologis lanjutan yang sesuai.')
    return ' '.join(parts)

def epidemiology_master_narrative(result,label,disease):
    """Single expert epidemiology resume combining TIME + PERSON + PLACE + priority scoring."""
    df=result.get('analysis_dataframe')
    if not isinstance(df,pd.DataFrame) or df.empty:
        return 'Data kasus pada scope analisis belum tersedia untuk membentuk resume epidemiologi terintegrasi.'
    work=df.copy()
    if 'Tanggal Sakit' in work.columns:
        work['Tanggal Sakit']=pd.to_datetime(work['Tanggal Sakit'],errors='coerce')
    total=len(work)
    death_series=pd.to_numeric(work.get('Is_Meninggal',pd.Series(0,index=work.index)),errors='coerce').fillna(0)
    if 'Status Penderita' in work.columns:
        death_series=pd.Series(np.maximum(death_series,work['Status Penderita'].astype(str).str.lower().eq('meninggal').astype(int)),index=work.index)
    deaths=int(death_series.sum());cfr=deaths/total*100 if total else 0
    parts=[f"**Resume Epidemiologi — {disease} | {label}.** Berdasarkan **{total:,} kasus** pada scope analisis, ditemukan **{deaths:,} kasus meninggal** dengan **CFR {cfr:.2f}%** (tingkat kematian di antara kasus penyakit; bukan mortality rate populasi)."]
    # TIME: latest 7 days vs preceding 7 days
    if 'Tanggal Sakit' in work.columns and work['Tanggal Sakit'].notna().any():
        end=work['Tanggal Sakit'].max().normalize()
        cur_start=end-pd.Timedelta(days=6); prev_start=end-pd.Timedelta(days=13); prev_end=end-pd.Timedelta(days=7)
        cur=work[work['Tanggal Sakit'].between(cur_start,end)]
        prev=work[work['Tanggal Sakit'].between(prev_start,prev_end)]
        cur_deaths=int(death_series.loc[cur.index].sum());prev_deaths=int(death_series.loc[prev.index].sum())
        cur_cfr=cur_deaths/len(cur)*100 if len(cur) else 0;prev_cfr=prev_deaths/len(prev)*100 if len(prev) else 0
        delta=(len(cur)-len(prev))/len(prev)*100 if len(prev) else None
        direction='meningkat' if len(cur)>len(prev) else ('menurun' if len(cur)<len(prev) else 'relatif tetap')
        parts.append(f"**TIME — 7 hari terakhir ({cur_start.date()}–{end.date()})** terdapat **{len(cur):,} kasus** dan **{cur_deaths:,} kematian**, CFR **{cur_cfr:.2f}%**; pada 7 hari sebelumnya ({prev_start.date()}–{prev_end.date()}) terdapat **{len(prev):,} kasus** dan **{prev_deaths:,} kematian**, CFR **{prev_cfr:.2f}%**. Secara temporal kasus **{direction}**" + (f" sebesar **{delta:.2f}%** dibandingkan 7 hari sebelumnya." if delta is not None else " dan baseline 7 hari sebelumnya belum tersedia."))
        # short-term daily direction
        daily=work.groupby(work['Tanggal Sakit'].dt.normalize()).size().sort_index()
        if len(daily)>=3:
            recent=daily.tail(3).tolist()
            parts.append(f"Fluktuasi jangka pendek menunjukkan jumlah kasus pada tiga hari pengamatan terakhir **{', '.join(map(str,recent))} kasus/hari**. Pola ini digunakan untuk membaca perubahan cepat dan tidak disamakan dengan tren jangka panjang.")
        span=(work['Tanggal Sakit'].max()-work['Tanggal Sakit'].min()).days
        if span>=180:
            monthly=work.groupby(work['Tanggal Sakit'].dt.to_period('M')).size()
            if len(monthly)>=6:
                parts.append(f"Pola musiman/tren jangka panjang dapat dieksplorasi karena tersedia sekitar **{span} hari** observasi; SI-HIS membandingkan distribusi bulanan dan perubahan antarperiode, sedangkan pola siklik beberapa tahun **belum dapat dinilai** tanpa seri waktu multi-tahun.")
        else:
            parts.append("Data yang tersedia belum cukup panjang untuk menyimpulkan pola musiman atau secular/cyclic trend; SI-HIS tidak akan mengklaim pola tersebut tanpa seri waktu yang memadai.")
    # PERSON
    person=[]
    if 'Umur' in work.columns:
        age=pd.to_numeric(work['Umur'],errors='coerce');bins=pd.cut(age,bins=[-np.inf,1,5,10,15,20,25,35,45,55,65,75,85,np.inf],labels=['<1 tahun','1-4 tahun','5-9 tahun','10-14 tahun','15-19 tahun','20-24 tahun','25-34 tahun','35-44 tahun','45-54 tahun','55-64 tahun','65-74 tahun','75-84 tahun','≥85 tahun'],right=False);vc=bins.value_counts(sort=False).dropna()
        if not vc.empty:
            k=vc.idxmax();n=int(vc.max());person.append(f"kelompok umur **{k}** sebanyak **{n:,} ({n/total*100:.2f}%)**")
    if 'Jenis Kelamin' in work.columns:
        vc=work['Jenis Kelamin'].astype(str).replace('nan','Tidak diketahui').value_counts(); 
        if not vc.empty: person.append(f"jenis kelamin terbanyak **{vc.index[0]}** sebanyak **{int(vc.iloc[0]):,} ({vc.iloc[0]/total*100:.2f}%)**")
    for col,label2 in [('Pekerjaan','pekerjaan'),('Status Imunisasi','status imunisasi'),('Merokok','merokok'),('Aktivitas Fisik','aktivitas fisik'),('Komorbid','komorbid')]:
        if col in work.columns:
            vc=work[col].astype(str).replace('nan','Tidak diketahui').value_counts()
            if len(vc)>0 and vc.index[0]!='Tidak diketahui':
                person.append(f"{label2} paling banyak: **{vc.index[0]}** ({int(vc.iloc[0]):,}; {vc.iloc[0]/total*100:.2f}%)")
    if person: parts.append("**PERSON —** Pada periode pengamatan yang sama, distribusi kasus menunjukkan " + "; ".join(person) + ". Kelompok dominan merupakan gambaran distribusi kasus dan belum otomatis berarti kelompok tersebut memiliki risiko tertinggi tanpa denominator populasi.")
    # PLACE + concentration
    if 'Provinsi' in work.columns:
        g=work.groupby('Provinsi').size().sort_values(ascending=False)
        if not g.empty:
            top=g.index[0];n=int(g.iloc[0]);parts.append(f"**PLACE —** Konsentrasi kasus terbesar berada di **{top}**, sebanyak **{n:,} kasus ({n/total*100:.2f}% dari kasus pada scope)**.")
            if len(g)>1:
                h=g.head(3);parts.append("Tiga wilayah dengan beban kasus terbesar: " + "; ".join([f"**{idx} {int(val):,} ({val/total*100:.2f}%)**" for idx,val in h.items()]) + ".")
    # scoring
    if 'Provinsi' in work.columns and total:
        sg=work.groupby('Provinsi').size().to_frame('Kasus');sg['Meninggal']=work.groupby('Provinsi').apply(lambda x:int(death_series.loc[x.index].sum()),include_groups=False);sg['CFR']=np.where(sg['Kasus']>0,sg['Meninggal']/sg['Kasus']*100,0)
        max_cases=float(sg['Kasus'].max()) if not sg.empty else 0;max_cfr=float(sg['CFR'].max()) if not sg.empty else 0
        sg['Burden Score']=np.where(max_cases>0,sg['Kasus']/max_cases*100,0).round(2)
        sg['CFR Score']=np.where(max_cfr>0,sg['CFR']/max_cfr*100,0).round(2)
        sg=sg.reset_index().sort_values(['Burden Score','CFR Score'],ascending=False)
        result['priority_score']=sg
        parts.append(f"**Prioritas epidemiologis:** Burden Score tertinggi berada pada **{sg.iloc[0]['Provinsi']} ({sg.iloc[0]['Burden Score']:.2f})**; CFR Score tertinggi berada pada **{sg.sort_values('CFR Score',ascending=False).iloc[0]['Provinsi']} ({sg['CFR Score'].max():.2f})**. Skor ini adalah skor prioritas operasional relatif antarwilayah dalam scope, bukan ukuran risiko individual atau signifikansi statistik.")
    parts.append("**Decision support:** berdasarkan gabungan TIME + PERSON + PLACE, SI-HIS mengidentifikasi pola yang ditemukan dalam data dan menyarankan Kemenkes/Dinkes melakukan kajian epidemiologi lebih mendalam pada wilayah/kelompok yang menjadi prioritas, termasuk penelusuran perubahan temporal, distribusi spasial, cluster/episentrum, outcome kematian, faktor risiko, dan kebutuhan penemuan kasus. Tujuannya adalah mempercepat intervensi yang tepat untuk menurunkan angka kesakitan dan kematian.")
    return " ".join(parts)


def ews_narrative(ews,rt):
    if not isinstance(ews,dict):return 'Data temporal belum cukup untuk membentuk Early Warning Score.'
    score=ews.get('ews_score',ews.get('score',ews.get('EWS')));trend=ews.get('trend_pct',ews.get('trend'));text='Early Warning membaca perubahan kasus terhadap periode pembanding. Skor SI-HIS adalah indikator kewaspadaan internal, bukan probabilitas KLB terkalibrasi.'
    if isinstance(score,(int,float)):text+=f' Skor saat ini **{score:.1f}**.'
    if isinstance(trend,(int,float)):text+=f' Perubahan tren 7 hari **{trend:.1f}%**.'
    if isinstance(rt,dict):text+=f" Rₜ relatif terbaru **{float(rt.get('rt_recent',1)):.2f}**."
    return text+' Penetapan KLB memerlukan baseline historis, definisi kasus, kelengkapan pelaporan dan verifikasi lapangan.'

def spatial_narrative(spatial):
    if not isinstance(spatial,pd.DataFrame) or spatial.empty:return 'Belum terdapat cukup koordinat untuk analisis kepadatan spasial.'
    counts=spatial['Cluster'].value_counts() if 'Cluster' in spatial else pd.Series(dtype=int);clusters=counts.drop(index=-1,errors='ignore');noise=int(counts.get(-1,0));return f"**DBSCAN** menemukan **{len(clusters)} cluster** dan **{noise} titik noise/outlier**. Cluster bukan otomatis episentrum penularan; interpretasi harus dikaitkan dengan TIME, paparan, populasi berisiko dan investigasi lapangan."

def curve_narrative(curve,disease):
    if not isinstance(curve,(tuple,list)) or len(curve)<4:return 'Klasifikasi kurva belum tersedia.'
    label,short,meaning,implication=curve[:4];return f'**Klasifikasi: {label}.** {short} **Makna:** {meaning} **Implikasi:** {implication} Pada {disease}, bentuk kurva tidak boleh digunakan sendirian untuk menyimpulkan mekanisme transmisi.'

def forecast_render(fc):
    if not isinstance(fc,dict):render_value(fc);return
    table=pd.DataFrame({'Tanggal':pd.to_datetime(fc.get('dates',[]),errors='coerce'),'Forecast':fc.get('forecast',[]),'Lower 95%':fc.get('lower',[]),'Upper 95%':fc.get('upper',[])});show_resume(f"Model **{fc.get('model','Holt-Winters')}** memperkirakan tren **{fc.get('trend','-')}**. Forecast adalah estimasi model dan intervalnya menunjukkan ketidakpastian, bukan jumlah kasus yang pasti terjadi.")
    if not table.empty:st.line_chart(table.set_index('Tanggal')[['Forecast','Lower 95%','Upper 95%']]);st.dataframe(table,use_container_width=True,hide_index=True)

def vulnerable_narrative(v):
    if not isinstance(v,list) or not v:return 'Belum ditemukan profil populasi rentan yang memenuhi batas minimal analisis.'
    d=pd.DataFrame(v);top=d.iloc[0];return f"Metode menggunakan stratifikasi **Kelompok Umur × Pekerjaan × Status Komorbid**, lalu menghitung kasus, kematian, tingkat kematian akibat penyakit (CFR) dan Risk Multiplier terhadap baseline. Profil teratas: **{top.get('Age_Group','-')} × {top.get('Pekerjaan','-')} × {top.get('Status Komorbid','-')}**, n={int(top.get('Total',0))}, CFR={float(top.get('CFR (%)',0)):.2f}%. Strata kecil harus ditafsirkan hati-hati."

def ml_narrative(ml):
    if not isinstance(ml,dict) or not ml:return 'ML belum dijalankan.'
    return 'ML digunakan sebagai decision-support, bukan kepastian klinis/epidemiologis. Model produksi harus dievaluasi dengan validasi internal-eksternal, diskriminasi, kalibrasi, class imbalance, explainability, bias, data drift dan human review.'

st.sidebar.header('⚙️ Panel Kontrol & Filter');source=st.sidebar.radio('Sumber Data',['Gunakan Data Simulasi (AI-Ready)','Upload File Excel/CSV Custom'])
if source.startswith('Gunakan'):df_raw=generate_data_simulasi().copy();st.sidebar.success(f'✅ {len(df_raw):,} data dimuat.')
else:
    uploaded=st.sidebar.file_uploader('Upload File Kasus',type=['xlsx','csv'])
    if uploaded is None:st.info('Upload file kasus untuk memulai.');st.stop()
    try:df_raw=pd.read_csv(uploaded) if uploaded.name.lower().endswith('.csv') else pd.read_excel(uploaded)
    except Exception as exc:st.error(f'Error membaca file: {exc}');st.stop()
    missing=[c for c in REQUIRED_COLUMNS if c not in df_raw.columns]
    if missing:st.error(f'Kolom wajib kurang: {missing}');st.stop()
try:st.sidebar.download_button('📥 Download Template Excel Standard',generate_excel_template(),'Template_Data_Surveilans.xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',use_container_width=True)
except Exception:pass
st.sidebar.markdown('---');st.sidebar.subheader('📍 Filter Analisis Epidemiologi');provinces=sorted(df_raw['Provinsi'].dropna().astype(str).unique()) if 'Provinsi' in df_raw else [];sel_prov=st.sidebar.selectbox('1. Provinsi',['Semua Provinsi']+provinces);dfp=df_raw if sel_prov=='Semua Provinsi' else df_raw[df_raw['Provinsi'].astype(str).eq(sel_prov)];districts=sorted(dfp['Kabupaten'].dropna().astype(str).unique()) if 'Kabupaten' in dfp else [];sel_kab=st.sidebar.selectbox('2. Kabupaten/Kota',['Semua Kabupaten/Kota']+districts);dfk=dfp if sel_kab=='Semua Kabupaten/Kota' else dfp[dfp['Kabupaten'].astype(str).eq(sel_kab)];disease_values=set()
for col in ['Diagnosis Konfirm','Diagnosis Probabel','Diagnosis Suspek']:
    if col in df_raw:disease_values.update(str(x).strip() for x in df_raw[col].dropna().unique() if str(x).strip().lower() not in {'','nan','bukan','none','tidak ada','-'})
sel_disease=st.sidebar.selectbox('3. Diagnosis Penyakit',['Semua Penyakit']+sorted(disease_values))
with st.sidebar.expander('Drill-down wilayah (opsional)'):
    kecs=sorted(dfk['Kecamatan'].dropna().astype(str).unique()) if 'Kecamatan' in dfk else [];sel_kec=st.selectbox('Kecamatan',['Semua Kecamatan']+kecs);dbase=dfk if sel_kec=='Semua Kecamatan' else dfk[dfk['Kecamatan'].astype(str).eq(sel_kec)];villages=sorted(dbase['Desa/Kelurahan'].dropna().astype(str).unique()) if 'Desa/Kelurahan' in dbase else [];sel_desa=st.selectbox('Desa/Kelurahan',['Semua Desa/Kelurahan']+villages);vbase=dbase if sel_desa=='Semua Desa/Kelurahan' else dbase[dbase['Desa/Kelurahan'].astype(str).eq(sel_desa)];pusk=sorted(vbase['Puskesmas'].dropna().astype(str).unique()) if 'Puskesmas' in vbase else [];sel_pusk=st.selectbox('Puskesmas',['Semua Puskesmas']+pusk)
include_ml=st.sidebar.checkbox('Aktifkan ML layer',False);scope=QueryScope(province=None if sel_prov=='Semua Provinsi' else sel_prov,district=None if sel_kab=='Semua Kabupaten/Kota' else sel_kab,kecamatan=None if sel_kec=='Semua Kecamatan' else sel_kec,village=None if sel_desa=='Semua Desa/Kelurahan' else sel_desa,puskesmas=None if sel_pusk=='Semua Puskesmas' else sel_pusk,disease=None if sel_disease=='Semua Penyakit' else sel_disease,period_days=3650)
if sel_disease=='Semua Penyakit':
    result=engine.descriptive(dfk);label=sel_kab if sel_kab!='Semua Kabupaten/Kota' else (sel_prov if sel_prov!='Semua Provinsi' else 'Indonesia');st.markdown(f'## 📊 Analisis Deskriptif — {label}');ov=result['overview'];a,b=st.columns(2);a.metric('Total Kunjungan Pasien',f"{ov['total_cases']:,}");b.metric('Kasus Meninggal',f"{ov['deaths']:,}");st.markdown('### Resume Epidemiologi');show_resume(descriptive_narrative(result,label));st.markdown('### 🏆 10 Besar Penyakit');show_resume('Tabel ini menunjukkan penyakit dengan beban kasus terbesar dalam scope yang dipilih. Jumlah kasus menggambarkan beban absolut; CFR menggambarkan proporsi kematian di antara kasus dan tidak boleh ditafsirkan sebagai mortality rate populasi tanpa denominator yang sesuai.');render_value(result['top10_diseases']);st.markdown('### Distribusi');show_resume('Distribusi berikut memperlihatkan komposisi kasus menurut penyakit, jenis kelamin, kelompok umur, provinsi, dan kabupaten/kota. Perbedaan jumlah kasus adalah temuan deskriptif dan tidak otomatis menunjukkan perbedaan risiko.');render_value(result['disease_distribution']);render_value(result['sex_distribution'],'Jenis Kelamin');render_value(result['age_distribution'],'Kelompok Umur');render_value(result['province_distribution'],'Provinsi');render_value(result['district_distribution'].head(50),'Kabupaten/Kota')
else:
    result=engine.analyze(df_raw,scope=scope,include_ml=include_ml,mode='epidemiology');label=sel_kab if sel_kab!='Semua Kabupaten/Kota' else (sel_prov if sel_prov!='Semua Provinsi' else 'Indonesia');st.markdown(f'## 🧬 Analisis Epidemiologi — {sel_disease}');st.caption(f'Scope: **{sel_disease} — {label}** | TIME + PERSON + PLACE')
    if not result.get('eligible',False):st.warning('Analisis epidemiologi belum dapat dijalankan.');render_value(result.get('eligibility'));st.stop()
    total=int(result['overview']['total_cases']);mortality=result.get('mortality');deaths=int(mortality.get('deaths',mortality.get('meninggal',0)) or 0) if isinstance(mortality,dict) else 0;cfr=deaths/total*100 if total else 0;a,b,c=st.columns(3);a.metric(f'Total {sel_disease}',f'{total:,}');b.metric('Meninggal',f'{deaths:,}');c.metric('CFR — tingkat kematian akibat penyakit',f'{cfr:.2f}%')
    # Navigation utama ditempatkan sebelum resume agar pengguna dapat memilih mode eksplorasi terlebih dahulu.
    tabs=st.tabs(['🚨 AI Prediction & Recommendation','📊 Trias Epidemiologi (Detail)','📈 Kurva Epidemik & Prediksi','🗺️ Peta Spasial & AI DBSCAN','🧪 Analisis Faktor Risiko','🤖 Analisis ML'])
    st.markdown('### 🧠 Resume Epidemiologi Terintegrasi — TIME + PERSON + PLACE')
    show_resume(epidemiology_master_narrative(result,label,sel_disease),'tabel dan grafik analisis di bawah')
    if isinstance(result.get('priority_score'),pd.DataFrame):
        st.markdown('#### 🎯 Prioritas Epidemiologis — Burden Score & CFR Score')
        st.dataframe(result['priority_score'],use_container_width=True,hide_index=True)
    with tabs[0]:
        st.markdown('### 🚨 AI Prediction & Recommendation')
        klb=result.get('klb',{})
        if isinstance(klb,dict):
            status=klb.get('status',klb.get('level',klb.get('classification','')))
            if status:
                s=str(status).upper()
                if 'ALERT' in s or 'INVESTIGATION' in s or 'KLB' in s: st.error(f'🔴 **ALERT SURVEILANS: {status}**')
                elif 'WARNING' in s: st.warning(f'🟡 **EARLY WARNING: {status}**')
                else: st.info(f'**Status surveilans:** {status}')
        show_resume(ews_narrative(result.get('ews'),result.get('rt')),'indikator EWS dan hasil investigasi di bawah')
        render_value(result.get('ews'),'Indikator EWS');render_value(result.get('rt'),'Rₜ')
        st.markdown('### Rekomendasi AI untuk investigasi dan pencegahan')
        rec=result.get('recommendations',result.get('recommendation'))
        if rec: render_value(rec)
        else: show_resume('SI-HIS mengarahkan tindakan berdasarkan sinyal yang terdeteksi: penyelidikan epidemiologi, penemuan kasus aktif, penelusuran kontak, analisis TIME + PERSON + PLACE, analisis cluster/episentrum, serta pengendalian faktor risiko sesuai penyakit dan temuan lapangan.')
    with tabs[1]:
        st.markdown('### 📊 Trias Epidemiologi (Detail)')
        tri=result.get('trias_summary',{});st.caption('Detail TIME + PERSON + PLACE berikut mendukung resume epidemiologi terintegrasi di atas.')
        top10=tri.get('top10_province') if isinstance(tri,dict) else None
        if isinstance(top10,pd.DataFrame) and not top10.empty:render_value(top10,'10 Besar Wilayah')
        render_value(result.get('place'),'PLACE');render_value(result.get('person'),'PERSON')
        t=result.get('time')
        if isinstance(t,pd.DataFrame) and not t.empty:
            chart=t.copy();chart['Tanggal Sakit']=pd.to_datetime(chart['Tanggal Sakit'],errors='coerce');valid_chart=chart.dropna(subset=['Tanggal Sakit'])
            if not valid_chart.empty:st.line_chart(valid_chart.set_index('Tanggal Sakit')['Jumlah Kasus'])
            render_value(t,'TIME')
        show_resume('Distribusi mortalitas dan CFR menjelaskan beban kematian relatif terhadap jumlah kasus. CFR adalah proporsi kasus penyakit yang meninggal akibat penyakit tersebut; mortality rate menggunakan populasi sebagai denominator.')
        render_value(mortality,'MORTALITY / CFR');render_value(result.get('vulnerable'),'Vulnerable Population')
    with tabs[2]:
        st.markdown('### 📈 Kurva Epidemik & Prediksi')
        st.caption('Kurva epidemik dan forecast mendukung interpretasi temporal pada resume epidemiologi terintegrasi di atas.')
        t=result.get('time')
        if isinstance(t,pd.DataFrame) and not t.empty:
            chart=t.copy();chart['Tanggal Sakit']=pd.to_datetime(chart['Tanggal Sakit'],errors='coerce');valid_chart=chart.dropna(subset=['Tanggal Sakit'])
            if not valid_chart.empty:st.line_chart(valid_chart.set_index('Tanggal Sakit')['Jumlah Kasus'])
        st.markdown('### Forecast 14 Hari');forecast_render(result.get('forecast'));waves=result.get('waves')
        if waves:render_value(waves,'Gelombang Temporal')
    with tabs[3]:
        st.markdown('### 🗺️ Peta Spasial & AI DBSCAN')
        spatial=result.get('spatial');st.caption('Analisis spasial, cluster dan episentrum digunakan untuk memperkuat interpretasi PLACE pada resume epidemiologi terintegrasi.');render_value(spatial,'Hasil DBSCAN')
        if isinstance(spatial,pd.DataFrame) and not spatial.empty and {'Latitude','Longitude'}.issubset(spatial.columns):
            geo=spatial.dropna(subset=['Latitude','Longitude'])
            if not geo.empty:
                m=folium.Map(location=[float(geo.Latitude.mean()),float(geo.Longitude.mean())],zoom_start=9)
                for _,row in geo.head(500).iterrows():folium.CircleMarker([float(row.Latitude),float(row.Longitude)],radius=4,popup=f"{row.get('Desa/Kelurahan','')} | Cluster {row.get('Cluster','')}").add_to(m)
                st_folium(m,width=None,height=500)
        render_value(result.get('epicenters'),'Episentrum / Titik Prioritas')
        st.caption('Cluster adalah hasil kepadatan spasial. Episentrum adalah lokasi konsentrasi prioritas dalam dataset; keduanya bukan otomatis sumber penularan dan harus dikaitkan dengan TIME, hubungan epidemiologis dan investigasi lapangan.')
    with tabs[4]:
        st.markdown('### 🧪 Analisis Faktor Risiko');st.markdown('#### Detail Analisis Faktor Risiko')
        s=risk_summary(result.get('risk_factors'))
        if not s.empty:st.dataframe(s,use_container_width=True,hide_index=True)
        outcomes=result.get('outcome_analyses',{})
        if isinstance(outcomes,dict):
            for outcome_name,outcome in outcomes.items():
                with st.expander(f'Outcome: {outcome_name}',expanded=(outcome_name=='Penyakit')):render_value(outcome)
        render_risk_factors(result.get('risk_factors'))
    with tabs[5]:
        st.markdown('### 🤖 Analisis ML');st.caption('Analisis ML mendukung decision support dan melengkapi analisis epidemiologi, bukan menggantikannya.')
        if include_ml:render_value(result.get('ml'))
        else:st.info('ML layer belum diaktifkan. Centang **Aktifkan ML layer** pada Panel Kontrol untuk menjalankan prediction models.')

st.caption('SI-HIS Intelligence — epidemiological decision-support with TIME + PERSON + PLACE.')
