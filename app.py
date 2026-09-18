import warnings
import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
from core.analytics import REQUIRED_COLUMNS, generate_excel_template, generate_data_simulasi, get_wib_time
from core.downloads import build_evaluation_excel, build_evaluation_csv, build_resume_pdf, data_sha256
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
    st.markdown('''**DISCLAIMER PENTING**\n\nHasil analisis berfungsi sebagai **Decision Support System (DSS)**. Keputusan operasional tetap berada di bawah wewenang otoritas kesehatan.''')

def show_tab_resume(title,text,detail='analisis di bawah'):
    st.markdown(f'### 🧭 {title}')
    show_resume(text,detail)

def build_export_metadata(df, disease, label, source_name, sel_prov, sel_kab, sel_kec, sel_desa, sel_pusk):
    sha = data_sha256(df)
    return {
        'Produk': 'SI-HIS Intelligence',
        'Jenis Dokumen': 'Resume Epidemiologi + Datasheet Verifikasi',
        'Penyakit': disease,
        'Scope Analisis': label,
        'Provinsi Filter': sel_prov,
        'Kabupaten/Kota Filter': sel_kab,
        'Kecamatan Filter': sel_kec,
        'Desa/Kelurahan Filter': sel_desa,
        'Puskesmas Filter': sel_pusk,
        'Sumber Data': source_name,
        'Jumlah Baris yang Dianalisis': len(df),
        'Jumlah Kolom': len(df.columns),
        'Waktu Ekspor': get_wib_time()['full'],
        'SHA-256 Datasheet': sha,
        'Catatan Verifikasi': 'Datasheet berisi baris yang benar-benar digunakan pada scope analisis ini.'
    }


def render_scoring_explanation():
    st.markdown('### ❓ Mengapa Perlu Skoring?')
    st.markdown('''Dalam analisis risiko epidemiologi, kita menghadapi masalah klasik: **jumlah kasus adalah angka absolut** (misal: 50 kasus), sedangkan **CFR adalah persentase** (misal: 5%). Keduanya memiliki satuan dan skala yang berbeda, sehingga tidak bisa langsung dijumlahkan.''')
    st.markdown('### 📊 Skoring Angka Kasus (Case Burden Score)')
    st.markdown('''Setiap wilayah diberikan skor 0–100 berdasarkan jumlah kasusnya relatif terhadap wilayah dengan jumlah kasus tertinggi.\n\n> **Rumus:** Skoring Angka Kasus = (Jumlah Kasus Wilayah / Jumlah Kasus Wilayah Tertinggi) × 100\n\n> **Contoh:** Jika Wilayah A punya 50 kasus dan wilayah dengan kasus tertinggi punya 100 kasus, maka Skoring Angka Kasus Wilayah A = (50/100) × 100 = **50**''')
    st.markdown('### 💀 Skoring Angka Kematian (Fatality Score)')
    st.markdown('''Setiap wilayah diberikan skor 0–100 berdasarkan CFR-nya relatif terhadap wilayah dengan CFR tertinggi.\n\n> **Rumus:** Skoring Angka Kematian = (CFR Wilayah / CFR Wilayah Tertinggi) × 100\n\n> **Contoh:** Jika Wilayah A punya CFR 5% dan wilayah dengan CFR tertinggi punya CFR 10%, maka Skoring Angka Kematian Wilayah A = (5/10) × 100 = **50**''')
    st.markdown('### ⚖️ Risk Score Komposit')
    st.markdown('''Kedua skor digabungkan dengan bobot 50:50 untuk menghasilkan **Risk Score** yang mencerminkan baik *beban jumlah* maupun *tingkat keparahan*:\n\n> **Rumus:** Risk Score = (0.5 × Skoring Angka Kasus) + (0.5 × Skoring Angka Kematian)\n\n**Klasifikasi Risiko:**\n\n- 🔴 **HIGH (≥60)** — Memerlukan intervensi masif segera\n- 🟡 **MEDIUM (35–59)** — Perlu penguatan surveilans dan kesiapsiagaan\n- 🟢 **LOW (<35)** — Pertahankan pemantauan rutin''')

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
    if sig.empty:return 'Pada data dan outcome yang dianalisis, belum ditemukan faktor dengan asosiasi statistik pada ambang p<0,05. Hal ini tidak membuktikan tidak adanya faktor risiko. Interpretasi harus mempertimbangkan crude OR (cOR), adjusted OR (aOR), interval kepercayaan, ukuran sampel, confounding, bias, dan definisi outcome.'
    return f"Ditemukan sinyal asosiasi statistik pada **{', '.join(sig['Faktor'].astype(str))}**. Ini adalah asosiasi pada dataset, bukan bukti kausal. **cOR** menggambarkan asosiasi sebelum penyesuaian, sedangkan **aOR** dari model multivariat menggambarkan asosiasi setelah mengontrol variabel lain yang masuk model. Keduanya perlu dibaca bersama CI 95% dan p-value."

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
        if isinstance(ors,pd.DataFrame) and not ors.empty:
            st.markdown('**Crude Odds Ratio (cOR) menurut kategori — referensi = kategori pertama yang tersedia**')
            st.dataframe(ors,use_container_width=True,hide_index=True)

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

def build_priority_score(work,death_series):
    """Composite burden/fatality priority at the finest available geographic level."""
    if not isinstance(work,pd.DataFrame) or work.empty:
        return pd.DataFrame()
    level=None
    for candidate in ['Desa/Kelurahan','Kecamatan','Kabupaten','Provinsi']:
        if candidate in work.columns and work[candidate].notna().any():
            level=candidate
            break
    if level is None:return pd.DataFrame()
    tmp=pd.DataFrame({level:work[level].astype(str), '_death':pd.to_numeric(death_series,errors='coerce').fillna(0).astype(int)})
    sg=tmp.groupby(level).agg(Kasus=(level,'size'),Meninggal=('_death','sum')).reset_index()
    sg['CFR']=np.where(sg['Kasus']>0,sg['Meninggal']/sg['Kasus']*100,0)
    max_cases=float(sg['Kasus'].max()) if not sg.empty else 0
    max_cfr=float(sg['CFR'].max()) if not sg.empty else 0
    sg['Burden Score']=np.where(max_cases>0,sg['Kasus']/max_cases*100,0).round(2)
    sg['CFR Score']=np.where(max_cfr>0,sg['CFR']/max_cfr*100,0).round(2)
    sg['Risk Score']=(0.5*sg['Burden Score']+0.5*sg['CFR Score']).round(2)
    sg['Risk Level']=np.select([sg['Risk Score']>=60,sg['Risk Score']>=35],['HIGH','MEDIUM'],default='LOW')
    return sg.sort_values(['Risk Score','Burden Score','CFR Score'],ascending=False).reset_index(drop=True)

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
    # scoring: use the finest geographic level available, matching the drill-down.
    sg=build_priority_score(work,death_series)
    if not sg.empty:
        result['priority_score']=sg
        level=sg.columns[0]
        top_b=sg.iloc[0]
        top_f=sg.sort_values('CFR Score',ascending=False).iloc[0]
        parts.append(f"**Prioritas epidemiologis:** pada level **{level}**, Risk Score tertinggi berada di **{top_b[level]} ({top_b['Risk Score']:.2f}; {top_b['Risk Level']})**. Burden Score tertinggi berada di **{sg.sort_values('Burden Score',ascending=False).iloc[0][level]} ({sg['Burden Score'].max():.2f})**, sedangkan CFR Score tertinggi berada di **{top_f[level]} ({top_f['CFR Score']:.2f}; CFR {top_f['CFR']:.2f}%).** Skor adalah prioritas operasional relatif dalam scope, bukan risiko individual, signifikansi statistik, atau kriteria legal KLB.")
    parts.append("**Decision support:** berdasarkan gabungan TIME + PERSON + PLACE, SI-HIS mengidentifikasi pola yang ditemukan dalam data dan menyarankan Kemenkes/Dinkes melakukan kajian epidemiologi lebih mendalam pada wilayah/kelompok yang menjadi prioritas, termasuk penelusuran perubahan temporal, distribusi spasial, cluster/episentrum, outcome kematian, faktor risiko, dan kebutuhan penemuan kasus. Tujuannya adalah mempercepat intervensi yang tepat untuk menurunkan angka kesakitan dan kematian.")
    return " ".join(parts)


def ews_narrative(ews,rt):
    if not isinstance(ews,dict):return 'Data temporal belum cukup untuk membentuk Early Warning Score.'
    score=ews.get('ews_score',ews.get('score',ews.get('EWS')));trend=ews.get('trend_pct',ews.get('trend'));text='Early Warning membaca perubahan kasus terhadap periode pembanding. Skor SI-HIS adalah indikator kewaspadaan internal, bukan probabilitas KLB terkalibrasi.'
    if isinstance(score,(int,float)):text+=f' Skor saat ini **{score:.1f}**.'
    if isinstance(trend,(int,float)):text+=f' Perubahan tren 7 hari **{trend:.1f}%**.'
    if isinstance(rt,dict):text+=f" Rₜ relatif terbaru **{float(rt.get('rt_recent',1)):.2f}**."
    return text+' Penetapan KLB memerlukan baseline historis, definisi kasus, kelengkapan pelaporan dan verifikasi lapangan.'

def _fmt_pct(value):
    try:
        return f"{float(value):.2f}%"
    except Exception:
        return "Tidak dapat dihitung"

def _fmt_int(value):
    try:
        return f"{int(value):,}"
    except Exception:
        return "Tidak dapat dihitung"

def _death_series_app(df):
    death=pd.to_numeric(df.get('Is_Meninggal',pd.Series(0,index=df.index)),errors='coerce').fillna(0).astype(int)
    if 'Status Penderita' in df.columns:
        status=df['Status Penderita'].astype(str).str.strip().str.lower().eq('meninggal').astype(int)
        death=pd.Series(np.maximum(death,status),index=df.index)
    return death

def _outcome_resume(outcomes):
    if not isinstance(outcomes,dict):
        return "Modul outcome-specific belum tersedia."
    sections=[]
    for name,obj in outcomes.items():
        if not isinstance(obj,dict):
            continue
        if obj.get('available'):
            n=obj.get('n',0);events=obj.get('events',0);rate=obj.get('event_rate_pct')
            sections.append(
                f"**{name} — MODEL TERSEDIA.** Pertanyaan analisis: {obj.get('question','-')} "
                f"Jumlah observasi lengkap **{_fmt_int(n)}**, event/outcome **{_fmt_int(events)}**"
                + (f" ({_fmt_pct(rate)})." if rate is not None else ".")
                + " Faktor independen dianalisis terhadap outcome ini secara terpisah; hasil asosiasi tidak boleh dipindahkan menjadi kesimpulan untuk outcome lain."
            )
        else:
            sections.append(
                f"**{name} — BELUM DAPAT DIMODELKAN.** {obj.get('status','Data belum memenuhi syarat.')}"
            )
    return "\n\n".join(sections) if sections else "Belum ada outcome yang dapat diringkas."

def ai_prediction_narrative(result,disease,label):
    """Reference-aligned, data-driven Situational Intelligence Report."""
    df=result.get('analysis_dataframe')
    if not isinstance(df,pd.DataFrame) or df.empty:
        return "**AI Epidemiologist Situational Awareness Report** belum dapat dibentuk karena data kasus pada scope ini kosong."

    work=df.copy()
    if 'Tanggal Sakit' in work.columns:
        work['Tanggal Sakit']=pd.to_datetime(work['Tanggal Sakit'],errors='coerce')
    total=len(work)
    death=_death_series_app(work)
    work['_death_tmp']=death
    deaths=int(death.sum())
    cfr=deaths/total*100 if total else 0.0

    def count_confirmed():
        if 'Is_Konfirm' in work.columns:
            x=pd.to_numeric(work['Is_Konfirm'],errors='coerce').fillna(0)
            return int((x==1).sum())
        if 'Diagnosis Konfirm' in work.columns:
            return int(work['Diagnosis Konfirm'].astype(str).str.strip().ne('').sum() - work['Diagnosis Konfirm'].astype(str).str.strip().str.lower().isin({'bukan','nan','none',''}).sum())
        return None

    def count_hospitalized():
        candidates=['Status Rawat','Status Perawatan','Rawat Inap','Is_Rawat_Inap','Hospitalized','Status Hospitalisasi']
        for col in candidates:
            if col not in work.columns:
                continue
            v=work[col]
            if pd.api.types.is_bool_dtype(v):
                return int(v.fillna(False).sum())
            num=pd.to_numeric(v,errors='coerce')
            if num.notna().any():
                return int((num==1).sum())
            return int(v.astype(str).str.strip().str.lower().isin({'ya','yes','rawat inap','inap','hospitalized','true','1'}).sum())
        return None

    confirmed=count_confirmed()
    hospitalized=count_hospitalized()
    sections=[]
    sections.append(
        f"### 🧠 AI EPIDEMIOLOGIST SITUATIONAL AWARENESS REPORT\n\n"
        f"Laporan ini digenerate otomatis oleh AI berdasarkan analisis **{_fmt_int(total)} kasus {disease}** "
        f"pada wilayah/scope **{label}**. Data dianalisis menggunakan pendekatan **Trias Epidemiologi "
        f"(Person–Place–Time)**, **skoring risiko komposit**, **deteksi pola temporal-spasial**, "
        f"outcome-specific analysis, early warning dan forecasting."
    )

    # 1 STATUS SITUASI
    status=(f"Pada periode surveilans aktif ini, sistem mencatat **{_fmt_int(total)} kasus {disease}** di scope **{label}**. ")
    if confirmed is not None:
        status+=f"Sebanyak **{_fmt_int(confirmed)} kasus ({confirmed/total*100:.1f}%)** berstatus terkonfirmasi berdasarkan field konfirmasi yang tersedia. "
    else:
        status+="Status jumlah kasus terkonfirmasi tidak dapat dihitung karena field konfirmasi yang dapat digunakan tidak tersedia. "
    if hospitalized is not None:
        status+=f"Sistem mencatat **{_fmt_int(hospitalized)} pasien ({hospitalized/total*100:.1f}%)** dengan status rawat inap. "
    else:
        status+="Jumlah rawat inap tidak ditampilkan karena field hospitalisasi tidak tersedia pada dataset. "
    status+=f"Terdapat **{_fmt_int(deaths)} kematian** dengan **CFR {cfr:.2f}%**."
    sections.append("### 📌 1. STATUS SITUASI EPIDEMIOLOGI\n"+status)
    if deaths>0:
        sections.append(
            "🔴 **ALERT MERAH — KEMATIAN TERDETEKSI.** "
            f"Ditemukan **{_fmt_int(deaths)} kasus meninggal**. Setiap kematian pada kasus penyakit memicu "
            "**ALERT merah SI-HIS**, memerlukan respons SKD dan **penyelidikan epidemiologi lebih lanjut**. "
            "Alert ini bukan penetapan status KLB otomatis."
        )
    # CFR interpretation tied to the composite score, not a claimed national threshold.
    sections.append(
        f"**Interpretasi CFR:** CFR pada scope ini adalah **{cfr:.2f}%**. Nilai CFR dibaca bersama jumlah kasus, "
        "kematian, distribusi wilayah dan outcome. Jika suatu wilayah memperoleh Risk Score ≥60, klasifikasi **HIGH** "
        "berasal dari kombinasi Case Burden Score + CFR/Fatality Score dalam ruleset operasional SI-HIS; "
        "bukan dari ambang CFR nasional dan bukan kriteria legal KLB."
    )

    # 2 PERSON
    person_lines=[]
    if 'Jenis Kelamin' in work.columns:
        vc=work['Jenis Kelamin'].astype(str).replace('nan','Tidak diketahui').value_counts()
        if not vc.empty:
            person_lines.append("jenis kelamin dominan **"+str(vc.index[0])+f"** ({_fmt_int(vc.iloc[0])}; {vc.iloc[0]/total*100:.1f}%)")
    if 'Umur' in work.columns:
        age=pd.to_numeric(work['Umur'],errors='coerce')
        bins=pd.cut(age,bins=[0,19,46,61,121],labels=['≤18','19-45','46-60','>60'],right=False)
        vc=bins.value_counts(sort=False).dropna()
        if not vc.empty:
            k=vc.idxmax();n=int(vc.max())
            person_lines.append(f"kelompok umur dominan **{k}** ({_fmt_int(n)}; {n/total*100:.1f}%)")
    if 'Pekerjaan' in work.columns:
        vc=work['Pekerjaan'].astype(str).replace('nan','Tidak diketahui').value_counts()
        if not vc.empty and vc.index[0]!='Tidak diketahui':
            person_lines.append(f"pekerjaan dominan **{vc.index[0]}** ({_fmt_int(vc.iloc[0])}; {vc.iloc[0]/total*100:.1f}%)")
    sections.append(
        "### 👥 2. PROFIL POPULASI TERDAMPAK (ASPEK PERSON)\n"
        + ("Analisis demografi menunjukkan bahwa beban penyakit terdistribusi tidak merata: "+", ".join(person_lines)+". " if person_lines else "Variabel person belum memadai. ")
        + "Dominasi kelompok merupakan distribusi kasus, bukan bukti bahwa kelompok tersebut memiliki risiko populasi tertinggi. "
        "Untuk menilai faktor risiko diperlukan denominator populasi dan analisis outcome yang sesuai."
    )

    # 3 PLACE + explicit drill-down when deaths exist
    place_sections=[]
    for col,label2 in [('Provinsi','provinsi'),('Kabupaten','kabupaten/kota'),('Kecamatan','kecamatan'),('Desa/Kelurahan','desa/kelurahan')]:
        if col in work.columns:
            g=work.groupby(col,dropna=False).agg(Kasus=(col,'size'),Meninggal=('_death_tmp','sum')).reset_index()
            g['CFR']=np.where(g['Kasus']>0,g['Meninggal']/g['Kasus']*100,0)
            g=g.sort_values(['Kasus',col],ascending=[False,True])
            if not g.empty:
                r=g.iloc[0]
                place_sections.append(f"**{label2} terbanyak:** {r[col]} — {_fmt_int(r['Kasus'])} kasus; {_fmt_int(r['Meninggal'])} meninggal; CFR {_fmt_pct(r['CFR'])}.")
    sections.append(
        "### 📍 3. EPICENTRUM & DISTRIBUSI SPASIAL (ASPEK PLACE)\n"
        + (" ".join(place_sections) if place_sections else "Variabel geografis belum tersedia.")
        + " Konsentrasi kasus adalah sinyal beban absolut. Konsentrasi tersebut tidak otomatis merupakan sumber penularan."
    )
    if deaths>0 and 'Desa/Kelurahan' in work.columns:
        dg=work.groupby('Desa/Kelurahan').agg(Kasus=('Desa/Kelurahan','size'),Meninggal=('_death_tmp','sum')).reset_index()
        dg['CFR']=np.where(dg['Kasus']>0,dg['Meninggal']/dg['Kasus']*100,0)
        death_v=dg[dg['Meninggal']>0].sort_values(['Meninggal','CFR','Kasus'],ascending=False)
        if not death_v.empty:
            rows=[]
            for _,r in death_v.head(10).iterrows():
                rows.append(f"**{r['Desa/Kelurahan']}**: {int(r['Kasus'])} kasus, {int(r['Meninggal'])} meninggal, CFR {r['CFR']:.2f}%")
            sections.append(
                "🔴 **Distribusi kematian pada level desa/kelurahan:** " + "; ".join(rows) +
                ". Lokasi kematian menjadi prioritas penelusuran untuk melihat apakah terdapat konsentrasi spasial atau keterkaitan epidemiologis."
            )

    # 4 TIME / EWS
    ews=result.get('ews',{}) if isinstance(result.get('ews',{}),dict) else {}
    rt=result.get('rt',{}) if isinstance(result.get('rt',{}),dict) else {}
    score=ews.get('ews_score',ews.get('score',ews.get('EWS')))
    trend=ews.get('trend_pct',ews.get('trend'))
    metrics=[]
    if score is not None:metrics.append(f"EWS **{float(score):.1f}/100**")
    if trend is not None:metrics.append(f"tren 7-hari **{float(trend):+.1f}%**")
    if rt.get('rt_recent') is not None:metrics.append(f"Rₜ **{float(rt.get('rt_recent')):.2f}**")
    sections.append(
        "### ⚡ 4. PREDIKSI & PERINGATAN DINI\n"
        + (("Indikator saat ini: "+", ".join(metrics)+". ") if metrics else "EWS/Rₜ belum dapat dihitung. ")
        + "Sistem membaca perubahan temporal pada skala harian/mingguan. Jika baseline pembanding tersedia, "
        "perubahan dibandingkan dengan baseline; jika baseline tidak tersedia, sistem tetap menampilkan sinyal observasional "
        "tanpa membuat probabilitas KLB yang tidak tervalidasi."
    )
    if 'Tanggal Sakit' in work.columns and work['Tanggal Sakit'].notna().any():
        dates=work['Tanggal Sakit'].dropna()
        end=dates.max().normalize();cur_start=end-pd.Timedelta(days=6);prev_start=end-pd.Timedelta(days=13);prev_end=end-pd.Timedelta(days=7)
        cur=work[work['Tanggal Sakit'].between(cur_start,end)];prev=work[work['Tanggal Sakit'].between(prev_start,prev_end)]
        delta=((len(cur)-len(prev))/len(prev)*100) if len(prev) else None
        sections.append(
            f"**TIME:** 7 hari terakhir ({cur_start.date()}–{end.date()}) = **{_fmt_int(len(cur))} kasus**; "
            f"7 hari sebelumnya ({prev_start.date()}–{prev_end.date()}) = **{_fmt_int(len(prev))} kasus**; "
            + (f"perubahan **{delta:+.2f}%**." if delta is not None else "baseline 7 hari sebelumnya belum tersedia.")
        )
        daily=work.groupby(work['Tanggal Sakit'].dt.normalize()).size().sort_index()
        if not daily.empty:
            peak_date=daily.idxmax();peak_n=int(daily.max())
            sections.append(f"**Puncak harian:** {peak_date.date()} dengan **{_fmt_int(peak_n)} kasus**. Sinyal perubahan cepat tidak harus menunggu tren bulanan.")

    # 5 RECOMMENDATIONS / RESPONSE
    # Risk score first, because reference recommendation depends on village risk.
    pri=result.get('priority_score')
    if not isinstance(pri,pd.DataFrame) or pri.empty:
        pri=build_priority_score(work,death)
        if not pri.empty: result['priority_score']=pri
    high=pri[pri['Risk Score']>=60] if isinstance(pri,pd.DataFrame) and not pri.empty else pd.DataFrame()
    medium=pri[(pri['Risk Score']>=35)&(pri['Risk Score']<60)] if isinstance(pri,pd.DataFrame) and not pri.empty else pd.DataFrame()
    low=pri[pri['Risk Score']<35] if isinstance(pri,pd.DataFrame) and not pri.empty else pd.DataFrame()

    sections.append("### 🎯 5. REKOMENDASI STRATEGIS AI")
    if deaths>0:
        sections.append(
            "🔴 **Prioritas 0 — Investigasi kematian:** review seluruh kasus meninggal, definisi kasus, onset, diagnosis, severity, "
            "tempat perawatan, keterlambatan mencari pertolongan dan kemungkinan keterkaitan antar-kasus."
        )
    if not high.empty:
        top_names=", ".join(high.iloc[:10,0].astype(str))
        sections.append(
            f"**Prioritas wilayah HIGH:** {top_names}. Wilayah ini menjadi prioritas relatif karena Risk Score ≥60. "
            "Gunakan hasil ini untuk menentukan lokasi penyelidikan dan active case finding sesuai pedoman penyakit."
        )
    sections.extend([
        "**Prioritas investigasi:** lakukan drill-down provinsi → kabupaten/kota → kecamatan → desa/kelurahan, terutama pada lokasi dengan kematian atau Risk Score tinggi.",
        "**Prioritas penemuan kasus:** lakukan active case finding/contact investigation sesuai pedoman penyakit dan temuan penyelidikan lapangan.",
        "**Prioritas faktor risiko:** analisis outcome Penyakit, Severity dan Kasus Meninggal secara terpisah; gunakan OR, CI 95% dan model multivariat bila memenuhi syarat.",
        "**Prioritas tindak lanjut:** setelah intervensi, ukur kembali TIME + PERSON + PLACE untuk melihat perubahan pola dan outcome."
    ])

    # 6 RISK STRATIFICATION + vulnerable profile, kept as a detailed subsection of the report
    if isinstance(pri,pd.DataFrame) and not pri.empty:
        level=pri.columns[0]
        sections.append(
            "### 🗺️ 6. RISK STRATIFICATION PER WILAYAH — PRIORITAS INTERVENSI SPASIAL\n"
            f"Pada level **{level}**, terdapat **{_fmt_int(len(high))} HIGH**, **{_fmt_int(len(medium))} MEDIUM**, "
            f"dan **{_fmt_int(len(low))} LOW** dari **{_fmt_int(len(pri))} wilayah**. "
            "Case Burden Score dan CFR/Fatality Score masing-masing dinormalisasi 0–100; Risk Score = 50% + 50%. "
            "Klasifikasi HIGH ≥60, MEDIUM 35–59, LOW <35."
        )
        if not pri.empty:
            top=pri.iloc[0]
            sections.append(
                f"**Prioritas komposit tertinggi:** {top[level]} — {int(top['Kasus'])} kasus, {int(top['Meninggal'])} meninggal, "
                f"CFR {top['CFR']:.2f}%, Burden Score {top['Burden Score']:.2f}, CFR/Fatality Score {top['CFR Score']:.2f}, "
                f"Risk Score **{top['Risk Score']:.2f} ({top['Risk Level']})**."
            )
        if deaths>0:
            sections.append(
                "**Makna epidemiologis:** satu kematian pada wilayah kecil tetap merupakan sinyal penting dan memicu ALERT merah, "
                "namun Risk Score adalah ukuran prioritas relatif sehingga tidak tepat menyatakan satu kematian 'sama dengan' jumlah kematian tertentu di kota besar."
            )

    vuln=result.get('vulnerable')
    if isinstance(vuln,list) and vuln:
        v=pd.DataFrame(vuln)
        if not v.empty:
            v['CFR (%)']=pd.to_numeric(v.get('CFR (%)',0),errors='coerce')
            top=v.sort_values(['CFR (%)','Total'],ascending=[False,False]).iloc[0]
            rm=top.get('Risk_Multiplier',top.get('Risk Multiplier',None))
            rm_text=f", Risk Multiplier **{float(rm):.2f}×** terhadap baseline" if rm is not None and pd.notna(rm) else ""
            sections.append(
                "### 👥 VULNERABLE POPULATION PROFILING\n"
                f"Profil dengan CFR tertinggi pada stratifikasi yang memenuhi batas minimal adalah **{top.get('Age_Group','-')} + "
                f"{top.get('Pekerjaan','-')} + {top.get('Status Komorbid','-')}**, CFR **{float(top.get('CFR (%)',0)):.2f}%**, "
                f"sample **{_fmt_int(top.get('Total',0))} kasus**{rm_text}. "
                "Profil ini merupakan sinyal stratifikasi dataset; ukuran sampel dan denominator harus diperhatikan sebelum menyimpulkan risiko populasi."
            )
            sections.append(
                "**Targeted response:** prioritaskan review klinis dan epidemiologis pada kelompok ini bila didukung data severity/mortality, "
                "kemudian sesuaikan penemuan kasus, KIE dan kesiapan layanan dengan pedoman penyakit. Rekomendasi klinis spesifik tidak "
                "ditetapkan AI tanpa konteks klinis individual."
            )

    # 7 OUTCOME-SPECIFIC ANALYSIS
    sections.append(
        "### 🧪 OUTCOME-SPECIFIC ANALYSIS\n"
        "Prinsip SI-HIS: **ONE OUTCOME → ONE ANALYTICAL MODEL → ONE EPIDEMIOLOGICAL INTERPRETATION.**\n\n"
        + _outcome_resume(result.get('outcome_analyses',{}))
    )

    # 8 SPATIAL/FORECAST
    spatial=result.get('spatial')
    if isinstance(spatial,pd.DataFrame) and not spatial.empty and 'Cluster' in spatial.columns:
        counts=spatial['Cluster'].value_counts();clusters=counts.drop(index=-1,errors='ignore');noise=int(counts.get(-1,0))
        sections.append(
            f"### 🛰️ SPATIAL INTELLIGENCE\nDBSCAN menemukan **{_fmt_int(len(clusters))} cluster kepadatan** dan "
            f"**{_fmt_int(noise)} titik noise/outlier**. Cluster adalah sinyal kepadatan data, bukan otomatis hotspot statistik "
            "atau sumber penularan. Bila kematian terdeteksi, lokasi kematian harus diperiksa kembali terhadap cluster dan waktu kejadian."
        )
    fc=result.get('forecast')
    if isinstance(fc,dict) and fc.get('forecast'):
        vals=pd.to_numeric(pd.Series(fc.get('forecast',[])),errors='coerce').dropna()
        if not vals.empty:
            sections.append(
                f"### 📈 FORECAST 14 HARI\nModel **{fc.get('model','Holt-Winters')}** memperkirakan sekitar "
                f"**{vals.iloc[0]:.1f} kasus** pada titik awal hingga **{vals.iloc[-1]:.1f} kasus** pada titik akhir. "
                "Forecast adalah estimasi model, bukan jumlah kasus yang pasti terjadi."
            )

    # 9 STRATEGIC RESOURCE MATRIX — preserve reference concept, but make it explicitly configurable/operational.
    sections.append(
        "### 🎯 PRIORITY INTERVENTION MATRIX\n"
        "Alokasi sumber daya dapat dipetakan berdasarkan kombinasi **Impact × Effort** setelah hasil investigasi divalidasi. "
        "Contoh baseline operasional yang dapat dikonfigurasi: **60% HIGH IMPACT, 30% MEDIUM IMPACT, 10% monitoring & evaluasi**. "
        "Proporsi tersebut adalah parameter perencanaan, bukan hasil perhitungan epidemiologis otomatis."
    )
    sections.append(
        "### 📄 KESIMPULAN AI PREDICTION\n"
        "SI-HIS menggabungkan **TIME + PERSON + PLACE**, outcome penyakit/severity/kematian, early warning, spatial intelligence, "
        "risk stratification dan forecasting untuk menghasilkan situational intelligence. **Kematian memicu ALERT merah tanpa menunggu "
        "kepastian KLB**, sementara investigasi epidemiologi digunakan untuk menemukan luas, pola, konsentrasi dan faktor yang perlu ditindaklanjuti. "
        "SI-HIS adalah DSS; keputusan operasional dan penetapan status tetap berada pada otoritas kesehatan."
    )
    sections.append(
        "**Catatan skoring:** Case Burden Score = kasus wilayah / kasus wilayah tertinggi × 100; "
        "CFR/Fatality Score = CFR wilayah / CFR tertinggi × 100; Risk Score = 0,5 × Burden Score + 0,5 × CFR/Fatality Score. "
        "HIGH ≥60, MEDIUM 35–59, LOW <35. Ambang tersebut adalah ruleset prioritas internal SI-HIS, bukan ambang kewaspadaan nasional dan bukan kriteria legal KLB.\n\n"
        "**DISCLAIMER PENTING:** Hasil analisis berfungsi sebagai **Decision Support System (DSS)**. Keputusan operasional tetap berada di bawah wewenang otoritas kesehatan."
    )
    return "\n\n".join(sections)

def spatial_narrative(spatial):
    """Resume DBSCAN yang menjelaskan metode, hasil, interpretasi, dan tindak lanjut."""
    if not isinstance(spatial,pd.DataFrame) or spatial.empty:
        return (
            "### 🧭 Resume Spatial Epidemiology\n\n"
            "Analisis DBSCAN belum dapat dijalankan karena koordinat geografis yang valid belum mencukupi."
        )
    if 'Cluster' not in spatial.columns:
        return "### 🧭 Resume Spatial Epidemiology\n\nKolom hasil cluster DBSCAN belum tersedia."

    d=spatial.copy()
    counts=d['Cluster'].value_counts()
    clusters=counts.drop(index=-1,errors='ignore').sort_index()
    noise=int(counts.get(-1,0))
    total=len(d)
    clustered=int(clusters.sum())
    clustered_pct=(clustered/total*100) if total else 0
    noise_pct=(noise/total*100) if total else 0

    # Profil tiap cluster bila atribut kasus tersedia.
    profiles=[]
    for cid,g in d[d['Cluster']!=-1].groupby('Cluster'):
        n=len(g)
        lat=pd.to_numeric(g.get('Latitude',pd.Series(dtype=float)),errors='coerce').mean()
        lon=pd.to_numeric(g.get('Longitude',pd.Series(dtype=float)),errors='coerce').mean()
        mean_age=pd.to_numeric(g.get('Umur',pd.Series(dtype=float)),errors='coerce').mean()
        deaths=_death_series_app(g).sum() if 'Is_Meninggal' in g.columns or 'Status Penderita' in g.columns else 0
        cfr=deaths/n*100 if n else 0
        density=None
        if 'Cluster_Radius_KM' in g.columns:
            radius=pd.to_numeric(g['Cluster_Radius_KM'],errors='coerce').max()
            if pd.notna(radius) and radius>0:
                density=n/(3.14159*radius*radius)
        place='-'
        for col in ['Desa/Kelurahan','Kecamatan','Kabupaten','Provinsi']:
            if col in g.columns and not g[col].dropna().empty:
                place=str(g[col].mode().iloc[0])
                break
        profiles.append({'cluster':int(cid),'n':n,'lat':lat,'lon':lon,'age':mean_age,'deaths':int(deaths),'cfr':cfr,'density':density,'place':place})
    profiles.sort(key=lambda x:x['n'],reverse=True)
    top=profiles[0] if profiles else None
    high_density=sum(1 for x in profiles if x['density'] is not None and x['density']>5)
    outlier_label="tidak ada titik noise" if noise==0 else f"{noise:,} titik noise/outlier"

    lines=[
        "### 🧭 Resume Spatial Epidemiology",
        "",
        "### 🔬 Apa itu DBSCAN?",
        "**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** adalah algoritma clustering berbasis kepadatan. "
        "Dalam SI-HIS, DBSCAN mengelompokkan titik kasus yang secara geografis berdekatan dan memiliki kepadatan yang cukup menurut parameter "
        "**eps** (jarak maksimum antar titik yang dianggap berdekatan) dan **min_samples** (jumlah minimum titik untuk membentuk kepadatan/cluster). "
        "Titik yang tidak memenuhi kepadatan tersebut ditandai sebagai **noise/outlier (-1)**.",
        "",
        "**Cara membaca hasil:** cluster menunjukkan konsentrasi titik kasus dalam ruang berdasarkan parameter algoritma; "
        "DBSCAN **tidak membuktikan sumber penularan, arah transmisi, maupun hubungan kausal**. Karena itu istilah *epicenter* pada dashboard harus dipahami "
        "sebagai **pusat konsentrasi kasus/cluster prioritas**, bukan otomatis sumber wabah.",
        "",
        "### 📊 Hasil Utama DBSCAN",
        f"- **Cluster terdeteksi:** **{len(profiles):,}**.",
        f"- **Kasus/titik dalam cluster:** **{clustered:,} ({clustered_pct:.1f}%)**.",
        f"- **Noise/outlier:** **{noise:,} ({noise_pct:.1f}%)** — {outlier_label}.",
        f"- **Cluster kepadatan >5 kasus/km²:** **{high_density:,}**"
    ]
    if top:
        lines += [
            "",
            "### 📍 Epicenter/Cluster Prioritas",
            f"- **Cluster terbesar:** **#{top['cluster']}** di sekitar **{top['place']}**, dengan **{top['n']:,} kasus**.",
            (f"- Rata-rata usia: **{top['age']:.1f} tahun**." if pd.notna(top['age']) else "- Rata-rata usia: data tidak tersedia."),
            f"- Kematian: **{top['deaths']:,}**; CFR klaster: **{top['cfr']:.2f}%**.",
            (f"- Perkiraan kepadatan: **{top['density']:.2f} kasus/km²**." if top['density'] is not None else "- Kepadatan kasus/km² belum dapat dihitung dari output DBSCAN yang tersedia.")
        ]
    lines += [
        "",
        "### 🧬 Analisis Pola Transmisi Berbasis Klaster",
        "DBSCAN dapat membantu mengenali pola konsentrasi spasial yang perlu ditelusuri. Namun, label **Point Source, Propagated-like, atau Mixed** "
        "tidak boleh ditetapkan hanya dari jumlah cluster. Penentuan pola transmisi harus menggabungkan **TIME + PLACE + riwayat paparan/perjalanan + "
        "kontak epidemiologis + karakteristik penyakit**.",
        "",
        "### 🎯 Panduan Interpretasi",
        "1. **Cluster/epicenter:** semakin banyak cluster yang terpisah, semakin banyak area konsentrasi yang perlu ditelusuri; bukan berarti otomatis penyebaran luas.",
        "2. **Kasus berklaster vs outlier:** proporsi kasus berklaster menunjukkan berapa banyak titik berada dalam konsentrasi spasial yang terdeteksi. Outlier perlu diperiksa sebagai kemungkinan kasus yang terisolasi, lokasi dengan kepadatan rendah, atau introduksi—status impor tidak dapat disimpulkan tanpa riwayat perjalanan.",
        "3. **Kepadatan:** angka kasus/km² hanya bermakna bila luas area cluster/radius dihitung dengan metode yang konsisten. Ambang >5 kasus/km² dapat dipakai sebagai ruleset operasional SI-HIS bila dikonfigurasi, bukan ambang epidemiologi universal.",
        "4. **Radius cluster:** radius kecil dengan kepadatan tinggi menunjukkan konsentrasi geografis yang lebih sempit; radius besar menunjukkan sebaran geografis lebih luas. Keduanya tetap perlu dikaitkan dengan waktu dan pola paparan.",
        "5. **Pola transmisi:** Point Source mengarah pada hipotesis paparan bersama; propagated-like dapat konsisten dengan transmisi berantai; Mixed berarti terdapat kombinasi pola yang perlu dibedakan melalui investigasi. Semua adalah hipotesis epidemiologis, bukan bukti tunggal dari DBSCAN.",
        "",
        "### 🚨 Rekomendasi Intervensi Berbasis Spasial",
        "### Aksi Prioritas 24–72 Jam",
        "1. Fokuskan peninjauan lapangan pada cluster dengan beban kasus/kematian tertinggi.",
        "2. Lakukan active case finding dan investigasi lingkungan pada radius operasional yang ditetapkan petugas (misalnya **2 km**) bila sesuai karakteristik penyakit dan protokol setempat; angka 2 km bukan aturan universal.",
        "3. Telusuri riwayat perjalanan, kontak, tempat tinggal, tempat kerja/sekolah, dan paparan bersama untuk membedakan cluster lokal dari kasus yang terpisah.",
        "4. Jika terdapat kematian, prioritaskan verifikasi setiap lokasi kematian terhadap waktu kejadian dan cluster.",
        "5. Ulangi DBSCAN secara berkala, misalnya **3–7 hari**, untuk melihat apakah cluster membesar, mengecil, berpindah, atau terbentuk cluster baru.",
        "",
        "### 🔴 Catatan Penting untuk Kemenkes/Dinkes",
        "Hasil DBSCAN pada dashboard harus dapat ditelusuri kembali ke **datasheet kasus yang dianalisis**. Karena itu, cluster, outlier, lokasi dan metrik yang ditampilkan harus dapat diverifikasi dari baris data sumber. "
        "DBSCAN merupakan alat *pattern detection* untuk membantu menentukan prioritas investigasi; konfirmasi epidemiologis tetap memerlukan validasi data dan investigasi lapangan."
    ]
    return "\n".join(lines)

def curve_narrative(curve,disease):
    if not isinstance(curve,(tuple,list)) or len(curve)<4:return 'Klasifikasi kurva belum tersedia.'
    label,short,meaning,implication=curve[:4];return f'**Klasifikasi: {label}.** {short} **Makna:** {meaning} **Implikasi:** {implication} Pada {disease}, bentuk kurva tidak boleh digunakan sendirian untuk menyimpulkan mekanisme transmisi.'

def forecast_render(fc):
    if not isinstance(fc,dict):render_value(fc);return
    table=pd.DataFrame({'Tanggal':pd.to_datetime(fc.get('dates',[]),errors='coerce'),'Forecast':fc.get('forecast',[]),'Lower 95%':fc.get('lower',[]),'Upper 95%':fc.get('upper',[])});st.caption(f"Model **{fc.get('model','Holt-Winters')}** memperkirakan tren **{fc.get('trend','-')}**. Forecast adalah estimasi model dan intervalnya menunjukkan ketidakpastian, bukan jumlah kasus yang pasti terjadi.")
    if not table.empty:st.line_chart(table.set_index('Tanggal')[['Forecast','Lower 95%','Upper 95%']]);st.dataframe(table,use_container_width=True,hide_index=True)

def vulnerable_narrative(v):
    if not isinstance(v,list) or not v:return 'Belum ditemukan profil populasi rentan yang memenuhi batas minimal analisis.'
    d=pd.DataFrame(v);top=d.iloc[0];return f"Metode menggunakan stratifikasi **Kelompok Umur × Pekerjaan × Status Komorbid**, lalu menghitung kasus, kematian, tingkat kematian akibat penyakit (CFR) dan Risk Multiplier terhadap baseline. Profil teratas: **{top.get('Age_Group','-')} × {top.get('Pekerjaan','-')} × {top.get('Status Komorbid','-')}**, n={int(top.get('Total',0))}, CFR={float(top.get('CFR (%)',0)):.2f}%. Strata kecil harus ditafsirkan hati-hati."

def ml_narrative(ml):
    if not isinstance(ml,dict) or not ml:return 'ML layer belum dijalankan.'
    return ('**SI-HIS menggunakan 5 lapisan ML/AI dengan fungsi yang berbeda:** '
            'prediksi severity, prediksi outbreak 7 hari, prediksi spasial, prediksi populasi rentan, '
            'dan forecasting temporal. Setiap model harus dibaca melalui target, data, metrik, '
            'feature importance, keterbatasan, dan status validasinya. Objek Pipeline tidak ditampilkan sebagai hasil analisis.')

def _metric_value(v, digits=3):
    try:
        if v is None or not np.isfinite(float(v)): return 'N/A'
        return f'{float(v):.{digits}f}'
    except Exception:
        return 'N/A'

def _render_ml_model_card(title, purpose, decision_question, result):
    st.markdown(f'### {title}')
    st.markdown(f'**Fungsi:** {purpose}')
    st.markdown(f'**Pertanyaan keputusan:** {decision_question}')
    if not isinstance(result,dict):
        st.info('Hasil model belum tersedia.')
        return
    if result.get('status') != 'ok':
        st.warning(f"Model belum menghasilkan metrik yang valid: {result.get('message','Data belum mencukupi.')}")
        return
    metrics=result.get('metrics',{})
    m1,m2,m3,m4=st.columns(4)
    m1.metric('ROC-AUC',_metric_value(metrics.get('roc_auc')))
    m2.metric('PR-AUC',_metric_value(metrics.get('pr_auc')))
    m3.metric('Recall',_metric_value(metrics.get('recall')))
    m4.metric('Specificity',_metric_value(metrics.get('specificity')))
    m5,m6,m7,m8=st.columns(4)
    m5.metric('Precision',_metric_value(metrics.get('precision')))
    m6.metric('F1',_metric_value(metrics.get('f1')))
    m7.metric('Brier',_metric_value(metrics.get('brier')))
    m8.metric('Accuracy',_metric_value(metrics.get('accuracy')))
    st.markdown(f"**Cara membaca:** recall={_metric_value(metrics.get('recall'))} menunjukkan proporsi outcome positif yang tertangkap; specificity={_metric_value(metrics.get('specificity'))} menunjukkan proporsi non-target yang tersaring; PR-AUC={_metric_value(metrics.get('pr_auc'))} penting ketika outcome positif relatif jarang. Brier={_metric_value(metrics.get('brier'))} menilai kualitas probabilitas, umumnya semakin kecil semakin baik.")
    if result.get('train_rows') is not None or result.get('test_rows') is not None:
        st.caption(f"Validasi: holdout temporal | training={result.get('train_rows','N/A'):,} | holdout={result.get('test_rows','N/A'):,} | positive rate={float(result.get('positive_rate',0))*100:.1f}%")
    fi=result.get('feature_importance')
    if isinstance(fi,pd.DataFrame) and not fi.empty:
        st.markdown('**Feature importance — kontribusi prediktif, bukan kausalitas**')
        st.dataframe(fi.head(10),use_container_width=True,hide_index=True)
    with st.expander('🔧 Detail teknis model',expanded=False):
        model=result.get('model')
        if model is not None:
            st.write(f'Model terlatih: {type(model).__name__}')
            st.caption('Konfigurasi pipeline disimpan di backend dan tidak diperlakukan sebagai hasil epidemiologi.')
        st.write(f"Jumlah fitur: {len(result.get('features',[]))}")
        st.write(f"Fitur: {', '.join(map(str,result.get('features',[])))}")
        if result.get('holdout_start') is not None: st.write(f"Awal data holdout temporal: {result.get('holdout_start')}")
    if result.get('metrics',{}).get('Narrative'):
        st.info(result['metrics']['Narrative'])

def _render_ml_forecast(result):
    st.markdown('### 5. Temporal Forecasting')
    st.markdown('**Fungsi:** memproyeksikan jumlah kasus ke depan untuk membantu membaca kemungkinan beban layanan, kesiapsiagaan, dan arah tren.')
    st.markdown('**Pertanyaan keputusan:** berapa kisaran kasus yang mungkin terjadi dalam horizon forecast dan kapan puncak model diperkirakan?')
    if not isinstance(result,dict) or result.get('status')!='ok':
        st.warning(f"Forecast belum tersedia: {result.get('message','Data belum mencukupi.') if isinstance(result,dict) else 'hasil tidak tersedia.'}")
        return
    m1,m2,m3=st.columns(3)
    m1.metric('Horizon',f"{len(result.get('forecast',[]))} hari")
    m2.metric('Peak forecast',f"{float(result.get('peak_value',0)):.1f} kasus")
    peak=result.get('peak_date');m3.metric('Tanggal peak',pd.to_datetime(peak).strftime('%d %b %Y') if peak is not None else 'N/A')
    st.markdown(f"**Backtest MAE:** {_metric_value(result.get('mae'),2)} | **RMSE:** {_metric_value(result.get('rmse'),2)}")
    weights=result.get('weights',{})
    if weights: st.write('Bobot ensemble:',{k:round(float(v),3) for k,v in weights.items()})
    dates=pd.to_datetime(result.get('dates',[]),errors='coerce')
    forecast=result.get('forecast',[])
    if len(dates)==len(forecast) and len(forecast):
        chart=pd.DataFrame({'Tanggal':dates,'Forecast':forecast}).set_index('Tanggal')
        st.line_chart(chart)
    with st.expander('📖 Cara membaca forecasting',expanded=False):
        st.markdown('Forecast adalah estimasi model, bukan jumlah kasus yang pasti terjadi. MAE/RMSE menggambarkan kesalahan pada backtest; semakin kecil umumnya semakin baik. Puncak forecast adalah nilai tertinggi yang diproyeksikan dalam horizon. Gunakan bersama kurva epidemik, EWS, Rₜ, dan konteks intervensi.')

def render_ml_report(ml):
    st.markdown('### 🧠 AI/ML Intelligence Report')
    st.caption('Lima lapisan ML/AI SI-HIS. Setiap lapisan memiliki target dan fungsi berbeda. Model teknis tidak ditampilkan sebagai pengganti hasil analisis.')
    if not isinstance(ml,dict) or not ml:
        st.info('ML layer belum dijalankan.')
        return
    _render_ml_model_card(
        '1. Case Severity Prediction',
        'Memprediksi probabilitas kasus masuk outcome severity yang didefinisikan sistem. Pada implementasi saat ini, target demo severity diturunkan dari kematian atau status rawat inap.',
        'Kasus mana yang perlu mendapat prioritas pemantauan/triase berdasarkan probabilitas outcome model?',
        ml.get('case_severity'))
    _render_ml_model_card(
        '2. KLB / Outbreak 7-Day Prediction',
        'Memprediksi apakah beban kasus 7 hari berikutnya pada desa mencapai threshold turunan dari distribusi historis. Threshold ini adalah threshold model, bukan definisi legal KLB.',
        'Desa mana yang perlu dipantau lebih dini karena sinyal temporalnya mengarah ke peningkatan beban 7 hari?',
        ml.get('klb'))
    _render_ml_model_card(
        '3. Spatial Outbreak Prediction',
        'Menggabungkan sinyal temporal dan lokasi untuk memperkirakan area yang berpotensi mengalami peningkatan outbreak pada horizon berikutnya.',
        'Wilayah mana yang perlu diprioritaskan untuk verifikasi spasial dan investigasi?',
        ml.get('spatial'))
    _render_ml_model_card(
        '4. Vulnerable Population Prediction',
        'Memprediksi outcome rentan berdasarkan karakteristik person dan paparan. Hasilnya dipakai untuk surveillance/population prioritization, bukan diagnosis individual.',
        'Kelompok atau karakteristik kasus mana yang perlu mendapat perhatian surveillance lebih lanjut?',
        ml.get('vulnerable'))
    _render_ml_forecast(ml.get('forecast'))
    st.markdown('### 📌 Cara Membaca Hasil ML secara Keseluruhan')
    st.markdown('1. **Mulai dari target:** pahami apa yang sebenarnya diprediksi.\n2. **Lihat holdout temporal:** jangan hanya melihat training performance.\n3. **Perhatikan PR-AUC, recall, precision, specificity dan Brier**, terutama bila outcome positif jarang.\n4. **Baca feature importance sebagai sinyal prediktif**, bukan faktor penyebab.\n5. **Hubungkan prediction dengan TIME + PERSON + PLACE dan outcome epidemiologis** sebelum membuat keputusan.\n6. **Validasi eksternal diperlukan** sebelum model digunakan sebagai dasar keputusan operasional.')

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
    # Navigation Bar utama tetap horizontal di atas. Setiap tab memiliki resume sendiri tepat di bawah tab.
    tabs=st.tabs(['🚨 AI Prediction & Recommendation','📊 Trias Epidemiologi (Detail)','📈 Kurva Epidemik & Prediksi','🗺️ Peta Spasial & AI DBSCAN','🧪 Analisis Faktor Risiko','🤖 Analisis ML'])
    with tabs[0]:
        st.markdown('### 🚨 AI Prediction & Recommendation')
        resume_text=ai_prediction_narrative(result,sel_disease,label)
        analyzed_df=result.get('analysis_dataframe')
        if not isinstance(analyzed_df,pd.DataFrame) or analyzed_df.empty:
            analyzed_df=df_raw.copy()
        export_meta=build_export_metadata(analyzed_df,sel_disease,label,'Simulasi' if source.startswith('Gunakan') else getattr(uploaded,'name','Upload'),sel_prov,sel_kab,sel_kec,sel_desa,sel_pusk)
        pdf_bytes=build_resume_pdf(f'SI-HIS Resume Epidemiologi — {sel_disease}',resume_text,export_meta)
        xlsx_bytes=build_evaluation_excel(analyzed_df,export_meta)
        csv_bytes=build_evaluation_csv(analyzed_df)
        st.markdown('#### 📥 Paket Verifikasi & Konfirmasi')
        st.caption('PDF adalah hardcopy resume analisis. Datasheet adalah baris data yang benar-benar dianalisis pada scope ini, sehingga tim Kemenkes/Dinkes dapat melakukan verifikasi dan konfirmasi langsung tanpa meminta ulang datasheet ke Dinkes sumber.')
        c1,c2,c3=st.columns(3)
        c1.download_button('📄 Download Resume PDF',pdf_bytes,f'SI-HIS_Resume_{sel_disease}_{label}.pdf','application/pdf',use_container_width=True)
        c2.download_button('📊 Download Datasheet Excel',xlsx_bytes,f'SI-HIS_Datasheet_Verifikasi_{sel_disease}_{label}.xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',use_container_width=True)
        c3.download_button('📋 Download Datasheet CSV',csv_bytes,f'SI-HIS_Datasheet_Verifikasi_{sel_disease}_{label}.csv','text/csv',use_container_width=True)
        st.info(f"🔐 **SHA-256 datasheet:** `{export_meta['SHA-256 Datasheet']}` — gunakan nilai ini untuk memastikan datasheet yang diverifikasi identik dengan data yang dianalisis.")
        show_tab_resume('Resume AI Prediction & Recommendation',resume_text,'alert, EWS, prediction, scoring dan rekomendasi di bawah')
        klb=result.get('klb',{})
        if isinstance(klb,dict):
            status=klb.get('status',klb.get('level',klb.get('classification','')))
            if status:
                s=str(status).upper()
                if 'ALERT' in s or 'INVESTIGATION' in s or 'KLB' in s: st.error(f'🔴 **ALERT SURVEILANS: {status}**')
                elif 'WARNING' in s: st.warning(f'🟡 **EARLY WARNING: {status}**')
                else: st.info(f'**Status surveilans:** {status}')
        render_value(result.get('ews'),'Indikator EWS');render_value(result.get('rt'),'Rₜ')
        st.markdown('### Rekomendasi AI untuk investigasi dan pencegahan')
        rec=result.get('recommendations',result.get('recommendation'))
        if rec: render_value(rec)
        else: show_resume('SI-HIS mengarahkan tindakan berdasarkan sinyal yang terdeteksi: penyelidikan epidemiologi, penemuan kasus aktif, penelusuran kontak, analisis TIME + PERSON + PLACE, analisis cluster/episentrum, serta pengendalian faktor risiko sesuai penyakit dan temuan lapangan.')
    with tabs[1]:
        st.markdown('### 📊 Trias Epidemiologi (Detail)')
        show_tab_resume('Resume TIME + PERSON + PLACE + OUTCOME',epidemiology_master_narrative(result,label,sel_disease),'tabel/grafik Trias dan scoring di bawah')
        if isinstance(result.get('priority_score'),pd.DataFrame):
            st.markdown('#### 🎯 Prioritas Epidemiologis — Burden Score & CFR Score')
            st.dataframe(result['priority_score'],use_container_width=True,hide_index=True)
            render_scoring_explanation()
        tri=result.get('trias_summary',{})
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
        show_tab_resume('Resume Kurva Epidemik & Prediksi',curve_narrative(result.get('epidemic_curve_classification'),sel_disease),'kurva, gelombang, forecast dan Rₜ di bawah')
        t=result.get('time')
        if isinstance(t,pd.DataFrame) and not t.empty:
            chart=t.copy();chart['Tanggal Sakit']=pd.to_datetime(chart['Tanggal Sakit'],errors='coerce');valid_chart=chart.dropna(subset=['Tanggal Sakit'])
            if not valid_chart.empty:st.line_chart(valid_chart.set_index('Tanggal Sakit')['Jumlah Kasus'])
        st.markdown('### Forecast 14 Hari');forecast_render(result.get('forecast'));waves=result.get('waves')
        if waves:render_value(waves,'Gelombang Temporal')
    with tabs[3]:
        st.markdown('### 🗺️ Peta Spasial & AI DBSCAN')
        spatial=result.get('spatial')
        show_tab_resume('Resume Spatial Epidemiology',spatial_narrative(spatial),'peta, cluster dan episentrum di bawah')
        st.caption('Analisis spasial, cluster dan episentrum digunakan untuk memperkuat interpretasi PLACE pada resume epidemiologi.');
        render_value(spatial,'Hasil DBSCAN')
        if isinstance(spatial,pd.DataFrame) and not spatial.empty and {'Latitude','Longitude'}.issubset(spatial.columns):
            geo=spatial.dropna(subset=['Latitude','Longitude'])
            if not geo.empty:
                m=folium.Map(location=[float(geo.Latitude.mean()),float(geo.Longitude.mean())],zoom_start=9)
                for _,row in geo.head(500).iterrows():folium.CircleMarker([float(row.Latitude),float(row.Longitude)],radius=4,popup=f"{row.get('Desa/Kelurahan','')} | Cluster {row.get('Cluster','')}").add_to(m)
                st_folium(m,width=None,height=500)
        render_value(result.get('epicenters'),'Episentrum / Titik Prioritas')
        st.caption('Cluster adalah hasil kepadatan spasial. Episentrum adalah lokasi konsentrasi prioritas dalam dataset; keduanya bukan otomatis sumber penularan dan harus dikaitkan dengan TIME, hubungan epidemiologis dan investigasi lapangan.')
    with tabs[4]:
        st.markdown('### 🧪 Analisis Faktor Risiko')
        show_tab_resume('Resume Analisis Faktor Risiko',risk_narrative(result.get('risk_factors')),'tabel bivariat, multivariat dan outcome di bawah')
        st.markdown('#### Detail Analisis Faktor Risiko')
        s=risk_summary(result.get('risk_factors'))
        if not s.empty:st.dataframe(s,use_container_width=True,hide_index=True)
        outcomes=result.get('outcome_analyses',{})
        if isinstance(outcomes,dict):
            for outcome_name,outcome in outcomes.items():
                with st.expander(f'Outcome: {outcome_name}',expanded=(outcome_name=='Penyakit')):render_value(outcome)
        render_risk_factors(result.get('risk_factors'))
    with tabs[5]:
        st.markdown('### 🤖 Analisis ML')
        show_tab_resume('Resume Machine Learning',ml_narrative(result.get('ml')),'performa model, feature importance dan hasil ML di bawah')
        st.caption('Analisis ML mendukung decision support dan melengkapi analisis epidemiologi, bukan menggantikannya.')
        if include_ml:render_value(result.get('ml'))
        else:st.info('ML layer belum diaktifkan. Centang **Aktifkan ML layer** pada Panel Kontrol untuk menjalankan prediction models.')

st.caption('SI-HIS Intelligence — epidemiological decision-support with TIME + PERSON + PLACE.')
