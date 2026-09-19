"""Expert narrative layer for SI-HIS Intelligence.

Transforms analytical outputs into structured epidemiological explanations for:
1) academic/technical users,
2) operational practitioners, and
3) the general public.

This layer interprets existing results; it does not invent missing data or replace
clinical/epidemiological judgment.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd


def _num(v, default=0.0):
    try:
        x = float(v)
        return default if not math.isfinite(x) else x
    except Exception:
        return default


def _fmt(v, digits=1):
    x = _num(v)
    return f"{x:,.{digits}f}"


def _pct(n, d):
    return (_num(n) / _num(d) * 100.0) if _num(d) else 0.0


def _first_valid(values):
    for v in values:
        if v is not None and str(v).strip() not in {"", "nan", "None", "N/A"}:
            return v
    return None


def _death_series(df):
    if not isinstance(df, pd.DataFrame):
        return pd.Series(dtype=float)
    s = pd.Series(0.0, index=df.index)
    if "Is_Meninggal" in df.columns:
        s = pd.to_numeric(df["Is_Meninggal"], errors="coerce").fillna(0).astype(float)
    if "Status Penderita" in df.columns:
        m = df["Status Penderita"].astype(str).str.strip().str.lower().eq("meninggal").astype(float)
        s = pd.Series(np.maximum(s, m), index=df.index)
    return s


def _top_category(df, column):
    if not isinstance(df, pd.DataFrame) or column not in df.columns:
        return None, 0, 0.0
    s = df[column].astype(str).replace({"nan": "Tidak diketahui"}).value_counts()
    if s.empty:
        return None, 0, 0.0
    n = int(s.iloc[0])
    return str(s.index[0]), n, _pct(n, len(df))


def _date_window(df):
    if not isinstance(df, pd.DataFrame) or "Tanggal Sakit" not in df.columns:
        return None, None
    d = pd.to_datetime(df["Tanggal Sakit"], errors="coerce").dropna()
    if d.empty:
        return None, None
    return d.min().normalize(), d.max().normalize()


def descriptive_expert(result, label):
    ov = result.get("overview", {}) if isinstance(result, dict) else {}
    total = int(_num(ov.get("total_cases")))
    deaths = int(_num(ov.get("deaths")))
    top = result.get("top10_diseases") if isinstance(result, dict) else None
    disease = cases = cfr = None
    if isinstance(top, pd.DataFrame) and not top.empty:
        r = top.iloc[0]
        disease = str(r.get("Nama Penyakit", "-"))
        cases = int(_num(r.get("Jumlah Kasus")))
        cfr = _num(r.get("CFR"))

    p, pn, pp = _top_category(result.get("province_distribution"), "Provinsi") if isinstance(result.get("province_distribution"), pd.DataFrame) else (None, 0, 0)
    sex, sn, sp = _top_category(result.get("sex_distribution"), "Jenis Kelamin") if isinstance(result.get("sex_distribution"), pd.DataFrame) else (None, 0, 0)
    age, an, ap = _top_category(result.get("age_distribution"), "Kelompok Umur") if isinstance(result.get("age_distribution"), pd.DataFrame) else (None, 0, 0)

    start = end = None
    source_df = result.get("source_dataframe") if isinstance(result, dict) else None
    start, end = _date_window(source_df)

    lines = [
        f"### 🧭 RESUME ANALISIS DESKRIPTIF — {label}",
        "",
        f"**Gambaran utama.** Scope **{label}** memuat **{total:,} baris kasus/kunjungan** dan **{deaths:,} kematian**. "
        "Analisis deskriptif menjawab pertanyaan dasar epidemiologi: *siapa yang terdampak, di mana kasus terkonsentrasi, kapan kasus terjadi, dan penyakit apa yang paling banyak ditemukan*. "
        "Hasil ini menggambarkan beban dalam dataset, bukan otomatis menggambarkan risiko populasi.",
    ]
    if disease is not None:
        lines.append(
            f"**Beban penyakit.** Penyakit dengan jumlah kasus terbesar adalah **{disease}** dengan **{cases:,} kasus**. "
            f"CFR yang tercatat untuk penyakit tersebut adalah **{cfr:.2f}%**. CFR adalah proporsi kematian di antara kasus penyakit, "
            "sedangkan *mortality rate* membutuhkan denominator jumlah penduduk/populasi berisiko."
        )
    if p is not None:
        lines.append(
            f"**PLACE.** Konsentrasi jumlah kasus terbesar berada di **{p}** ({pn:,} kasus; {pp:.1f}% dari distribusi wilayah yang ditampilkan). "
            "Konsentrasi absolut tidak sama dengan risiko tertinggi karena ukuran populasi tiap wilayah dapat berbeda."
        )
    if sex is not None:
        lines.append(f"**PERSON — jenis kelamin.** Kategori terbanyak adalah **{sex}**, {sn:,} kasus ({sp:.1f}%).")
    if age is not None:
        lines.append(f"**PERSON — umur.** Kelompok umur terbanyak pada tabel distribusi adalah **{age}**, {an:,} kasus ({ap:.1f}%).")
    if start is not None:
        lines.append(f"**TIME.** Rentang tanggal yang dapat dibaca dari data adalah **{start.date()} sampai {end.date()}**.")
    lines += [
        "",
        "### 🔬 Penjelasan akademik/praktisi",
        "Analisis ini bersifat **deskriptif**, sehingga belum menjawab hubungan sebab-akibat. Untuk menyimpulkan risiko diperlukan denominator populasi, "
        "definisi kasus yang konsisten, kualitas data, stratifikasi yang tepat, serta analisis inferensial atau pemodelan sesuai pertanyaan penelitian.",
        "",
        "### 👥 Penjelasan untuk masyarakat",
        "Angka pada dashboard adalah cara untuk melihat **berapa banyak kasus, siapa yang paling banyak tercatat, dan wilayah mana yang memiliki beban kasus lebih besar**. "
        "Wilayah dengan angka kasus paling besar belum tentu merupakan wilayah dengan kemungkinan seseorang sakit paling tinggi; untuk itu kita perlu mengetahui jumlah penduduk dan faktor lain yang memengaruhi risiko.",
        "",
        "### ⚠️ Batas interpretasi",
        "Data hilang, perbedaan akses pelayanan, perubahan definisi kasus, keterlambatan pelaporan, duplikasi, dan perbedaan jumlah penduduk dapat mengubah pola yang terlihat. "
        "Karena itu hasil deskriptif harus menjadi dasar pertanyaan epidemiologis berikutnya, bukan satu-satunya dasar keputusan."
    ]
    return "\n".join(lines)


def epidemiology_expert(result, label, disease):
    df = result.get("analysis_dataframe") if isinstance(result, dict) else None
    if not isinstance(df, pd.DataFrame) or df.empty:
        return "### 🧭 RESUME EPIDEMIOLOGI\n\nData kasus pada scope ini belum tersedia."
    total = len(df)
    death = _death_series(df)
    deaths = int(death.sum())
    cfr = _pct(deaths, total)
    start, end = _date_window(df)

    parts = [
        f"### 🧭 RESUME EPIDEMIOLOGI — {disease} | {label}",
        "",
        f"**Kesimpulan situasi.** Dalam scope ini terdapat **{total:,} kasus {disease}** dan **{deaths:,} kematian**, dengan **CFR {cfr:.2f}%**. "
        "Interpretasi utama dibangun melalui **TIME + PERSON + PLACE + OUTCOME**, bukan dari satu angka tunggal.",
    ]

    if start is not None:
        d = pd.to_datetime(df["Tanggal Sakit"], errors="coerce")
        end = d.max().normalize()
        cur_start = end - pd.Timedelta(days=6)
        prev_start = end - pd.Timedelta(days=13)
        prev_end = end - pd.Timedelta(days=7)
        cur = df[d.between(cur_start, end)]
        prev = df[d.between(prev_start, prev_end)]
        delta = None if len(prev) == 0 else (len(cur) - len(prev)) / len(prev) * 100
        direction = "meningkat" if len(cur) > len(prev) else ("menurun" if len(cur) < len(prev) else "relatif tetap")
        parts.append(
            f"**TIME.** Tujuh hari terakhir ({cur_start.date()}–{end.date()}) mencatat **{len(cur):,} kasus**, "
            f"sedangkan tujuh hari sebelumnya ({prev_start.date()}–{prev_end.date()}) mencatat **{len(prev):,} kasus**. "
            f"Secara deskriptif, beban kasus **{direction}**" +
            (f" sebesar **{delta:.1f}%** dibandingkan minggu sebelumnya." if delta is not None else "; baseline pembanding belum tersedia.")
        )

    age, an, ap = _top_category(df, "Umur")
    if age is not None:
        # For numeric age, replace the generic category result with median/mean.
        a = pd.to_numeric(df["Umur"], errors="coerce").dropna()
        if not a.empty:
            parts.append(
                f"**PERSON.** Usia rata-rata **{a.mean():.1f} tahun**, median **{a.median():.1f} tahun**. "
                f"Kelompok umur dengan jumlah kasus terbanyak dalam distribusi adalah **{age}** ({an:,} kasus; {ap:.1f}%)."
            )
    sex, sn, sp = _top_category(df, "Jenis Kelamin")
    if sex is not None:
        parts.append(f"Distribusi jenis kelamin terbesar adalah **{sex}** ({sn:,} kasus; {sp:.1f}%).")

    for col, label_col in [("Provinsi", "provinsi"), ("Kabupaten", "kabupaten/kota"), ("Kecamatan", "kecamatan")]:
        name, n, pct = _top_category(df, col)
        if name is not None:
            parts.append(f"**PLACE.** Beban kasus terbesar pada tingkat {label_col} yang tersedia adalah **{name}** ({n:,} kasus; {pct:.1f}%).")
            break

    rt = result.get("rt")
    if isinstance(rt, dict):
        rv = _first_valid([rt.get("rt"), rt.get("Rt"), rt.get("R_t"), rt.get("estimate")])
        if rv is not None:
            parts.append(f"**TRANSMISSION SIGNAL.** Estimasi Rₜ yang tersedia adalah **{_fmt(rv,2)}**. Rₜ harus dibaca bersama metode estimasi, ketidakpastian, kualitas tanggal onset, dan asumsi model; angka tunggal tidak membuktikan transmisi meningkat atau menurun.")

    parts += [
        "",
        "### 🔬 Penjelasan akademik/praktisi",
        "Trias epidemiologi digunakan untuk menghubungkan **Person–Place–Time** dengan outcome. Perubahan jumlah kasus dapat berasal dari perubahan transmisi, "
        "perubahan perilaku mencari layanan, intensitas testing, pelaporan, definisi kasus, atau keterlambatan pencatatan. Karena itu interpretasi temporal harus dibandingkan dengan konteks surveilans.",
        "",
        "### 👥 Penjelasan untuk masyarakat",
        "Bagian ini mencoba menjawab tiga pertanyaan sederhana: **siapa yang paling banyak terkena, di mana kasus terkonsentrasi, dan kapan kasus meningkat atau menurun**. "
        "Jika salah satu angka berubah, kita belum boleh langsung mengatakan penyebabnya; perubahan itu perlu diperiksa dengan kondisi lapangan dan kualitas pelaporan.",
        "",
        "### ⚠️ Hal yang perlu diverifikasi",
        "Verifikasi definisi kasus, tanggal onset versus tanggal pelaporan, duplikasi, kelengkapan kematian, denominator populasi, dan perubahan sistem pencatatan sebelum hasil digunakan untuk keputusan operasional."
    ]
    return "\n".join(parts)


def curve_expert(result, disease):
    curve = result.get("epidemic_curve_classification") if isinstance(result, dict) else None
    t = result.get("time") if isinstance(result, dict) else None
    forecast = result.get("forecast") if isinstance(result, dict) else None
    waves = result.get("waves") if isinstance(result, dict) else None

    lines = [
        f"### 📈 RESUME KURVA EPIDEMIK & FORECAST — {disease}",
        "",
        "**Apa yang dianalisis?** Kurva epidemik menggambarkan distribusi kasus menurut waktu. Bentuknya membantu epidemiolog "
        "menilai pola temporal dan mengarahkan hipotesis, tetapi **bentuk kurva saja tidak membuktikan sumber, mekanisme, atau arah penularan**."
    ]
    if isinstance(curve, (tuple, list)) and len(curve) >= 4:
        lines.append(f"**Klasifikasi pola:** **{curve[0]}**. {curve[1]} **Makna:** {curve[2]} **Implikasi:** {curve[3]}")
    if isinstance(t, pd.DataFrame) and not t.empty:
        y = pd.to_numeric(t.get("Jumlah Kasus"), errors="coerce").dropna()
        if not y.empty:
            peak_i = y.idxmax()
            peak_date = pd.to_datetime(t.loc[peak_i, "Tanggal Sakit"], errors="coerce") if "Tanggal Sakit" in t else None
            lines.append(f"**Puncak observasi:** {int(y.max()):,} kasus pada **{peak_date.date() if pd.notna(peak_date) else 'tanggal tidak tersedia'}**.")
            if len(y) >= 7:
                first = y.head(7).mean()
                last = y.tail(7).mean()
                direction = "meningkat" if last > first else ("menurun" if last < first else "relatif tetap")
                lines.append(f"Rata-rata 7 hari pada awal deret dibandingkan 7 hari terakhir menunjukkan beban **{direction}**.")
    if isinstance(waves, list):
        lines.append(f"**Gelombang temporal teridentifikasi:** {len(waves)} pola/gelombang menurut algoritma yang digunakan; hasil perlu diperiksa terhadap epidemiologi lapangan.")
    if isinstance(forecast, dict) and forecast.get("status") == "ok":
        lines.append(
            f"**Forecast.** Model memproyeksikan **{len(forecast.get('forecast', []))} hari** ke depan. "
            f"Peak model sekitar **{_fmt(forecast.get('peak_value'),1)} kasus** pada **{forecast.get('peak_date','N/A')}**. "
            "Forecast adalah estimasi berbasis pola historis, bukan kepastian kejadian."
        )
        lines.append(
            f"Backtest error yang tersedia: **MAE {_fmt(forecast.get('mae'),2)}** dan **RMSE {_fmt(forecast.get('rmse'),2)}**. "
            "Kinerja historis tidak menjamin akurasi saat kondisi epidemiologi berubah."
        )
    else:
        lines.append("Forecast belum tersedia atau data belum mencukupi.")
    lines += [
        "",
        "### 🔬 Penjelasan akademik/praktisi",
        "Interpretasi kurva harus mempertimbangkan interval waktu, keterlambatan pelaporan, tanggal onset, perubahan testing, seasonality, serial interval, dan intervensi. "
        "Forecast sebaiknya dibandingkan dengan baseline sederhana dan dievaluasi secara temporal berulang.",
        "",
        "### 👥 Penjelasan untuk masyarakat",
        "Kurva adalah **cerita jumlah kasus dari hari ke hari**. Bila garis naik, berarti jumlah kasus yang tercatat bertambah; bila turun, jumlah kasus yang tercatat berkurang. "
        "Prediksi di sebelahnya adalah perkiraan komputer berdasarkan pola sebelumnya, sehingga hasil nyata bisa berbeda.",
        "",
        "### ⚠️ Catatan",
        "Interval forecast pada prototipe tidak boleh dianggap sebagai prediction interval epidemiologis yang sudah terkalibrasi sebelum dilakukan validasi eksternal."
    ]
    return "\n".join(lines)


def spatial_expert(spatial, disease="penyakit"):
    if not isinstance(spatial, pd.DataFrame) or spatial.empty:
        return "### 🗺️ RESUME SPATIAL EPIDEMIOLOGY\n\nKoordinat geografis yang valid belum mencukupi untuk analisis spasial."
    total = len(spatial)
    clusters = spatial["Cluster"].value_counts().drop(index=-1, errors="ignore") if "Cluster" in spatial.columns else pd.Series(dtype=int)
    noise = int((spatial["Cluster"] == -1).sum()) if "Cluster" in spatial.columns else 0
    clustered = int(clusters.sum()) if not clusters.empty else 0
    lines = [
        f"### 🗺️ RESUME SPATIAL EPIDEMIOLOGY — {disease}",
        "",
        f"**Hasil utama.** Dari **{total:,} titik kasus**, algoritma menemukan **{len(clusters):,} cluster** dan **{noise:,} titik noise/outlier**. "
        f"Sebanyak **{_pct(clustered,total):.1f}%** titik masuk ke cluster.",
        "",
        "### 🔬 Bagaimana membaca DBSCAN?",
        "DBSCAN mengelompokkan titik berdasarkan **kepadatan spasial** menggunakan parameter jarak (*eps*) dan jumlah minimum titik (*min_samples*). "
        "Cluster berarti terdapat konsentrasi titik yang berdekatan menurut parameter tersebut. Noise berarti titik tidak memenuhi kepadatan yang dipersyaratkan.",
        "",
        "### 🧠 Interpretasi epidemiologis",
        "Cluster spasial adalah **sinyal untuk investigasi**, bukan bukti bahwa cluster tersebut merupakan sumber penularan. Untuk menyatakan transmisi diperlukan keterkaitan waktu, hubungan antar kasus, pajanan, mobilitas, dan investigasi lapangan.",
    ]
    if not clusters.empty:
        cid = int(clusters.index[0])
        n = int(clusters.iloc[0])
        g = spatial[spatial["Cluster"] == cid]
        place = None
        for col in ["Desa/Kelurahan", "Kecamatan", "Kabupaten", "Provinsi"]:
            if col in g.columns and not g[col].dropna().empty:
                place = str(g[col].mode().iloc[0])
                break
        lines.append(f"**Cluster terbesar:** #{cid} dengan **{n:,} titik**" + (f", dominan di **{place}**." if place else "."))
    lines += [
        "",
        "### 👥 Penjelasan untuk masyarakat",
        "Peta membantu melihat **apakah kasus cenderung berkumpul di lokasi tertentu**. Titik yang berkumpul bukan berarti tempat itu pasti sumber penyakit. "
        "Peta hanya memberi petunjuk lokasi mana yang layak diperiksa lebih lanjut.",
        "",
        "### ⚠️ Hal yang harus diperiksa",
        "Kualitas koordinat, geocoding, kepadatan penduduk, perbedaan akses pelayanan, perubahan waktu pelaporan, serta pilihan parameter DBSCAN dapat mengubah jumlah dan bentuk cluster."
    ]
    return "\n".join(lines)


def risk_expert(result):
    risk = result.get("risk_factors") if isinstance(result, dict) else result
    if not isinstance(risk, dict):
        return "### 🧪 RESUME FAKTOR RISIKO\n\nBelum ada hasil analisis faktor risiko."
    rows = []
    for factor, obj in risk.items():
        if not isinstance(obj, dict) or str(factor).startswith("MULTIVARIAT"):
            continue
        p = obj.get("p_value")
        if p is not None and pd.notna(p):
            rows.append((str(factor), float(p)))
    lines = [
        "### 🧪 RESUME ANALISIS FAKTOR RISIKO",
        "",
        "Analisis faktor risiko menilai apakah karakteristik/paparan tertentu **berasosiasi** dengan outcome pada dataset. "
        "Asosiasi statistik **bukan bukti sebab-akibat**.",
    ]
    if rows:
        sig = [(f,p) for f,p in rows if p < 0.05]
        if sig:
            lines.append(
                "**Sinyal statistik:** " + ", ".join(f"**{f}** (p={p:.4f})" for f,p in sig) +
                ". Hasil ini perlu dibaca bersama ukuran efek, CI 95%, ukuran sampel, confounding, dan bias."
            )
        else:
            lines.append(
                "Tidak ada faktor yang mencapai p<0,05 pada hasil yang tersedia. Ini **tidak membuktikan tidak adanya faktor risiko**; "
                "kekuatan analisis bergantung pada ukuran sampel, kualitas pengukuran, dan variasi outcome."
            )
    lines += [
        "",
        "### 🔬 Untuk akademisi/praktisi",
        "**cOR (crude odds ratio)** menunjukkan asosiasi sebelum penyesuaian. **aOR (adjusted odds ratio)** berasal dari model multivariat setelah mengontrol variabel yang dimasukkan ke model. "
        "OR harus dibaca bersama CI 95%, definisi reference category, missing data, dan strategi pemilihan kovariat.",
        "Perlu dibedakan antara **confounding**, **effect modification**, dan hubungan kausal. Uji Chi-square hanya menjawab ada/tidaknya asosiasi pada tabel kategorik; "
        "ia tidak mengukur besarnya risiko dan tidak membuktikan mekanisme.",
        "",
        "### 👥 Untuk masyarakat",
        "Jika komputer menemukan bahwa suatu kelompok lebih sering muncul pada kasus, itu berarti **ada pola hubungan dalam data**. Bukan berarti faktor tersebut pasti menyebabkan penyakit. "
        "Untuk menyatakan penyebab, dibutuhkan penelitian dan bukti yang lebih kuat.",
        "",
        "### ⚠️ Prioritas perbaikan",
        "Tambahkan denominator populasi, ukuran efek dan CI 95%, pemeriksaan confounding, missingness, multiple testing, serta validasi eksternal sebelum hasil digunakan sebagai dasar kebijakan."
    ]
    return "\n".join(lines)


def ml_expert(ml, disease=None, label=None):
    """Human-readable ML narrative scoped to the selected disease and geographic scope."""
    disease_name = disease or "penyakit terpilih"
    scope_name = label or "scope analisis"
    if not isinstance(ml, dict) or not ml:
        return f"### 🤖 RESUME MACHINE LEARNING — {disease_name}\n\nML layer belum dijalankan untuk **{disease_name}** pada **{scope_name}**."

    primary = ml.get("primary_engines", {})
    ci = ml.get("continuous_intelligence", {})
    lines = [
        f"### 🤖 RESUME MACHINE LEARNING & EPIDEMIOLOGICAL AI — {disease_name}",
        "",
        f"Analisis ML ini **khusus menggunakan data {disease_name}** pada scope **{scope_name}**. Ketika pengguna mengganti filter penyakit, seluruh model, sinyal, metrik, dan interpretasi pada menu ini harus dibaca ulang berdasarkan kohort penyakit yang baru dipilih.",
        f"SI-HIS menjalankan **{len(primary) if isinstance(primary,dict) else 5} primary engine** dan **{len(ci) if isinstance(ci,dict) else 0} supporting intelligence modules**. ML memperkuat surveillance dan decision support; analisis epidemiologi TIME + PERSON + PLACE tetap menjadi dasar interpretasi.",
    ]

    explanations = {
        "case_severity": (
            f"Untuk {disease_name}, model ini memperkirakan probabilitas outcome keparahan yang didefinisikan sistem pada kasus penyakit tersebut. "
            "Hasilnya digunakan untuk membantu menentukan kasus yang perlu diprioritaskan untuk pemantauan, bukan untuk menetapkan diagnosis atau keputusan klinis otomatis."
        ),
        "klb": (
            f"Untuk {disease_name}, model mencoba mengenali sinyal peningkatan beban kasus dalam 7 hari berikutnya pada unit wilayah yang dianalisis. "
            "Target ini merupakan target operasional/model dan **bukan penetapan legal KLB**."
        ),
        "spatial": (
            f"Untuk {disease_name}, model menggabungkan pola temporal dan lokasi geografis untuk mencari wilayah yang perlu diverifikasi lebih dini. "
            "Prediksi spasial adalah sinyal prioritas dan tidak membuktikan sumber penularan."
        ),
        "vulnerable": (
            f"Untuk {disease_name}, model mencari pola karakteristik person yang berkaitan dengan outcome target sehingga kelompok yang perlu mendapat perhatian surveillance dapat diprioritaskan. "
            "Profil rentan tidak berarti setiap anggota kelompok tersebut pasti mengalami outcome yang sama."
        ),
        "forecast": (
            f"Forecasting memproyeksikan kecenderungan jumlah kasus **{disease_name}** berdasarkan pola waktu pada data yang difilter. "
            "Forecast adalah estimasi model dan harus dibandingkan dengan observasi aktual secara berkala."
        ),
    }

    for key, title in [
        ("case_severity","Severity"),
        ("klb","Outbreak/KLB 7 Hari"),
        ("spatial","Spatial Outbreak"),
        ("vulnerable","Population Vulnerability"),
        ("forecast","Forecasting"),
    ]:
        r = primary.get(key) if isinstance(primary,dict) else None
        if not isinstance(r,dict):
            continue
        status = str(r.get("status","-")).lower()
        metrics = r.get("metrics",{}) if isinstance(r.get("metrics",{}),dict) else {}
        if key == "forecast":
            lines.append(f"**{title}:** status **{r.get('status','-')}**. {explanations[key]}")
            if r.get("mae") is not None:
                lines.append(f"MAE backtest sekitar **{_fmt(r.get('mae'),3)}**; semakin kecil error absolut, semakin dekat prediksi model dengan observasi pada backtest. Evaluasi forecasting tidak menggunakan ROC-AUC/PR-AUC/recall/Brier sebagai metrik utama.")
        else:
            roc = metrics.get("roc_auc")
            pr = metrics.get("pr_auc")
            recall = metrics.get("recall")
            brier = metrics.get("brier")
            metric_text = (
                f"ROC-AUC **{_fmt(roc,3)}**, PR-AUC **{_fmt(pr,3)}**, "
                f"recall **{_fmt(recall,3)}**, Brier **{_fmt(brier,3)}**."
            )
            if status == "error":
                metric_text += " Karena status model **error**, angka 0,000 atau nilai kosong pada metrik tidak boleh ditafsirkan sebagai performa model nol; evaluasi belum valid dan perlu diperiksa pada data/label."
            else:
                recall_value = float(recall) if recall is not None and pd.notna(recall) else None
                metric_text += (
                    f" Recall sebesar **{recall_value*100:.1f}%** berarti model menemukan sekitar {recall_value*100:.1f}% "
                    "dari kasus positif/target pada data uji, sehingga kasus target yang terlewat masih perlu diperhatikan."
                    if recall_value is not None else ""
                )
            lines.append(f"**{title}:** status **{r.get('status','-')}**. {explanations[key]} {metric_text}")

    lines += [
        "",
        "### 📖 Cara membaca angka ML dengan bahasa sederhana",
        "**ROC-AUC** menggambarkan kemampuan model membedakan kelompok target dan non-target secara keseluruhan; nilainya makin mendekati 1 menunjukkan diskriminasi yang makin baik pada data uji. **PR-AUC** melihat kualitas deteksi kelompok positif dengan memperhatikan precision dan recall, sehingga penting ketika kasus target relatif jarang. **Recall** menjawab pertanyaan: dari semua kasus target yang benar-benar ada, berapa persen yang berhasil ditemukan model. **Brier score** menilai seberapa baik probabilitas prediksi dibandingkan outcome aktual; semakin kecil umumnya semakin baik. Angka-angka ini harus dibaca bersama distribusi kelas, ukuran sampel, dan validasi temporal.",
        "",
        "### 🔬 Interpretasi epidemiologis",
        f"Semua keluaran ML di atas harus dipahami sebagai **sinyal untuk {disease_name}**, bukan sebagai kesimpulan yang berlaku untuk semua penyakit. Perubahan hasil ketika filter penyakit diganti adalah hal yang diharapkan karena model menerima kohort, pola waktu, karakteristik person, distribusi wilayah, dan outcome penyakit yang berbeda.",
        "Feature importance menunjukkan kontribusi prediktif dalam model, bukan bukti bahwa suatu faktor menyebabkan penyakit. Cluster spasial menunjukkan konsentrasi berdasarkan algoritma dan parameter yang digunakan, bukan otomatis sumber penularan.",
        "",
        "### 👥 Untuk praktisi dan pengambil keputusan",
        f"Gunakan hasil ML {disease_name} bersama kurva epidemik, Rₜ, EWS, TIME + PERSON + PLACE, distribusi spasial, outcome, dan hasil investigasi lapangan. Jika beberapa sinyal konsisten, wilayah/kasus tersebut dapat diprioritaskan untuk verifikasi; keputusan akhir tetap berada pada tenaga kesehatan dan otoritas yang berwenang.",
        "",
        "### ⚠️ Kesiapan operasional",
        "Model yang berstatus error harus diperbaiki sebelum digunakan. Model yang berstatus OK tetap memerlukan calibration monitoring, data/model drift monitoring, external validation, version registry, audit trail, dan pembaruan label berdasarkan outcome nyata.",
    ]
    return "\n".join(lines)
def ai_prediction_expert(result, disease, label):
    df = result.get("analysis_dataframe") if isinstance(result,dict) else None
    if not isinstance(df,pd.DataFrame) or df.empty:
        return "### 🧠 AI EPIDEMIOLOGIST SITUATIONAL REPORT\n\nData belum mencukupi."
    total = len(df)
    death = _death_series(df)
    deaths = int(death.sum())
    cfr = _pct(deaths,total)
    klb = result.get("klb",{}) if isinstance(result,dict) else {}
    ews = result.get("ews",{}) if isinstance(result,dict) else {}
    rt = result.get("rt",{}) if isinstance(result,dict) else {}
    ml = result.get("ml",{}) if isinstance(result,dict) else {}

    lines = [
        f"### 🧠 AI EPIDEMIOLOGIST — SITUATIONAL AWARENESS REPORT",
        "",
        f"**Scope:** {label} | **Penyakit:** {disease} | **Kasus:** {total:,} | **Meninggal:** {deaths:,} | **CFR:** {cfr:.2f}%.",
        "",
        "### 1. STATUS SITUASI",
        "Status situasi merupakan sintesis dari beban kasus, kematian, perubahan waktu, distribusi Person–Place–Time, EWS, spatial signal, forecast, dan ML. "
        "Sistem membedakan **sinyal untuk investigasi** dari **penetapan status resmi**.",
    ]
    if isinstance(klb,dict):
        status = _first_valid([klb.get("status"),klb.get("level"),klb.get("classification")])
        if status:
            lines.append(f"**Surveillance status:** **{status}**. Status ini harus dibaca sesuai ruleset yang digunakan dan tidak otomatis berarti penetapan legal KLB.")
    if isinstance(ews,dict):
        lines.append("**EWS:** " + "; ".join(f"{k}={v}" for k,v in list(ews.items())[:6]) + ".")
    if isinstance(rt,dict):
        rv = _first_valid([rt.get("rt"),rt.get("Rt"),rt.get("estimate")])
        if rv is not None:
            lines.append(f"**Rₜ:** { _fmt(rv,2) }. Interpretasi membutuhkan interval ketidakpastian dan asumsi estimasi.")

    lines += [
        "",
        "### 2. APA YANG TERJADI?",
        "Pertanyaan pertama adalah apakah beban kasus sedang berubah, apakah terdapat kematian, dan apakah perubahan terkonsentrasi pada kelompok atau lokasi tertentu. "
        "Jawaban harus ditarik dari data observasi dan dibandingkan dengan periode sebelumnya, bukan dari skor AI saja.",
    ]

    # Add highest burden place and person.
    for col, name in [("Provinsi","provinsi"),("Kabupaten","kabupaten/kota"),("Kecamatan","kecamatan"),("Desa/Kelurahan","desa/kelurahan")]:
        c,n,p = _top_category(df,col)
        if c:
            lines.append(f"**Lokasi dengan kasus terbanyak pada tingkat {name}:** {c} ({n:,} kasus; {p:.1f}%).")
            break
    age, an, ap = _top_category(df,"Umur")
    if age:
        lines.append(f"**Kelompok umur terbanyak:** {age} ({an:,}; {ap:.1f}%).")

    lines += [
        "",
        "### 3. MENGAPA INI PENTING?",
        "Beban absolut, CFR, perubahan temporal, dan konsentrasi spasial menjawab dimensi risiko yang berbeda. Satu indikator dapat tinggi sementara indikator lain rendah. "
        "Karena itu SI-HIS menyajikan indikator secara bersamaan agar keputusan tidak dibuat dari satu angka.",
        "",
        "### 4. PREDIKSI & REKOMENDASI",
        "Prediksi digunakan untuk **prioritas surveillance dan kesiapsiagaan**, sedangkan rekomendasi adalah decision support. "
        "AI tidak menetapkan diagnosis, status KLB legal, atau keputusan klinis/kebijakan secara otomatis.",
        "",
        "### 🔬 Untuk akademisi/praktisi",
        "Gunakan laporan ini sebagai *situational awareness layer*. Verifikasi case definition, denominator, missingness, reporting delay, confounding, calibration, "
        "external validity, dan konteks intervensi. Setiap alert idealnya memiliki jejak: **signal → evidence → interpretation → recommended action → human confirmation → outcome**.",
        "",
        "### 👥 Untuk masyarakat",
        "Laporan ini mengubah tabel angka menjadi cerita yang lebih mudah dipahami: **apa yang terjadi, siapa yang terdampak, di mana, kapan, seberapa serius, dan apa yang perlu diperiksa berikutnya**. "
        "AI membantu menemukan pola, tetapi keputusan akhir tetap dibuat oleh tenaga kesehatan dan otoritas yang berwenang.",
    ]
    if isinstance(ml,dict) and ml:
        lines.append("### 5. KESIMPULAN INTELIJEN")
        lines.append("Sinyal epidemiologi, spasial, forecast, dan ML harus diperlakukan sebagai **bukti yang saling melengkapi**. "
                     "Semakin banyak sinyal yang konsisten, semakin kuat alasan untuk melakukan verifikasi lapangan; namun konsistensi sinyal tetap bukan pengganti konfirmasi epidemiologis.")
    return "\n".join(lines)


def vulnerable_expert(v):
    if not isinstance(v,list) or not v:
        return "### 👥 RESUME POPULASI RENTAN\n\nBelum ada profil rentan yang dapat dihitung."
    d = pd.DataFrame(v)
    lines = [
        "### 👥 RESUME VULNERABLE POPULATION",
        "",
        "Stratifikasi kerentanan digunakan untuk menemukan kombinasi karakteristik yang memiliki beban atau outcome lebih berat dalam dataset. "
        "Strata kecil harus ditafsirkan hati-hati karena estimasi dapat tidak stabil."
    ]
    if not d.empty:
        r = d.iloc[0]
        lines.append(
            f"Profil dengan nilai prioritas tertinggi pada hasil tersedia adalah **{r.get('Age_Group','-')} × "
            f"{r.get('Pekerjaan','-')} × {r.get('Status Komorbid','-')}**, n={int(_num(r.get('Total')))}, "
            f"CFR={_num(r.get('CFR (%)')):.2f}%."
        )
    lines += [
        "",
        "### 🔬 Akademik/praktisi",
        "Kerentanan bukan sinonim severity. Vulnerability sebaiknya didefinisikan menggunakan faktor yang relevan terhadap paparan, susceptibility, akses layanan, komorbiditas, usia, imunisasi, dan konteks sosial; "
        "definisi target harus ditetapkan sebelum model dilatih.",
        "",
        "### 👥 Masyarakat",
        "Kelompok rentan berarti kelompok yang mungkin membutuhkan perhatian lebih karena karakteristik tertentu. Ini bukan berarti setiap orang dalam kelompok tersebut pasti akan sakit berat."
    ]
    return "\n".join(lines)
