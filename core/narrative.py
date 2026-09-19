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
        "### 🔬 Penjelasan akademik/praktisi",        "Interpretasi kurva harus mempertimbangkan interval waktu, keterlambatan pelaporan, tanggal onset, perubahan testing, seasonality, serial interval, dan intervensi. "        "Forecast sebaiknya dibandingkan dengan baseline sederhana dan dievaluasi secara temporal berulang.",
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
    """Integrated risk-analysis resume: bivariate, multivariate and clinical outcomes."""
    risk = result.get("risk_factors") if isinstance(result, dict) else result
    outcomes = result.get("outcome_analyses", {}) if isinstance(result, dict) else {}
    if not isinstance(risk, dict):
        risk = {}

    bivariate = []
    for factor, obj in risk.items():
        if str(factor).startswith("MULTIVARIAT") or not isinstance(obj, dict):
            continue
        p = obj.get("p_value")
        if p is not None and pd.notna(p):
            bivariate.append((str(factor), float(p)))

    multivariate = []
    for key, obj in risk.items():
        if str(key).startswith("MULTIVARIAT"):
            multivariate.append((str(key), obj))

    lines = [
        "### 🧪 RESUME ANALISIS FAKTOR RISIKO",
        "",
        f"Analisis mencakup **{len(bivariate)} faktor pada analisis bivariat**"
        + (f" dan **{len(multivariate)} komponen multivariat**." if multivariate else "."),
        "Interpretasi difokuskan pada pola asosiasi dalam dataset; asosiasi statistik bukan bukti sebab-akibat.",
    ]

    sig = [(f, p) for f, p in bivariate if p < 0.05]
    if sig:
        lines.append(
            "**Bivariat:** sinyal asosiasi statistik terdeteksi pada "
            + ", ".join(f"**{f}** (p={p:.4f})" for f, p in sig)
            + ". Ukuran efek, CI 95%, confounding, dan kualitas data tetap perlu diperiksa."
        )
    elif bivariate:
        lines.append(
            "**Bivariat:** tidak ada faktor yang mencapai p<0,05 pada hasil yang tersedia. "
            "Ini tidak membuktikan tidak adanya faktor risiko."
        )

    if multivariate:
        available = []
        for key, obj in multivariate:
            if isinstance(obj, dict):
                analysis = obj.get("analysis", obj)
                if isinstance(analysis, dict):
                    available.append(key)
        if available:
            lines.append(
                "**Multivariat:** komponen/model yang tersedia digunakan untuk melihat apakah pola asosiasi "
                "tetap setelah mempertimbangkan variabel lain. Baca **aOR, CI 95%, dan p-value** bersama-sama; "
                "model tidak membuktikan kausalitas."
            )
        else:
            lines.append("**Multivariat:** struktur analisis tersedia, tetapi hasil model belum cukup untuk diringkas secara statistik.")
    else:
        lines.append("**Multivariat:** belum tersedia model yang dapat diringkas pada scope/data ini.")

    outcome_labels = [("Penyakit", "Penyakit"), ("Severity", "Severity"), ("Kasus Meninggal", "Kasus Meninggal")]
    outcome_lines = []
    for key, label in outcome_labels:
        obj = outcomes.get(key) if isinstance(outcomes, dict) else None
        if not isinstance(obj, dict):
            continue
        if not obj.get("available", False):
            outcome_lines.append(f"**Outcome {label}:** belum memenuhi kecukupan data/model pada scope ini.")
            continue
        n = int(obj.get("n", 0) or 0)
        events = int(obj.get("events", 0) or 0)
        rate = obj.get("event_rate_pct")
        p = obj.get("analysis", {}).get("p_value") if isinstance(obj.get("analysis"), dict) else None
        detail = f"**Outcome {label}:** {n:,} observasi, {events:,} event"
        if rate is not None and pd.notna(rate):
            detail += f" ({float(rate):.2f}%)"
        if p is not None and pd.notna(p):
            detail += f", p={float(p):.4g}"
        outcome_lines.append(detail + ".")
    if outcome_lines:
        lines.append("")
        lines.append("**Ringkasan outcome:** " + " ".join(outcome_lines))

    lines += [
        "",
        "**Catatan metodologis:** cOR menggambarkan asosiasi sebelum penyesuaian, sedangkan aOR menggambarkan asosiasi setelah variabel dalam model dikontrol. Interpretasi akhir perlu mempertimbangkan definisi outcome, reference category, missing data, ukuran sampel, confounding, effect modification, dan validasi eksternal.",
    ]
    return "\n".join(lines)


def ml_expert(ml, disease=None, label=None):
    """Expert ML narrative: epidemiological ML as a capability layer over 5 primary engines."""
    disease_name = disease or "penyakit terpilih"
    scope_name = label or "scope analisis"
    if not isinstance(ml, dict) or not ml:
        return (
            f"### 🤖 RESUME MACHINE LEARNING & EPIDEMIOLOGICAL AI — {disease_name}\n\n"
            f"ML layer belum dijalankan untuk **{disease_name}** pada **{scope_name}**."
        )

    primary = ml.get("primary_engines", {})
    supporting = ml.get("continuous_intelligence", {})
    vulnerability = supporting.get("vulnerability_clustering", {}) if isinstance(supporting, dict) else {}

    lines = [
        f"### 🤖 RESUME MACHINE LEARNING & EPIDEMIOLOGICAL AI — {disease_name}",
        "",
        "### 🧬 Apa yang Dilakukan ML Epidemiologi?",
        "",
        "ML Epidemiologi pada SI-HIS merupakan lapisan kecerdasan analitik yang menggunakan machine learning dan epidemiological intelligence untuk mengolah data surveillance menjadi **sinyal, prediksi, stratifikasi risiko, dan prioritas investigasi**.",
        "",
        "ML Epidemiologi menganalisis pola **TIME + PERSON + PLACE** serta karakteristik penyakit terpilih untuk membantu menjawab pertanyaan epidemiologis utama:",
        "",
        "- **Apa yang sedang berubah?** — mendeteksi anomali, perubahan tren, dan perubahan pola epidemiologi.",
        "- **Di mana perubahan terjadi?** — mengidentifikasi konsentrasi dan pola risiko spasial.",
        "- **Siapa yang perlu diperhatikan?** — mengidentifikasi kelompok atau populasi dengan karakteristik risiko tertentu.",
        "- **Seberapa serius sinyal tersebut?** — memperkirakan severity atau outcome target pada kasus yang dianalisis.",
        "- **Apakah beban penyakit berpotensi meningkat?** — memprediksi pertumbuhan atau peningkatan kasus.",
        "- **Apa yang mungkin terjadi berikutnya?** — melakukan forecasting berdasarkan pola temporal historis.",
        "- **Wilayah mana yang perlu diverifikasi lebih dahulu?** — melakukan prioritisasi surveillance dan investigasi.",
        "- **Sinyal mana yang memerlukan perhatian lebih lanjut?** — menggabungkan berbagai indikator menjadi epidemiological risk signal.",
        "",
        "### 🧠 Struktur ML Epidemiologi SI-HIS",
        "",
        f"ML Epidemiologi terdiri dari **5 Primary ML Engine** dan **{len(supporting) if isinstance(supporting, dict) else 9} Supporting Intelligence Module**. Kelima primary engine menghasilkan prediksi pada aspek yang berbeda, sedangkan supporting intelligence membantu menemukan perubahan, pola, dan sinyal epidemiologis yang memerlukan verifikasi lebih lanjut.",
        "",
        "**5 Primary ML Engine:**",
        "1. **Case Severity Prediction**",
        "2. **Outbreak/KLB 7-Day Prediction**",
        "3. **Spatial Outbreak Prediction**",
        "4. **Vulnerable Population Prediction**",
        "5. **Temporal Forecasting**",
        "",
        f"**Scope:** {scope_name}. Seluruh model dan supporting signal pada menu ini ditujukan khusus untuk **{disease_name}**. Saat filter penyakit berubah, kohort, target outcome, distribusi kelas, pola waktu, pola tempat, metrik, dan interpretasi harus dibaca ulang.",
        f"SI-HIS saat ini menjalankan **{len(primary) if isinstance(primary, dict) else 5} primary engine** dan supporting intelligence untuk memperkuat surveillance dan decision support.",
        "",
    ]

    if isinstance(vulnerability, dict) and vulnerability.get("status") == "ok":
        strata = vulnerability.get("vulnerability_strata")
        if isinstance(strata, pd.DataFrame) and not strata.empty:
            top = strata.iloc[0]
            lines += [
                "### 👥 Stratifikasi Kerentanan",
                "",
                f"Strata ditampilkan dengan label eksplisit: **usia, pekerjaan, dan status komorbid**. "
                f"Strata prioritas pada dataset adalah **{top.get('Strata_Label','-')}**, "
                f"n={int(top.get('Jumlah_Observasi',0))}, CFR={float(top.get('CFR_Persen',0)):.2f}%, "
                f"stabilitas={top.get('Stabilitas','-')}.",
                "",
                "CFR mentah tidak digunakan sebagai dasar tunggal untuk menentukan prioritas. Strata kecil perlu diperlakukan hati-hati karena estimasinya dapat tidak stabil.",
                "Strata **RENDAH** harus dibaca sebagai sinyal pada dataset, bukan sebagai estimasi risiko populasi. "
                "Sebagai prinsip kehati-hatian, ukuran kecil dapat menghasilkan ketidakpastian yang besar.",
                "",
            ]

    items = [
        ("Case Severity", "memperkirakan probabilitas outcome severity yang didefinisikan sistem pada kasus penyakit terpilih", "prioritas pemantauan"),
        ("Outbreak/KLB 7 Hari", "mencari sinyal peningkatan beban kasus dalam 7 hari berikutnya", "prioritas verifikasi wilayah"),
        ("Spatial Outbreak", "menggabungkan dimensi waktu dan lokasi untuk mencari area yang perlu diverifikasi lebih dini", "prioritas investigasi spasial"),
        ("Vulnerable Population", "mencari pola karakteristik person yang berkaitan dengan outcome target", "prioritas kelompok surveillance"),
        ("Forecasting", "memproyeksikan beban kasus berdasarkan pola waktu historis", "kesiapsiagaan dan kapasitas"),
    ]
    for title, purpose, use in items:
        lines.append(f"**{title}:** Untuk **{disease_name}**, model {purpose}. Keluaran digunakan sebagai **{use}**, bukan diagnosis atau keputusan otomatis.")

    lines += [
        "",
        "### 📖 Bahasa sederhana untuk metrik classifier",
        "**Recall** = dari semua kasus yang benar-benar memiliki outcome target, berapa banyak yang berhasil ditemukan model. Jadi recall 0,077 berarti sekitar **7,7% kasus target tertangkap** pada data uji; sebagian besar target masih dapat terlewat.",
        "**Specificity** = dari semua kasus yang sebenarnya bukan target, berapa banyak yang berhasil dikenali sebagai bukan target. Specificity 0,939 berarti sekitar **93,9% non-target tersaring dengan benar** pada data uji.",
        "**Precision** = dari semua kasus yang diprediksi sebagai target, berapa banyak yang benar-benar target. **PR-AUC** merangkum trade-off precision-recall dan penting ketika target jarang. **ROC-AUC** menilai kemampuan diskriminasi model pada berbagai threshold. **Brier score** menilai kualitas probabilitas; semakin kecil umumnya semakin baik.",
        "",
        "### 🔬 Interpretasi epidemiologis",
        f"Feature importance menunjukkan kontribusi prediktif variabel dalam model **{disease_name}**, bukan sebab-akibat. Sinyal spasial bukan otomatis sumber penularan. Forecast bukan kepastian. Risk score adalah alat prioritisasi internal dan bukan status KLB legal.",
        f"Kinerja model harus dibaca pada populasi dan periode uji yang digunakan. Nilai yang baik pada satu penyakit atau wilayah tidak otomatis berlaku pada penyakit/wilayah lain. Untuk **{disease_name}**, validasi eksternal dan monitoring drift tetap diperlukan sebelum penggunaan operasional.",
        "",
        "### 👥 Untuk praktisi/pengambil keputusan",
        f"Gunakan ML **{disease_name}** bersama TIME + PERSON + PLACE, kurva epidemik, Rₜ, EWS, outcome, analisis spasial, dan investigasi lapangan. ML membantu menentukan **di mana dan apa yang perlu diperiksa lebih dahulu**; otoritas manusia menentukan tindakan.",
        "",
        "### ⚠️ Batasan penting",
        "Outcome target harus memiliki definisi dan label yang valid. Data leakage, missingness, class imbalance, reporting delay, perubahan case definition, calibration drift, dan perubahan pola epidemiologi dapat mengubah performa model. Status ERROR berarti evaluasi belum valid, bukan performa model nol.",
        "",
        "### 🔄 Prinsip continuous intelligence",
        "DATA → ANALYSIS → PREDICTION → RECOMMENDATION → INTERVENTION → OUTCOME → NEW DATA → RE-ANALYSIS → CONTINUOUS LEARNING",
    ]
    return "\n".join(lines)

def ai_prediction_expert(result, disease, label):
    df = result.get("analysis_dataframe") if isinstance(result,dict) else None
    if not isinstance(df,pd.DataFrame) or df.empty:
        return "### 🧠 AI EPIDEMIOLOGIST SITUATIONAL REPORT\n\nData belum mencukupi."
    total = len(df)
    death = _death_series(df)
    deaths = int(death.sum())
    cfr = _pct(deaths,total)    klb = result.get("klb",{}) if isinstance(result,dict) else {}
    ews = result.get("ews",{}) if isinstance(result,dict) else {}    rt = result.get("rt",{}) if isinstance(result,dict) else {}
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
    """Narrative for vulnerable-population stratification.

    The narrative describes observed strata without ranking a small stratum as
    the highest-priority population based on raw CFR alone.
    """
    if not isinstance(v, list) or not v:
        return "### 👥 RESUME POPULASI RENTAN\\n\\nBelum ada profil rentan yang dapat dihitung."

    d = pd.DataFrame(v)
    lines = [
        "### 👥 RESUME POPULASI RENTAN",
        "",
        "Stratifikasi kerentanan digunakan untuk mengidentifikasi kombinasi karakteristik "
        "yang menunjukkan perbedaan beban penyakit atau outcome dalam dataset. Analisis ini "
        "merupakan sinyal epidemiologis untuk ditindaklanjuti, bukan penetapan risiko individual "
        "atau urutan prioritas intervensi.",
    ]

    if not d.empty:
        # Keep the first row as a representative observed stratum only when the
        # upstream result provides an explicit, stability-aware ordering.
        r = d.iloc[0]
        age = r.get("Age_Group", "-")
        occupation = r.get("Pekerjaan", "-")
        comorbidity = r.get("Status Komorbid", "-")
        n = int(_num(r.get("Total")))
        cfr = _num(r.get("CFR (%)"))

        lines.append(
            f"**Profil yang teridentifikasi pada hasil tersedia:** **{age} × "
            f"{occupation} × {comorbidity}**, dengan **n={n:,} observasi** "
            f"dan **CFR {cfr:.2f}%**."
        )
        lines.append(
            "Profil tersebut tidak diberi label sebagai *prioritas tertinggi* karena nilai CFR "
            "dapat terlihat tinggi pada strata dengan jumlah observasi kecil. Besarnya outcome "
            "perlu dibaca bersama ukuran sampel, stabilitas estimasi, dan ketidakpastian statistik."
        )

    lines += [
        "",
        "### 🔬 Akademik/praktisi",
        "Kerentanan sebaiknya ditafsirkan berdasarkan faktor yang relevan terhadap paparan, "
        "susceptibility, komorbiditas, usia, imunisasi, akses layanan, serta konteks sosial dan "
        "lingkungan. Strata dengan outcome relatif tinggi merupakan **sinyal untuk validasi dan "
        "investigasi lebih lanjut**, bukan bukti hubungan sebab-akibat.",
        "",
        "Sebelum digunakan untuk keputusan operasional, periksa ukuran strata, confidence interval "
        "atau ukuran ketidakpastian yang tersedia, definisi outcome, missing data, confounding, "
        "dan validitas eksternal. CFR adalah proporsi kematian di antara kasus dalam strata; "
        "CFR tidak sama dengan risiko kematian populasi.",
        "",
        "### 👥 Masyarakat",
        "Kelompok yang terlihat memiliki outcome lebih tinggi dalam dataset dapat menjadi kelompok "
        "yang perlu diperhatikan lebih lanjut. Namun, hasil tersebut tidak berarti setiap orang "
        "dalam kelompok tersebut pasti mengalami kondisi yang sama. Data yang lebih besar dan "
        "verifikasi lapangan diperlukan untuk memastikan apakah pola tersebut konsisten.",
    ]
    return "\n".join(lines)