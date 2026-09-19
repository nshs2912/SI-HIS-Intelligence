"""Client Information Delivery Layer for SI-HIS.

Maps the common intelligence engine to the information granularity and decision
questions of each client. This module is intentionally presentation-oriented:
it does not replace the clinical/epidemiological engines.
"""

CLIENT_INTELLIGENCE = {
    "Kemenkes": {
        "title": "National Health Intelligence",
        "level": "national",
        "scope": "Indonesia → provinsi → kabupaten/kota → agregat fasilitas",
        "questions": [
            "Apa yang sedang terjadi secara nasional?",
            "Di mana perubahan pola kesehatan terjadi?",
            "Bagaimana tren, forecast, burden, outcome, dan kelompok rentan?",
            "Sinyal apa yang memerlukan verifikasi lintas wilayah?",
        ],
        "outputs": [
            "National disease burden and trend",
            "Epidemiological surveillance and early-warning signals",
            "PTM risk and outcome intelligence",
            "Spatial and spatiotemporal signals",
            "Forecast and uncertainty",
            "Cross-region comparison and policy intelligence",
        ],
    },
    "BPJS": {
        "title": "JKN Health & Utilization Intelligence",
        "level": "national/payer",
        "scope": "peserta → pelayanan → rujukan → klaim → outcome → biaya",
        "questions": [
            "Bagaimana pola utilisasi JKN berubah?",
            "Penyakit dan pathway mana yang menghasilkan beban pelayanan tinggi?",
            "Bagaimana referral, admission, readmission, medication dan outcome berubah?",
            "Di mana preventive intervention berpotensi mengurangi beban pelayanan?",
        ],
        "outputs": [
            "Population and utilization intelligence",
            "Referral and care-pathway signals",
            "High-utilization/high-cost disease signals",
            "Forecast of service demand",
            "Chronic disease progression signals",
            "Outcome and efficiency intelligence",
        ],
    },
    "Dinkes Provinsi": {
        "title": "Provincial Health Intelligence",
        "level": "provincial",
        "scope": "provinsi → kabupaten/kota → agregat puskesmas",
        "questions": [
            "Kabupaten/kota mana yang menunjukkan perubahan pola?",
            "Apakah sinyal bersifat lokal atau meluas?",
            "Bagaimana distribusi penyakit, outcome dan kelompok rentan antarwilayah?",
            "Koordinasi dan verifikasi lintas kabupaten/kota apa yang diperlukan?",
        ],
        "outputs": [
            "Provincial epidemiological profile",
            "District/city comparative profiles",
            "Regional early-warning signals",
            "Spatial clusters and trend changes",
            "Forecast and vulnerable-population signals",
            "Coordination intelligence",
        ],
    },
    "Dinkes Kabupaten/Kota": {
        "title": "District/City Local Health Intelligence",
        "level": "district/city",
        "scope": "kabupaten/kota → kecamatan → desa → puskesmas",
        "questions": [
            "Puskesmas mana yang menunjukkan perubahan sinyal?",
            "Apakah pola TIME-PERSON-PLACE berubah?",
            "Apakah sinyal memerlukan verifikasi lapangan?",
            "Bagaimana hasil investigasi dikembalikan ke sistem untuk re-analysis?",
        ],
        "outputs": [
            "Local surveillance and early warning",
            "Puskesmas-by-Puskesmas aggregate intelligence",
            "Village/district spatial signals",
            "Risk and vulnerable-population signals",
            "Forecast and ML decision support",
            "Verification/follow-up workflow",
        ],
    },
    "Puskesmas": {
        "title": "Frontline / Puskesmas Health Intelligence",
        "level": "facility/frontline",
        "scope": "puskesmas → wilayah kerja → desa/kelurahan → population",
        "questions": [
            "Apa perubahan kesehatan pada wilayah kerja Puskesmas?",
            "Desa/kelompok mana yang menunjukkan sinyal?",
            "Apakah perlu verifikasi kasus, follow-up, atau penguatan surveilans?",
            "Apa outcome setelah tindakan dan bagaimana feedback masuk kembali?",
        ],
        "outputs": [
            "Local population-health profile",
            "Disease trend and epidemiology",
            "Early-warning/anomaly signals",
            "Spatial and village-level signals",
            "Risk/vulnerability intelligence",
            "Preventive and follow-up decision support",
        ],
    },
    "Workforce Health Intelligence": {
        "title": "Corporate Workforce Health Intelligence",
        "level": "corporate",
        "scope": "employee → workforce segment → site/department → aggregate",
        "questions": [
            "Bagaimana profil kesehatan workforce?",
            "Segmen/site mana yang menunjukkan perubahan indikator agregat?",
            "Apa kebutuhan preventive program?",
            "Bagaimana outcome program berubah dari waktu ke waktu?",
        ],
        "outputs": [
            "Aggregate workforce health profile",
            "Risk segmentation",
            "Temporal workforce signals",
            "Predictive health-risk signals when validated targets exist",
            "Preventive/prescriptive program intelligence",
            "Outcome feedback",
        ],
    },
    "Healthcare Provider Intelligence": {
        "title": "Healthcare Provider Intelligence",
        "level": "provider",
        "scope": "rumah sakit/klinik/lab/apotek → service → patient-flow aggregate",
        "questions": [
            "Bagaimana demand dan service mix berubah?",
            "Apa bottleneck operasional dan clinical pathway signal?",
            "Bagaimana kebutuhan laboratorium, farmasi dan kapasitas berubah?",
            "Apa outcome dan tindak lanjut yang perlu dipantau?",
        ],
        "outputs": [
            "Service demand and operational intelligence",
            "Clinical/service mix",
            "Laboratory and pharmacy intelligence",
            "Provider-specific predictive signals",
            "Capacity/resource decision support",
            "FHIR/SATUSEHAT canonical information mapping",
        ],
    },
}

PUSKESMAS_FLOW = [
    "Puskesmas mengirim/menyediakan data pelayanan dan surveillance sesuai kewenangan.",
    "SI-HIS membersihkan, memvalidasi, mengagregasi, dan menjalankan analitik.",
    "ML mendeteksi anomaly, trend, forecast, risk, spatial signal, dan outcome sesuai disease family.",
    "Dinkes Kabupaten/Kota menerima intelligence agregat lintas Puskesmas.",
    "Dinkes memilih Puskesmas/area yang memerlukan verifikasi berdasarkan evidence yang tersedia.",
    "Informasi tindak lanjut dikirim kembali ke Puskesmas yang relevan.",
    "Puskesmas melakukan verifikasi/intervensi sesuai SOP dan kewenangan.",
    "Outcome dan observasi baru masuk kembali ke SI-HIS untuk re-analysis dan continuous learning.",
]

def client_spec(name):
    return CLIENT_INTELLIGENCE.get(name, {})

def client_summary(name):
    spec = client_spec(name)
    if not spec:
        return {"status": "unknown", "client": name}
    return {"status": "ok", "client": name, **spec}
