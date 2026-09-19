import pandas as pd

from core.disease_intelligence import classify_disease, analyze_onset_reporting_bias, extract_epicurve_features, candidate_index_cases


def test_disease_taxonomy_ptm_and_infectious():
    assert classify_disease('Diabetes Melitus').family == 'PTM'
    assert classify_disease('Demam Dengue').subgroup == 'vector-borne'
    assert classify_disease('Leptospirosis').subgroup == 'zoonotic'
    assert classify_disease('TBC').transmission == 'direct'


def test_onset_bias_and_candidate_index_are_explicit():
    df = pd.DataFrame({
        'Nama':['A','B','C'],
        'Tanggal Onset':['2026-09-01','2026-09-03','2026-09-01'],
        'Tanggal Pemeriksaan':['2026-09-05','2026-09-04','2026-09-02'],
        'Provinsi':['Jawa Tengah']*3,
    })
    q=analyze_onset_reporting_bias(df)
    assert q['status']=='ok'
    assert q['onset_completeness_pct']==100.0
    assert q['onset_to_examination']['negative_delay_n']==0
    idx=candidate_index_cases(df)
    assert idx['candidate_count']==2
    assert idx['status']=='candidate_only'


def test_epicurve_features_use_onset():
    df=pd.DataFrame({'Tanggal Onset':['2026-09-01','2026-09-01','2026-09-02','2026-09-04']})
    out=extract_epicurve_features(df)
    assert out['status']=='ok'
    assert out['features']['total_cases']==4
    assert out['features']['peak_cases']==2
