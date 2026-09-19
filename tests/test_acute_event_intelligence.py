import pandas as pd
from core.acute_event_intelligence import detect_point_source_cluster, detect_mass_casualty_burst, analyze_exposure_window, poisoning_vs_disaster_signal


def test_point_source_cluster():
    onset=pd.date_range('2026-09-19 08:00',periods=8,freq='5min')
    df=pd.DataFrame({'Tanggal Onset':onset,'Kabupaten':['A']*8})
    out=detect_point_source_cluster(df,time_window_minutes=60,min_cases=5)
    assert out['point_source_signal'] is True
    assert out['cases_in_window']==8


def test_exposure_delay():
    df=pd.DataFrame({'Tanggal Paparan':pd.to_datetime(['2026-09-19 08:00','2026-09-19 08:00']), 'Tanggal Onset':pd.to_datetime(['2026-09-19 08:30','2026-09-19 09:30'])})
    out=analyze_exposure_window(df)
    assert out['status']=='ok'
    assert out['median_hours']==0.5


def test_poisoning_disaster_signal_is_differential():
    onset=pd.date_range('2026-09-19 08:00',periods=12,freq='3min')
    df=pd.DataFrame({'Tanggal Onset':onset})
    out=poisoning_vs_disaster_signal(df)
    assert out['status']=='signal'
    assert 'Keracunan/common exposure' in out['differential_hypotheses']
    assert 'Bencana non-infeksi/chemical/environmental exposure' in out['differential_hypotheses']
