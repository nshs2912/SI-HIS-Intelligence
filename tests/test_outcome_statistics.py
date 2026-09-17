import numpy as np
import pandas as pd
from core.statistics import hitung_bivariat_lengkap


def _sample():
    n=80
    rng=np.random.default_rng(7)
    return pd.DataFrame({
        "Umur":rng.integers(18,80,n),
        "Jenis Kelamin":rng.choice(["Laki-laki","Perempuan"],n),
        "Pekerjaan":rng.choice(["Petani","Pekerja Formal","Wiraswasta"],n),
        "Status Imunisasi":rng.choice(["Lengkap","Tidak Lengkap"],n),
        "Status Komorbid":rng.choice(["Ada Komorbid","Tidak Ada"],n),
        "Riwayat Perjalanan":rng.choice(["Ya","Tidak"],n),
        "Faktor Risiko Lain":rng.choice(["Tidak Ada","Kontak Erat"],n),
        "Outcome_Disease":rng.integers(0,2,n),
        "Outcome_Death":rng.integers(0,2,n),
    })


def test_explicit_outcomes_are_independent():
    df=_sample()
    disease=hitung_bivariat_lengkap(df,var_dep_binary="Outcome_Disease")
    death=hitung_bivariat_lengkap(df,var_dep_binary="Outcome_Death")
    assert disease["Outcome"]=="Outcome_Disease"
    assert death["Outcome"]=="Outcome_Death"
    assert disease["Umur"]["crosstab"].shape[1]>=2
    assert death["Umur"]["crosstab"].shape[1]>=2
    assert not disease["MULTIVARIAT — Logistic Regression"].empty
    assert not death["MULTIVARIAT — Logistic Regression"].empty
