import pandas as pd
from core.laboratory_intelligence import laboratory_intelligence
from core.pharmacy_intelligence import pharmacy_intelligence
from core.dietitian_intelligence import dietitian_intelligence
from core.clinical_journey_intelligence import clinical_journey_intelligence
from core.patient_summary import summarize_patient
from core.privacy_security import AccessContext
from core.regulatory_boundary import regulatory_inventory

def test_clinical_modules_return_guardrails():
    df=pd.DataFrame({"Patient ID":["P1","P1"],"Date":["2026-01-01","2026-02-01"],"Pemeriksaan":["Hb","Hb"],"Hasil":[10,12],"Reference Low":[12,12],"Reference High":[16,16],"Obat":["Drug A","Drug A"],"Diagnosis":["Condition A","Condition A"],"Berat Badan":[70,69],"BMI":[25,24.6],"Protein":[70,72],"Jenis Event":["lab","follow-up"]})
    for fn in [laboratory_intelligence, pharmacy_intelligence, dietitian_intelligence, clinical_journey_intelligence]:
        assert "guardrail" in fn(df)

def test_patient_summary_is_role_scoped():
    df=pd.DataFrame({"Patient ID":["P1"],"Diagnosis":["A"],"Berat Badan":[70],"Obat":["X"],"Tanggal":["2026-01-01"]})
    out=summarize_patient(df, AccessContext("U1","dietitian","care","P1"))
    assert out["status"]=="OK"
    assert "medication" not in out["summary"]["fields"]

def test_regulatory_domains_are_separate():
    inv=regulatory_inventory()
    assert inv["population"]["domain"]=="HEALTH_INFORMATION_SYSTEM"
    assert inv["clinical"]["laboratory"]["domain"]=="CLINICAL_SAMD_CANDIDATE"
