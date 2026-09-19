"""Clinical/SaMD candidate facade. Deliberately separate from epidemiological engine."""
from .laboratory_intelligence import laboratory_intelligence
from .pharmacy_intelligence import pharmacy_intelligence
from .dietitian_intelligence import dietitian_intelligence
from .clinical_journey_intelligence import clinical_journey_intelligence
from .patient_summary import summarize_patient
from .regulatory_boundary import regulatory_inventory
from .evidence_safety import safety_gate

class ClinicalIntelligenceEngine:
    """Non-epidemiological clinical intelligence boundary.

    This facade never calls the population epidemiology engine. Each clinical
    module has an independent intended purpose and should be validated,
    risk-managed and regulated according to its actual deployment.
    """
    def analyze(self, domain, df, **kwargs):
        if domain=="laboratory": result=laboratory_intelligence(df, kwargs.get("patient_id"))
        elif domain=="pharmacy": result=pharmacy_intelligence(df)
        elif domain=="dietitian": result=dietitian_intelligence(df)
        elif domain=="clinical_journey": result=clinical_journey_intelligence(df, kwargs.get("patient_id"))
        elif domain=="patient_summary": result=summarize_patient(df, kwargs["context"])
        elif domain=="regulatory_inventory": result=regulatory_inventory()
        else: raise ValueError(f"Unknown clinical domain: {domain}")
        return safety_gate(result)
