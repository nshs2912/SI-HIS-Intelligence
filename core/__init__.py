"""SI-HIS core package.

The package initializer is kept as a small compatibility facade for the legacy
Streamlit application. New code should import explicit modules directly:
``data_provider``, ``scope``, ``statistics``, ``analytics`` and ``ml_engine``.
"""

__version__ = "0.4.0"

from . import analytics as _analytics
from .data_provider import get_cases, get_metadata, get_cases_copy, validate_case_schema
from .scope import QueryScope, apply_scope, scope_label
from .statistics import AGE_GROUPS, add_age_groups, hitung_bivariat_lengkap, multivariable_logistic
from .national_dummy import generate_national_dummy
from .clinical_intelligence import ClinicalIntelligenceEngine
from .regulatory_boundary import regulatory_inventory

# Transitional compatibility only: legacy app.py imports these symbols from
# core.analytics. No dataset is mutated; the generator returns a deep copy.
_analytics.generate_data_simulasi = generate_national_dummy
_analytics.hitung_bivariat_lengkap = hitung_bivariat_lengkap

__all__ = [
    "__version__",
    "get_cases", "get_metadata", "get_cases_copy", "validate_case_schema",
    "QueryScope", "apply_scope", "scope_label",
    "AGE_GROUPS", "add_age_groups", "hitung_bivariat_lengkap", "multivariable_logistic",
]
