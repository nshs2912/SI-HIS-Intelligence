"""SI-HIS core package.

The package initializer is intentionally lightweight. Business logic belongs in
explicit modules so imports do not mutate analytics functions or third-party
classes at runtime.
"""

__version__ = "0.3.0"

from .data_provider import get_cases, get_metadata, get_cases_copy, validate_case_schema
from .scope import QueryScope, apply_scope, scope_label
from .statistics import AGE_GROUPS, add_age_groups, hitung_bivariat_lengkap, multivariable_logistic

__all__ = [
    "__version__",
    "get_cases", "get_metadata", "get_cases_copy", "validate_case_schema",
    "QueryScope", "apply_scope", "scope_label",
    "AGE_GROUPS", "add_age_groups", "hitung_bivariat_lengkap", "multivariable_logistic",
]
