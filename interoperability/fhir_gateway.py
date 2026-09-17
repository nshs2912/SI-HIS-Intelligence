"""Transport-neutral FHIR gateway boundary for SI-HIS.

Actual SATUSEHAT authentication/submission must be implemented only after
organization/IHS credentials, environment, use-case profile and current API
requirements are configured. No secrets are stored in source code.
"""
from fhir.validator import validate_bundle

class FHIRGateway:
    def __init__(self, base_url=None):
        self.base_url = base_url

    def validate(self, bundle):
        return validate_bundle(bundle)

    def submit(self, bundle):
        errors = self.validate(bundle)
        if errors:
            raise ValueError("FHIR validation failed: " + "; ".join(errors))
        raise NotImplementedError(
            "Submission transport requires configured SATUSEHAT credentials, "
            "organization identifiers, environment and use-case-specific profile mapping."
        )
