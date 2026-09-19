# Privacy & Security by Design

The clinical modules use a defense-in-depth contract:

Identity → Role → Purpose → Patient/context → Minimum necessary fields → Clinical summary → Evidence traceability → Audit.

Controls represented in code:
- role-based field allow-list
- purpose requirement
- minimum-necessary projection
- pseudonymization helper
- de-identification helper
- provenance metadata
- human-review gate

These controls are not a substitute for a complete privacy/security program, DPIA/risk assessment where required, organizational policies, access provisioning, encryption, key management, network security, incident response, retention/deletion, vendor governance and legal review.

Health and genetic/biometric information can fall within protected/specific personal-data categories under Indonesian law; the actual processing basis and safeguards must be assessed for the deployment.
