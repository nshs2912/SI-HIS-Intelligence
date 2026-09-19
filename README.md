# SI-HIS Intelligence

**Smart Integrated Health Intelligence System** — canonical epidemiology, statistics, ML and decision-intelligence engine.

## Core architecture

```text
Data Sources
    ↓
Data Provider / Data Hub
    ↓
Query Scope (isolated, read-only)
    ↓
SI-HIS Intelligence Engine
    ├── Epidemiology: Person · Place · Time
    ├── Mortality / CFR
    ├── Risk Stratification
    ├── EWS / KLB signal
    ├── Epidemic Curve / Rₜ / Wave
    ├── Spatial / DBSCAN
    ├── Bivariate / Multivariable Statistics
    ├── Vulnerable Population
    ├── Forecasting
    └── ML Prediction
    ↓
Intelligence API
    ↓
Kemenkes · Dinkes · Puskesmas · Doctor · NutriMed / Mobile
```

## Module responsibilities

| Module | Responsibility |
|---|---|
| `core/data_provider.py` | Single canonical data access boundary; returns isolated copies |
| `core/scope.py` | Province → district → kecamatan → village → puskesmas → disease scope |
| `core/analytics.py` | Core epidemiological/time/spatial analytics restored from validated prototype |
| `core/statistics.py` | Age grouping, bivariate and multivariable logistic analysis |
| `core/ml_engine.py` | ML training, prediction and robust forecasting |
| `core/engine.py` | Stateless facade that connects scope to analytics/ML |
| `core/national_dummy.py` | Synthetic national dataset generator |
| `core/orchestration.py` | Continuous intelligence lifecycle and routing contract |
| `api/` | External intelligence API boundary |
| `fhir/` | FHIR mapping, terminology and validation |
| `interoperability/` | SATUSEHAT/FHIR gateway components |
| `app.py` | Streamlit prototype/presentation layer |

## Isolation principle

A dashboard filter is a **query context**, not a mutation of the canonical dataset. Each request receives a deep-copy/query-local dataframe. Filtering Kabupaten Bandung therefore cannot change the data used by Kabupaten lain or another user request.

The production target is **stateless API + isolated query scope + cached/precomputed intelligence + worker-based heavy analytics**, so computationally expensive spatial/ML jobs do not block unrelated users.

## Age and risk-factor analysis

Age is not treated as one raw multi-category OR. It is converted to mutually-exclusive groups:
`<1`, `1-4`, `5-9`, `10-14`, `15-19`, `20-24`, `25-34`, `35-44`, `45-54`, `55-64`, `65-74`, `75-84`, `≥85` years.

Bivariate analysis returns crosstab, chi-square and age-specific OR/95% CI. Multivariable analysis uses categorical age groups plus available epidemiological risk variables and reports adjusted OR, confidence intervals and p-values.

## National dummy data

The bundled dataset is **synthetic only**. It is intended to exercise the full national analytics pipeline. The current national generator contains 38 provinces and 514 synthetic district records; administrative names/codes must be replaced by an authoritative government master before production integration. Do not interpret synthetic epidemiological values as real prevalence, incidence or outbreak evidence.

## ML engine

1. Case Severity Prediction
2. KLB / Outbreak 7-Day Prediction
3. Spatial Outbreak Prediction
4. Vulnerable Population Prediction
5. Robust Forecasting

ML output is decision support. It requires historical labelled data, temporal validation, calibration, external validation, drift monitoring, auditability and human oversight before operational use.

## Development

The repository uses the supplied `.devcontainer/devcontainer.json` with Python 3.11 Bookworm. Streamlit is exposed on port 8501.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Run core tests:

```bash
pytest -q
```

## Data source modes

The prototype now exposes two explicit data modes:

- DEMO / Synthetic Nasional — safe for presentations, development and testing.
- FACTUAL / Operational — accepts factual CSV/Excel through the canonical ingestion boundary during the prototype phase.

The factual source registry is designed for future adapters from SIMPUS, FKTP, hospital/RME, NutriMed MyLab Mobile, doctor/dietitian/physiotherapy networks, radiology, laboratory/LIS, pharmacy and other supporting services.

The intelligence engine is intentionally source-agnostic. IT integrations should map each source into the canonical SI-HIS schema before analytics are executed.

See docs/DATA_INTEGRATION_ARCHITECTURE.md for the adapter contract and recommended production flow.
