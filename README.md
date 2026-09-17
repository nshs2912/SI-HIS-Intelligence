# SI-HIS Intelligence

Smart Integrated Health Intelligence System — epidemiology analytics and machine-learning prototype.

## ML Engine
1. Case Severity Prediction
2. KLB / Outbreak 7-Day Prediction
3. Spatial Outbreak Prediction
4. Vulnerable Population Prediction
5. Robust Forecasting

## Existing analytics
Trias Epidemiologi, risk stratification, EWS, Rₜ, epidemic-wave analysis, DBSCAN spatial clustering, bivariate and multivariate analysis.

## Development
The repository uses the supplied `.devcontainer/devcontainer.json` with Python 3.11 Bookworm. Streamlit is exposed on port 8501.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Important validation note
The bundled simulation is for UI/pipeline testing. It is not evidence of clinical or epidemiological model performance. Production deployment requires historical labeled data, temporal validation, calibration, external validation, model monitoring, audit trail, privacy/security controls, and human oversight.
