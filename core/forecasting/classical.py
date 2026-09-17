"""Classical time-series forecast adapter."""
from ..analytics import prediksi_kurva_holt_winters


def holt_winters_forecast(epidemic_curve, forecast_days: int = 14):
    return prediksi_kurva_holt_winters(epidemic_curve, forecast_days)
