"""SI-HIS core compatibility, synthetic national data, and presentation helpers."""

from . import analytics as _analytics

# National synthetic dataset becomes the canonical demo source for the existing
# app.py API without rewriting the validated analytics implementation.
try:
    from .national_dummy import generate_national_dummy
    _analytics.generate_data_simulasi = generate_national_dummy
except Exception:
    generate_national_dummy = None

_original_hitung_bivariat_lengkap = _analytics.hitung_bivariat_lengkap

def _hitung_bivariat_compat(df, var_indep=None, var_dep_binary=None):
    if var_indep is not None and var_dep_binary is not None:
        return _original_hitung_bivariat_lengkap(df, var_indep, var_dep_binary)
    results = {}
    dependent = "Is_Konfirm"
    independent_vars = ["Umur", "Jenis Kelamin", "Pekerjaan", "Status Imunisasi", "Status Komorbid", "Riwayat Perjalanan", "Faktor Risiko Lain"]
    if dependent not in df.columns:
        return results
    for variable in independent_vars:
        if variable not in df.columns:
            continue
        try:
            results[variable] = _original_hitung_bivariat_lengkap(df.dropna(subset=[variable, dependent]), variable, dependent)
        except Exception as exc:
            results[variable] = {"error": str(exc)}
    return results

_analytics.hitung_bivariat_lengkap = _hitung_bivariat_compat

try:
    import folium
    if not getattr(folium.Map, "_sihis_epicenter_patch", False):
        _OriginalMap = folium.Map
        class SIHISEpicenterMap(_OriginalMap):
            _sihis_epicenter_patch = True
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                try:
                    lat, lon = float(self.location[0]), float(self.location[1])
                    folium.Circle(location=[lat, lon], radius=500, weight=2, fill=False, tooltip="Zona episentrum / centroid wilayah terfilter").add_to(self)
                    folium.Marker([lat, lon], tooltip="📍 Episentrum", popup=("<b>📍 Episentrum / Centroid</b><br>Titik pusat spasial dari koordinat kasus yang sedang difilter.<br>Bukan bukti sumber atau arah penularan."), icon=folium.Icon(color="red", icon="star")).add_to(self)
                except Exception:
                    pass
        folium.Map = SIHISEpicenterMap
except Exception:
    pass

hitung_bivariat_lengkap = _hitung_bivariat_compat
