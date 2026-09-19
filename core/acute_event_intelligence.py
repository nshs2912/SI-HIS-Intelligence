"""Acute poisoning and mass-casualty intelligence for SI-HIS.

This module detects patterns compatible with point-source poisoning or other
acute clustered events. It does not declare a legal KLB or disaster status.
It separates signal detection from attribution and requires human
investigation, case definition, exposure history and laboratory confirmation.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AcuteEventProfile:
    event_family: str
    expected_pattern: str
    onset_window: str
    key_dimensions: tuple[str, ...]
    interpretation: str


POISONING_PROFILES = {
    "chemical": AcuteEventProfile(
        "KERACUNAN_KIMIA", "point_source_or_common_exposure",
        "minutes_to_hours", ("TIME","PLACE","EXPOSURE","PERSON"),
        "Onset dapat sangat cepat setelah paparan; interval onset yang rapat meningkatkan kecurigaan common exposure tetapi tidak membuktikan etiologi."
    ),
    "bacterial_toxin": AcuteEventProfile(
        "KERACUNAN_TOKSIN_BAKTERI", "point_source_or_foodborne",
        "hours_to_days", ("TIME","PLACE","FOOD","EXPOSURE","PERSON"),
        "Onset dapat berkelompok setelah konsumsi/paparan bersama; rentang onset perlu dibandingkan dengan distribusi masa inkubasi yang relevan."
    ),
    "unknown_toxic": AcuteEventProfile(
        "KERACUNAN_AKUT_UNSPECIFIED", "point_source_or_unknown_exposure",
        "minutes_to_days", ("TIME","PLACE","EXPOSURE","PERSON"),
        "Pola klaster akut dapat merupakan common exposure, tetapi sumber dan etiologi tidak dapat ditentukan dari kurva saja."
    ),
}


def _dates(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return pd.to_datetime(df[c], errors="coerce")
    return pd.Series(pd.NaT, index=df.index)


def classify_acute_event_type(df: pd.DataFrame, event_type: str | None = None) -> dict[str, Any]:
    key = str(event_type or "unknown_toxic").strip().lower()
    if key not in POISONING_PROFILES:
        key = "unknown_toxic"
    p = POISONING_PROFILES[key]
    return asdict(p)


def detect_point_source_cluster(
    df: pd.DataFrame,
    time_window_minutes: int = 60,
    min_cases: int = 5,
    geographic_columns: tuple[str, ...] = ("Provinsi","Kabupaten","Kecamatan","Desa/Kelurahan","Puskesmas"),
) -> dict[str, Any]:
    """Detect unusually dense onset clusters without assigning etiology."""
    onset = _dates(df, ["Tanggal Onset","Onset Date","onset_date","Tanggal Sakit"])
    valid = onset.notna()
    if valid.sum() < min_cases:
        return {"status":"insufficient_data","point_source_signal":False,"message":f"Kasus onset valid < {min_cases}."}

    work=df.loc[valid].copy()
    work["_onset"]=onset.loc[valid]
    work=work.sort_values("_onset")
    best=None
    window=pd.Timedelta(minutes=time_window_minutes)

    for i,row in work.iterrows():
        end=row["_onset"]+window
        mask=(work["_onset"]>=row["_onset"])&(work["_onset"]<=end)
        n=int(mask.sum())
        if best is None or n>best["cases_in_window"]:
            best={"start":row["_onset"],"end":end,"cases_in_window":n,"indices":work.index[mask].tolist()}

    if best is None:
        return {"status":"no_signal","point_source_signal":False}

    cluster=work.loc[best["indices"]].copy()
    geo_summary={}
    for col in geographic_columns:
        if col in cluster.columns:
            geo_summary[col]=cluster[col].astype(str).value_counts().head(5).to_dict()

    concentration=float(best["cases_in_window"]/max(len(work),1))
    signal=best["cases_in_window"]>=min_cases
    return {
        "status":"signal" if signal else "no_signal",
        "point_source_signal":signal,
        "time_window_minutes":time_window_minutes,
        "cases_in_window":best["cases_in_window"],
        "cluster_start":best["start"],
        "cluster_end":best["end"],
        "concentration_pct":round(concentration*100,2),
        "geography":geo_summary,
        "interpretation":"Klaster onset yang sangat rapat secara waktu konsisten dengan pola point-source/common exposure dan perlu investigasi segera; bukan bukti etiologi atau penetapan KLB.",
    }


def detect_mass_casualty_burst(
    df: pd.DataFrame,
    baseline_days: int = 28,
    burst_window_minutes: int = 60,
    min_cases: int = 10,
) -> dict[str, Any]:
    """Compare an acute time burst with the historical daily baseline."""
    onset=_dates(df,["Tanggal Onset","Onset Date","onset_date","Tanggal Sakit"])
    onset=onset.dropna().sort_values()
    if len(onset)<min_cases:
        return {"status":"insufficient_data","signal":False}

    daily=onset.dt.floor("D").value_counts().sort_index()
    recent_date=daily.index.max()
    recent_window=onset[(onset>=recent_date)&(onset<recent_date+pd.Timedelta(days=1))]
    # Use a conservative historical daily baseline excluding the latest date.
    hist=daily[daily.index<recent_date].tail(baseline_days)
    baseline=float(hist.mean()) if len(hist) else 0.0
    burst=int(len(recent_window))
    ratio=burst/max(baseline,1.0)
    signal=burst>=min_cases and ratio>=3.0
    return {
        "status":"signal" if signal else "no_signal",
        "signal":signal,
        "latest_date":recent_date,
        "latest_day_cases":burst,
        "baseline_daily_mean":round(baseline,2),
        "burst_to_baseline_ratio":round(ratio,2),
        "interpretation":"Lonjakan mendadak dibanding baseline dapat menandakan acute event, mass gathering, outbreak, poisoning/common exposure, atau bencana; klasifikasi penyebab memerlukan data paparan dan investigasi.",
    }


def analyze_exposure_window(df: pd.DataFrame) -> dict[str, Any]:
    onset=_dates(df,["Tanggal Onset","Onset Date","onset_date","Tanggal Sakit"])
    exposure=_dates(df,["Tanggal Paparan","Exposure Date","exposure_datetime"])
    valid=onset.notna()&exposure.notna()
    if valid.sum()==0:
        return {"status":"unavailable","message":"Tanggal paparan belum tersedia."}
    delay=(onset[valid]-exposure[valid]).dt.total_seconds()/3600
    return {
        "status":"ok",
        "n":int(len(delay)),
        "median_hours":round(float(delay.median()),2),
        "p10_hours":round(float(delay.quantile(.10)),2),
        "p90_hours":round(float(delay.quantile(.90)),2),
        "negative_delay_n":int((delay<0).sum()),
        "interpretation":"Distribusi onset-paparan membantu membedakan hipotesis common exposure akut dari pola lain, tetapi harus dibandingkan dengan karakteristik agen dan kualitas waktu paparan.",
    }


def poisoning_vs_disaster_signal(df: pd.DataFrame) -> dict[str, Any]:
    """Generate a neutral differential signal, never a final classification."""
    cluster=detect_point_source_cluster(df)
    burst=detect_mass_casualty_burst(df)
    exposure=analyze_exposure_window(df)
    flags=[]
    if cluster.get("point_source_signal"): flags.append("POINT_SOURCE_COMMON_EXPOSURE")
    if burst.get("signal"): flags.append("MASS_CASUALTY_BURST")
    if exposure.get("status")=="ok" and exposure.get("median_hours") is not None and exposure.get("median_hours")<=6: flags.append("RAPID_ONSET_EXPOSURE_PATTERN")

    if not flags:
        differential=["Tidak ada sinyal kuat acute clustered event dari waktu yang tersedia."]
    else:
        differential=[
            "Keracunan/common exposure",
            "Foodborne/toxin-associated event",
            "Bencana non-infeksi/chemical/environmental exposure",
            "Mass gathering atau kejadian dengan paparan bersama",
            "Outbreak infeksi akut dengan onset yang berdekatan",
        ]
    priority = "URGENT_FIELD_VERIFICATION" if flags else "ROUTINE_SURVEILLANCE"
    return {
        "status":"signal" if flags else "no_signal",
        "priority":priority,
        "signals":flags,
        "differential_hypotheses":differential,
        "point_source":cluster,
        "mass_casualty_burst":burst,
        "exposure_window":exposure,
        "guardrail":"Signal detector tidak membedakan secara definitif keracunan, bencana, atau outbreak. Perlu line list, waktu dan lokasi paparan, gejala, attack rate, pemeriksaan klinis, laboratorium/toksikologi, serta investigasi lapangan.",
    }


# Clinical-toxicology knowledge base. These are candidate-pattern profiles for
# epidemiological differential support; they are not diagnostic rules.
TOXICOLOGY_AGENT_PROFILES = [
    {
        "agent": "Staphylococcus aureus enterotoxin",
        "agent_type": "bacterial_preformed_toxin",
        "route": ["foodborne_ingestion"],
        "typical_onset": "30 minutes–8 hours",
        "clinical_syndrome": ["sudden nausea", "prominent vomiting", "abdominal cramps", "diarrhea", "fever uncommon"],
        "exposure_clues": ["food handled after cooking", "sandwiches", "pastries", "sliced meats"],
    },
    {
        "agent": "Bacillus cereus — emetic toxin",
        "agent_type": "bacterial_preformed_toxin",
        "route": ["foodborne_ingestion"],
        "typical_onset": "1–6 hours",
        "clinical_syndrome": ["prominent vomiting", "nausea", "abdominal symptoms", "diarrhea may occur", "fever uncommon"],
        "exposure_clues": ["cooked rice", "starchy foods", "improper temperature control"],
    },
    {
        "agent": "Bacillus cereus — diarrheal toxin",
        "agent_type": "bacterial_toxin",
        "route": ["foodborne_ingestion"],
        "typical_onset": "6–24 hours",
        "clinical_syndrome": ["diarrhea", "abdominal cramps", "vomiting sometimes", "fever uncommon"],
        "exposure_clues": ["meat", "vegetables", "sauces", "foods held at unsafe temperatures"],
    },
    {
        "agent": "Clostridium perfringens toxin",
        "agent_type": "bacterial_toxin",
        "route": ["foodborne_ingestion"],
        "typical_onset": "6–24 hours",
        "clinical_syndrome": ["diarrhea", "abdominal cramps", "vomiting uncommon", "fever uncommon"],
        "exposure_clues": ["meat", "poultry", "gravy", "large-batch foods"],
    },
    {
        "agent": "Clostridium botulinum neurotoxin",
        "agent_type": "bacterial_preformed_neurotoxin",
        "route": ["foodborne_ingestion", "wound", "other_exposure"],
        "typical_onset": "usually 18–36 hours for foodborne illness",
        "clinical_syndrome": ["diplopia", "blurred vision", "ptosis", "dysarthria", "dysphagia", "descending weakness", "respiratory compromise"],
        "exposure_clues": ["improperly canned or fermented foods"],
    },
    {
        "agent": "Salmonella",
        "agent_type": "bacterial_infection",
        "route": ["foodborne_ingestion", "waterborne_ingestion"],
        "typical_onset": "6 hours–6 days",
        "clinical_syndrome": ["diarrhea", "fever", "abdominal cramps", "vomiting"],
        "exposure_clues": ["poultry", "eggs", "meat", "unpasteurized milk or juice", "reptiles"],
    },
    {
        "agent": "Campylobacter",
        "agent_type": "bacterial_infection",
        "route": ["foodborne_ingestion", "waterborne_ingestion"],
        "typical_onset": "2–5 days",
        "clinical_syndrome": ["diarrhea often bloody", "fever", "abdominal cramps"],
        "exposure_clues": ["undercooked poultry", "unpasteurized milk", "contaminated water"],
    },
    {
        "agent": "Shiga toxin-producing Escherichia coli (STEC)",
        "agent_type": "bacterial_toxin_associated_infection",
        "route": ["foodborne_ingestion", "waterborne_ingestion"],
        "typical_onset": "about 3–4 days",
        "clinical_syndrome": ["severe abdominal cramps", "diarrhea often bloody", "vomiting", "possible hemolytic uremic syndrome"],
        "exposure_clues": ["undercooked ground beef", "unpasteurized milk or juice", "raw vegetables", "sprouts", "contaminated water"],
    },
    {
        "agent": "Vibrio spp.",
        "agent_type": "bacterial_infection",
        "route": ["foodborne_ingestion", "waterborne_ingestion"],
        "typical_onset": "within about 24 hours for some foodborne syndromes",
        "clinical_syndrome": ["watery diarrhea", "nausea", "abdominal cramps", "vomiting", "fever/chills may occur"],
        "exposure_clues": ["raw or undercooked shellfish", "marine exposure"],
    },
    {
        "agent": "Norovirus",
        "agent_type": "viral_infection",
        "route": ["foodborne_ingestion", "waterborne_ingestion", "person_to_person"],
        "typical_onset": "12–48 hours",
        "clinical_syndrome": ["vomiting", "diarrhea", "nausea", "abdominal pain", "fever/headache/body aches may occur"],
        "exposure_clues": ["shared food", "contaminated water", "close-contact setting"],
    },
    {
        "agent": "Ciguatoxin",
        "agent_type": "marine_biotoxin",
        "route": ["foodborne_ingestion"],
        "typical_onset": "1–48 hours, often 2–8 hours",
        "clinical_syndrome": ["gastrointestinal symptoms", "paresthesia", "neurologic symptoms", "hot-cold sensation reversal"],
        "exposure_clues": ["reef fish such as snapper, grouper, barracuda"],
    },
    {
        "agent": "Scombroid (histamine)",
        "agent_type": "marine_biotoxin",
        "route": ["foodborne_ingestion"],
        "typical_onset": "minutes–3 hours, often <1 hour",
        "clinical_syndrome": ["flushing", "headache", "burning mouth/throat", "urticaria", "pruritus", "gastrointestinal symptoms", "dizziness"],
        "exposure_clues": ["histamine-rich spoiled fish"],
    },
    {
        "agent": "Mushroom toxins",
        "agent_type": "natural_toxin",
        "route": ["foodborne_ingestion"],
        "typical_onset": "varies from hours to longer latent periods depending on toxin",
        "clinical_syndrome": ["vomiting/diarrhea", "neurologic or autonomic features", "possible hepatic or renal injury for delayed toxins"],
        "exposure_clues": ["wild or unidentified mushrooms"],
    },
    {
        "agent": "Organophosphate/carbamate pesticide syndrome",
        "agent_type": "chemical",
        "route": ["foodborne_ingestion", "inhalation", "dermal", "mixed"],
        "typical_onset": "often rapid after significant exposure",
        "clinical_syndrome": ["salivation", "lacrimation", "urination", "diarrhea", "vomiting", "bronchorrhea/bronchospasm", "miosis", "bradycardia", "fasciculations", "weakness"],
        "exposure_clues": ["agricultural pesticide", "spraying", "contaminated food/water", "mixed-route exposure"],
    },
    {
        "agent": "Chlorine/chloramine respiratory irritant",
        "agent_type": "chemical",
        "route": ["inhalation"],
        "typical_onset": "minutes to hours",
        "clinical_syndrome": ["eye irritation", "cough", "throat irritation", "dyspnea", "wheeze", "chest tightness"],
        "exposure_clues": ["pool/cleaning chemical release", "enclosed-space exposure"],
    },
    {
        "agent": "Ammonia respiratory irritant",
        "agent_type": "chemical",
        "route": ["inhalation"],
        "typical_onset": "minutes to hours",
        "clinical_syndrome": ["eye/nose/throat irritation", "cough", "wheeze", "dyspnea", "chest tightness"],
        "exposure_clues": ["industrial/refrigeration/cleaning exposure"],
    },
    {
        "agent": "Carbon monoxide",
        "agent_type": "chemical_asphyxiant",
        "route": ["inhalation"],
        "typical_onset": "minutes to hours depending on concentration",
        "clinical_syndrome": ["headache", "dizziness", "weakness", "nausea", "confusion", "altered consciousness"],
        "exposure_clues": ["enclosed-space combustion", "generator", "vehicle exhaust", "multiple persons affected in same environment"],
    },
    {
        "agent": "Cyanide",
        "agent_type": "chemical_systemic_toxicant",
        "route": ["inhalation", "ingestion", "mixed"],
        "typical_onset": "often very rapid in significant exposure",
        "clinical_syndrome": ["altered mental status", "respiratory distress", "hypotension", "seizures", "cardiovascular collapse"],
        "exposure_clues": ["industrial/fire/specific chemical exposure context"],
    },
]


def toxicology_agent_profiles() -> list[dict[str, Any]]:
    """Return the agent-pattern catalogue used for differential support."""
    return [dict(item) for item in TOXICOLOGY_AGENT_PROFILES]


def _text_columns(df: pd.DataFrame) -> pd.Series:
    cols = [
        c for c in df.columns
        if any(token in str(c).lower() for token in (
            "gejala", "symptom", "tanda", "sign", "keluhan", "paparan",
            "exposure", "makanan", "food", "pajanan", "anamnesis", "catatan"
        ))
    ]
    if not cols:
        return pd.Series("", index=df.index)
    return df[cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()


def toxicology_differential(df: pd.DataFrame) -> dict[str, Any]:
    """Rank compatible agent patterns from route, symptoms and onset.

    This is a differential-support layer, not a diagnosis. Candidate agents
    require clinical, epidemiological and laboratory/toxicology confirmation.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"status": "insufficient_data", "candidates": [], "profiles": toxicology_agent_profiles()}

    text = _text_columns(df)
    route_text = " ".join(text.tolist())
    onset = _dates(df, ["Tanggal Onset", "Onset Date", "onset_date", "Tanggal Sakit"])
    exposure = _dates(df, ["Tanggal Paparan", "Exposure Date", "exposure_datetime"])
    delays = ((onset - exposure).dt.total_seconds() / 3600).dropna() if onset.notna().any() and exposure.notna().any() else pd.Series(dtype=float)
    median_hours = float(delays.median()) if not delays.empty else None

    route_keywords = {
        "foodborne_ingestion": ["makan", "makanan", "food", "ingest", "minum", "meal"],
        "inhalation": ["hirup", "inhalasi", "asap", "gas", "uap", "fume", "inhal"],
        "dermal": ["kulit", "dermal", "terpapar kulit", "splash"],
        "waterborne_ingestion": ["air minum", "water", "air tercemar"],
    }

    route_hits = {route: sum(route_text.count(k) for k in keys) for route, keys in route_keywords.items()}
    detected_routes = [r for r, n in route_hits.items() if n > 0]

    def score(profile: dict[str, Any]) -> tuple[float, list[str]]:
        score_value = 0.0
        evidence = []
        routes = set(profile["route"])
        route_overlap = routes.intersection(detected_routes)
        if route_overlap:
            score_value += 3.0
            evidence.append("route=" + ",".join(sorted(route_overlap)))

        symptom_hits = 0
        for symptom in profile["clinical_syndrome"]:
            token = symptom.lower().split("(")[0].strip()
            if token and token in route_text:
                symptom_hits += 1
        if symptom_hits:
            score_value += min(5.0, float(symptom_hits))
            evidence.append(f"clinical_features={symptom_hits}")

        if median_hours is not None:
            # Broad compatibility windows used only to prioritize candidates.
            onset_text = profile["typical_onset"]
            if ("minutes" in onset_text or "min" in onset_text) and median_hours <= 3:
                score_value += 2.0
                evidence.append("rapid_onset")
            elif "1–6 hours" in onset_text and 1 <= median_hours <= 6:
                score_value += 2.0
                evidence.append("onset_compatible")
            elif "6–24 hours" in onset_text and 6 <= median_hours <= 24:
                score_value += 2.0
                evidence.append("onset_compatible")
            elif "12–48 hours" in onset_text and 12 <= median_hours <= 48:
                score_value += 2.0
                evidence.append("onset_compatible")
            elif "2–5 days" in onset_text and 48 <= median_hours <= 120:
                score_value += 2.0
                evidence.append("onset_compatible")
            elif "3–4 days" in onset_text and 48 <= median_hours <= 120:
                score_value += 2.0
                evidence.append("onset_compatible")

        return score_value, evidence

    ranked = []
    for profile in TOXICOLOGY_AGENT_PROFILES:
        s, evidence = score(profile)
        if s > 0:
            ranked.append({
                "agent": profile["agent"],
                "agent_type": profile["agent_type"],
                "score": round(s, 2),
                "evidence": evidence,
                "typical_onset": profile["typical_onset"],
                "clinical_syndrome": profile["clinical_syndrome"],
                "route": profile["route"],
                "exposure_clues": profile["exposure_clues"],
                "interpretation": "Kandidat diferensial; bukan diagnosis dan harus dikonfirmasi dengan investigasi klinis, epidemiologis, serta pemeriksaan laboratorium/toksikologi yang sesuai.",
            })
    ranked.sort(key=lambda x: (-x["score"], x["agent"]))
    return {
        "status": "candidate_differential" if ranked else "no_candidate_pattern",
        "detected_routes": detected_routes,
        "median_exposure_to_onset_hours": round(median_hours, 2) if median_hours is not None else None,
        "candidates": ranked[:10],
        "profiles": toxicology_agent_profiles(),
        "guardrail": "Agen spesifik hanya kandidat pola. Penetapan etiologi memerlukan case definition, riwayat paparan, pemeriksaan klinis, spesimen biologis/environmental, dan/atau konfirmasi laboratorium/toksikologi.",
    }
