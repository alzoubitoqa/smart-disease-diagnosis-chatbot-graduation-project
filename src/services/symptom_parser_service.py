import difflib
from pathlib import Path
from typing import List

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DATASET_PATH = DATA_DIR / "dataset.csv"


def _load_all_symptoms() -> List[str]:
    df = pd.read_csv(DATASET_PATH)

    all_symptoms = set()
    for col in df.columns:
        if "Symptom" in col:
            values = df[col].dropna().astype(str).str.strip().str.lower()
            values = values.str.replace(" ", "_", regex=False)
            all_symptoms.update(values.tolist())

    return sorted(list(all_symptoms))


ALL_SYMPTOMS = _load_all_symptoms()

# ✅ Manual mapping for user-friendly inputs
SYMPTOM_MAPPING = {
    "fever": "high_fever",
    "high temperature": "high_fever",
    "temperature": "high_fever",
    "sneezing": "continuous_sneezing",
    "runny nose": "runny_nose",
    "cold": "chills",
}


def normalize_user_symptom(symptom: str) -> str:
    symptom = symptom.strip().lower().replace(",", " ").replace(".", " ")
    symptom = " ".join(symptom.split())
    symptom_underscore = symptom.replace(" ", "_")

    # direct mapping by spaced version
    if symptom in SYMPTOM_MAPPING:
        return SYMPTOM_MAPPING[symptom]

    # direct mapping by underscore version
    if symptom_underscore in SYMPTOM_MAPPING:
        return SYMPTOM_MAPPING[symptom_underscore]

    return symptom_underscore


def correct_spelling(word: str):
    normalized = normalize_user_symptom(word)

    # if mapped symptom already exists in dataset, return it directly
    if normalized in ALL_SYMPTOMS:
        return normalized

    matches = difflib.get_close_matches(normalized, ALL_SYMPTOMS, n=1, cutoff=0.75)
    return matches[0] if matches else None


def extract_symptoms_from_text(text: str) -> List[str]:
    text = text.lower().strip().replace(",", " ").replace(".", " ")
    tokens = text.split()

    found = set()

    # single words
    for token in tokens:
        corrected = correct_spelling(token)
        if corrected:
            found.add(corrected)

    # two-word phrases
    for i in range(len(tokens) - 1):
        phrase = f"{tokens[i]} {tokens[i+1]}"
        corrected = correct_spelling(phrase)
        if corrected:
            found.add(corrected)

    # three-word phrases
    for i in range(len(tokens) - 2):
        phrase = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
        corrected = correct_spelling(phrase)
        if corrected:
            found.add(corrected)

    return list(found)