import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
KB_DIR = BASE_DIR / "artifacts" / "knowledge_base"
SEVERITY_PATH = KB_DIR / "symptom_severity.json"


def format_symptom_label(symptom: str) -> str:
    return symptom.replace("_", " ").title()


def get_all_symptoms():
    if not SEVERITY_PATH.exists():
        return []

    with open(SEVERITY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    symptoms = sorted(list(data.keys()))

    base_items = [
        {
            "value": symptom,
            "label": format_symptom_label(symptom)
        }
        for symptom in symptoms
    ]

    # ✅ aliases user may type, mapped to real model symptoms
    aliases = [
        {"value": "high_fever", "label": "Fever"},
        {"value": "high_fever", "label": "High Temperature"},
        {"value": "continuous_sneezing", "label": "Sneezing"},
        {"value": "runny_nose", "label": "Runny Nose"},
        {"value": "chills", "label": "Cold"},
    ]

    # منع التكرار بنفس label
    seen = set()
    final_items = []

    for item in base_items + aliases:
        key = (item["value"], item["label"].lower())
        if key not in seen:
            seen.add(key)
            final_items.append(item)

    return final_items