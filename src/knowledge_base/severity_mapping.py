import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
KB_DIR = BASE_DIR / "artifacts" / "knowledge_base"

SEVERITY_PATH = KB_DIR / "symptom_severity.json"


def normalize_key(text: str) -> str:
    return str(text).strip().lower().replace(" ", "_")


def load_severity_mapping():
    if not SEVERITY_PATH.exists():
        return {}

    with open(SEVERITY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {normalize_key(k): v for k, v in data.items()}


_SEVERITY = load_severity_mapping()


def get_symptom_severity(symptom_name: str):
    key = normalize_key(symptom_name)
    return _SEVERITY.get(key)