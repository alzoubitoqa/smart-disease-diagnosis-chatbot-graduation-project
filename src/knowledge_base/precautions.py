import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
KB_DIR = BASE_DIR / "artifacts" / "knowledge_base"

PRECAUTIONS_PATH = KB_DIR / "disease_precautions.json"


def normalize_key(text: str) -> str:
    return str(text).strip().lower().replace(" ", "_")


def load_precautions():
    if not PRECAUTIONS_PATH.exists():
        return {}

    with open(PRECAUTIONS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {normalize_key(k): v for k, v in data.items()}


_PRECAUTIONS = load_precautions()


def get_disease_precautions(disease_name: str):
    key = normalize_key(disease_name)
    value = _PRECAUTIONS.get(key, {})

    if isinstance(value, list):
        return [p for p in value if p]

    if isinstance(value, dict):
        return [v for v in value.values() if v]

    return []