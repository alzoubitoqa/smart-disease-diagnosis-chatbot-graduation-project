import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
KB_DIR = BASE_DIR / "artifacts" / "knowledge_base"

DESCRIPTION_PATH = KB_DIR / "disease_descriptions.json"


def normalize_key(text: str) -> str:
    return str(text).strip().lower().replace(" ", "_")


def load_disease_descriptions():
    if not DESCRIPTION_PATH.exists():
        return {}

    with open(DESCRIPTION_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {normalize_key(k): v for k, v in data.items()}


_DESCRIPTIONS = load_disease_descriptions()


def get_disease_description(disease_name: str) -> str:
    key = normalize_key(disease_name)
    return _DESCRIPTIONS.get(key, "")