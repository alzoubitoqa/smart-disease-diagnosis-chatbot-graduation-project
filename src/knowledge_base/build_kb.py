import json
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
KB_DIR = BASE_DIR / "artifacts" / "knowledge_base"

KB_DIR.mkdir(parents=True, exist_ok=True)


def normalize_key(text: str) -> str:
    return str(text).strip().lower().replace(" ", "_")


# =========================
# Load CSV files
# =========================

description_df = pd.read_csv(DATA_DIR / "symptom_Description.csv")
precaution_df = pd.read_csv(DATA_DIR / "symptom_precaution.csv")


# =========================
# Build descriptions
# =========================

descriptions = {}

for _, row in description_df.iterrows():
    key = normalize_key(row["Disease"])
    descriptions[key] = row["Description"]


# =========================
# Build precautions
# =========================

precautions = {}

for _, row in precaution_df.iterrows():
    key = normalize_key(row["Disease"])
    precautions[key] = [
        row.get("Precaution_1"),
        row.get("Precaution_2"),
        row.get("Precaution_3"),
        row.get("Precaution_4"),
    ]


# =========================
# Save JSON files
# =========================

with open(KB_DIR / "disease_descriptions.json", "w", encoding="utf-8") as f:
    json.dump(descriptions, f, ensure_ascii=False, indent=2)

with open(KB_DIR / "disease_precautions.json", "w", encoding="utf-8") as f:
    json.dump(precautions, f, ensure_ascii=False, indent=2)


print("✅ Knowledge Base created successfully!")
print(f"📁 Saved in: {KB_DIR}")