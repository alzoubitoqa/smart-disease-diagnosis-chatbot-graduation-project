import json
from pathlib import Path
from typing import Dict, List, Union, Optional

import pandas as pd

# ============================================================================
# Paths
# ============================================================================

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
KB_DIR = BASE_DIR / "artifacts" / "knowledge_base"
KB_DIR.mkdir(parents=True, exist_ok=True)

DESCRIPTION_PATH = KB_DIR / "disease_descriptions.json"
PRECAUTIONS_PATH = KB_DIR / "disease_precautions.json"
SEVERITY_PATH = KB_DIR / "symptom_severity.json"

# ============================================================================
# Helpers
# ============================================================================

def normalize_key(text: str) -> str:
    if pd.isna(text) or text is None:
        return ""

    text = str(text).strip().lower()

    # إزالة الأقواس
    text = text.replace("(", "").replace(")", "")

    # توحيد المسافات
    text = text.replace(" ", "_")

    # إزالة التكرارات المزعجة
    while "__" in text:
        text = text.replace("__", "_")

    # aliases مهمة جدًا لحل اختلاف أسماء الأمراض بين المودل والـ dataset
    aliases = {
        "vertigo_paroymsal_positional_vertigo": "paroxysmal_positional_vertigo",
        "vertigo_paroxysmal_positional_vertigo": "paroxysmal_positional_vertigo",
        "paroymsal_positional_vertigo": "paroxysmal_positional_vertigo",
        "paroxysmal_positional_vertigo": "paroxysmal_positional_vertigo",

        "dimorphic_hemorrhoids_piles": "dimorphic_hemorrhoids",
        "dimorphic_haemorrhoids_piles": "dimorphic_hemorrhoids",

        "peptic_ulcer_diseae": "peptic_ulcer_disease",

        "heartattack": "heart_attack",
        "heart_attack": "heart_attack",

        "fungalinfection": "fungal_infection",
        "commoncold": "common_cold",

        # alias مهم للمدخلات
        "fever": "high_fever",
        "high_temperature": "high_fever",
        "temperature": "high_fever",
    }

    return aliases.get(text, text)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ============================================================================
# Build KB from CSV
# ============================================================================

def build_json_from_csv() -> bool:
    try:
        desc_df = pd.read_csv(DATA_DIR / "symptom_Description.csv")
        prec_df = pd.read_csv(DATA_DIR / "symptom_precaution.csv")
        sev_df = pd.read_csv(DATA_DIR / "Symptom-severity.csv")
    except FileNotFoundError:
        try:
            desc_df = pd.read_csv(DATA_DIR / "symptom_Description.csv")
            prec_df = pd.read_csv(DATA_DIR / "symptom_precaution.csv")
            sev_df = pd.read_csv(DATA_DIR / "symptom-severity.csv")
        except FileNotFoundError as e:
            print(f"❌ Missing CSV file: {e}")
            return False

    descriptions = {}
    for _, row in desc_df.iterrows():
        key = normalize_key(row["Disease"])
        descriptions[key] = row["Description"]

    precautions = {}
    for _, row in prec_df.iterrows():
        key = normalize_key(row["Disease"])
        values = [
            row.get("Precaution_1"),
            row.get("Precaution_2"),
            row.get("Precaution_3"),
            row.get("Precaution_4"),
        ]
        precautions[key] = [v for v in values if pd.notna(v)]

    severity_weights = {}
    for _, row in sev_df.iterrows():
        key = normalize_key(row["Symptom"])
        severity_weights[key] = int(row["weight"])

    with open(DESCRIPTION_PATH, "w", encoding="utf-8") as f:
        json.dump(descriptions, f, ensure_ascii=False, indent=2)

    with open(PRECAUTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(precautions, f, ensure_ascii=False, indent=2)

    with open(SEVERITY_PATH, "w", encoding="utf-8") as f:
        json.dump(severity_weights, f, ensure_ascii=False, indent=2)

    print("✅ Knowledge base JSON files created successfully.")
    return True

# ============================================================================
# Lazy load
# ============================================================================

def _ensure_kb_files() -> None:
    if not (
        DESCRIPTION_PATH.exists()
        and PRECAUTIONS_PATH.exists()
        and SEVERITY_PATH.exists()
    ):
        build_json_from_csv()


_ensure_kb_files()

_DESCRIPTIONS = _load_json(DESCRIPTION_PATH)
_PRECAUTIONS = _load_json(PRECAUTIONS_PATH)
_SEVERITY_WEIGHTS = _load_json(SEVERITY_PATH)

# ============================================================================
# Getters
# ============================================================================

def get_disease_description(disease_name: str) -> str:
    key = normalize_key(disease_name)
    return _DESCRIPTIONS.get(key, "Description not available.")


def get_disease_precautions(disease_name: str) -> List[str]:
    key = normalize_key(disease_name)
    value = _PRECAUTIONS.get(key, [])

    if isinstance(value, list):
        return [v for v in value if v]

    if isinstance(value, dict):
        return [v for v in value.values() if v]

    return []


def get_symptom_base_weight(symptom_name: str) -> int:
    key = normalize_key(symptom_name)
    return int(_SEVERITY_WEIGHTS.get(key, 1))

# ============================================================================
# Severity Calculation
# ============================================================================

def calculate_severity(
    symptoms: List[str],
    user_severities: Optional[Dict[str, int]] = None,
    default_user_sev: int = 1
) -> Dict[str, Union[int, float, str, Dict[str, int]]]:
    """
    Severity summary should reflect the USER'S selected severity values,
    not (dataset weight × user severity).

    We still keep base_weights separately for reference.
    """
    if not symptoms:
        return {
            "total": 0,
            "avg": 0.0,
            "condition": "Mild",
            "weights": {},
            "base_weights": {}
        }

    user_severities = user_severities or {}
    total = 0
    weights = {}
    base_weights = {}

    for symptom in symptoms:
        norm_symptom = normalize_key(symptom)

        # مرجع من الداتا فقط
        base_weight = get_symptom_base_weight(norm_symptom)
        base_weights[norm_symptom] = base_weight

        # الشدة الفعلية التي اختارها المستخدم
        user_weight = int(user_severities.get(norm_symptom, default_user_sev))
        weights[norm_symptom] = user_weight
        total += user_weight

    avg = total / len(symptoms)

    # thresholds مناسبة لواجهة 1..5
    if avg < 2:
        condition = "Mild"
    elif avg < 4:
        condition = "Moderate"
    else:
        condition = "Severe"

    return {
        "total": total,
        "avg": round(avg, 2),
        "condition": condition,
        "weights": weights,
        "base_weights": base_weights
    }

# ============================================================================
# Optional Arabic translations
# ============================================================================

TRANSLATIONS = {
    "symptoms": {
        "abdominal_pain": "ألم في البطن",
        "anxiety": "قلق",
        "back_pain": "ألم في الظهر",
        "chest_pain": "ألم في الصدر",
        "cough": "سعال",
        "dizziness": "دوخة",
        "fatigue": "تعب",
        "fever": "حمى",
        "high_fever": "حمى",
        "headache": "صداع",
        "joint_pain": "ألم في المفاصل",
        "nausea": "غثيان",
        "vomiting": "تقيؤ",
        "weight_loss": "نقصان الوزن",
        "yellowish_skin": "اصفرار الجلد",
    },
    "diseases": {
        "fungal_infection": "عدوى فطرية",
        "allergy": "حساسية",
        "gerd": "ارتجاع المريء",
        "diabetes": "السكري",
        "hypertension": "ارتفاع ضغط الدم",
        "migraine": "الشقيقة",
        "heart_attack": "نوبة قلبية",
        "common_cold": "نزلة برد",
        "pneumonia": "ذات الرئة",
        "tuberculosis": "السل",
        "acne": "حب الشباب",
        "psoriasis": "الصدفية",
        "paroxysmal_positional_vertigo": "الدوار الوضعي الانتيابي",
    }
}


def translate_symptom(symptom: str) -> str:
    key = normalize_key(symptom)
    return TRANSLATIONS["symptoms"].get(key, symptom)


def translate_disease(disease: str) -> str:
    key = normalize_key(disease)
    return TRANSLATIONS["diseases"].get(key, disease)


if __name__ == "__main__":
    print("=" * 60)
    print("Knowledge Base Test")
    print("=" * 60)

    disease = "(vertigo)_paroymsal_positional_vertigo"
    print("Disease:", disease)
    print("Normalized:", normalize_key(disease))
    print("Description:", get_disease_description(disease))
    print("Precautions:", get_disease_precautions(disease))

    symptoms = ["fever", "cough", "fatigue"]
    user_input = {"high_fever": 2, "cough": 1, "fatigue": 3}
    severity_result = calculate_severity(symptoms, user_input)

    print("\nSeverity Result:")
    print(severity_result)

    print("\nTranslations:")
    print("fever ->", translate_symptom("fever"))
    print("vertigo ->", translate_disease("paroxysmal_positional_vertigo"))