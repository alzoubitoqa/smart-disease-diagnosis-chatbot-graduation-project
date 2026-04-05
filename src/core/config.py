import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

MODELS_DIR = BASE_DIR / "artifacts" / "models"
PROCESSED_DIR = BASE_DIR / "artifacts" / "processed"
REPORTS_DIR = BASE_DIR / "artifacts" / "reports"

BEST_MODEL_PATH = MODELS_DIR / "bilstm_final.keras"
SYMPTOM2IDX_PATH = MODELS_DIR / "symptom2idx.pkl"
SEVERITY2IDX_PATH = MODELS_DIR / "severity2idx.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"

DISEASE_DESCRIPTIONS_PATH = REPORTS_DIR / "disease_descriptions.json"
DISEASE_PRECAUTIONS_PATH = REPORTS_DIR / "disease_precautions.json"