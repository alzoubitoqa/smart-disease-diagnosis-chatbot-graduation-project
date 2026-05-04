from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
KNOWLEDGE_DIR = ARTIFACTS_DIR / "knowledge_base"
ENCODERS_DIR = ARTIFACTS_DIR / "encoders"

OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

def ensure_directories() -> None:
    dirs = [
        INTERIM_DIR,
        PROCESSED_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        KNOWLEDGE_DIR,
        ENCODERS_DIR,
        FIGURES_DIR,
        PROCESSED_DIR / "rnn",
        PROCESSED_DIR / "vanilla_lstm",
        PROCESSED_DIR / "deep_lstm",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)