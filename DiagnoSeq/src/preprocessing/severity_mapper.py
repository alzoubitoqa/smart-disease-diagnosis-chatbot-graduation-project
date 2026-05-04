import pandas as pd
from src.utils import save_json
from src.paths import KNOWLEDGE_DIR


def build_severity_map(df: pd.DataFrame) -> dict[str, int]:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    symptom_col = df.columns[0]
    weight_col = df.columns[1]

    severity_map = {}
    for _, row in df.iterrows():
        symptom = str(row[symptom_col]).strip().lower().replace(" ", "_")
        try:
            weight = int(row[weight_col])
        except Exception:
            weight = 1
        severity_map[symptom] = weight

    save_json(severity_map, KNOWLEDGE_DIR / "symptom_severity.json")
    return severity_map