import re
import pandas as pd
from src.config import SYMPTOM_COLUMNS, TARGET_COLUMN


def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = text.replace(" ", "_")
    text = re.sub(r"_+", "_", text)
    text = re.sub(r"[^a-z0-9_()\-]", "", text)
    return text


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = [col.strip() for col in df.columns]

    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).apply(normalize_text)

    for col in SYMPTOM_COLUMNS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).apply(normalize_text)

    return df


def extract_symptom_sequence(row: pd.Series) -> list[str]:
    symptoms = []
    for col in SYMPTOM_COLUMNS:
        value = row.get(col, "")
        if value and value != "nan":
            symptoms.append(value)
    return symptoms