import pandas as pd
from src.paths import RAW_DIR


def load_main_dataset() -> pd.DataFrame:
    path = RAW_DIR / "dataset.csv"
    return pd.read_csv(path)


def load_description_dataset() -> pd.DataFrame:
    path = RAW_DIR / "symptom_Description.csv"
    return pd.read_csv(path)


def load_precaution_dataset() -> pd.DataFrame:
    path = RAW_DIR / "symptom_precaution.csv"
    return pd.read_csv(path)


def load_severity_dataset() -> pd.DataFrame:
    path = RAW_DIR / "Symptom-severity.csv"
    return pd.read_csv(path)