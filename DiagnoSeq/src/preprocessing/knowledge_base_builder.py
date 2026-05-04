from src.paths import KNOWLEDGE_DIR
from src.utils import save_json


def build_description_map(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    disease_col = df.columns[0]
    desc_col = df.columns[1]

    result = {}
    for _, row in df.iterrows():
        disease = str(row[disease_col]).strip().lower().replace(" ", "_")
        desc = str(row[desc_col]).strip()
        result[disease] = desc

    save_json(result, KNOWLEDGE_DIR / "disease_description.json")
    return result


def build_precaution_map(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    disease_col = df.columns[0]
    precaution_cols = df.columns[1:]

    result = {}
    for _, row in df.iterrows():
        disease = str(row[disease_col]).strip().lower().replace(" ", "_")
        precautions = []
        for col in precaution_cols:
            val = str(row[col]).strip()
            if val and val.lower() != "nan":
                precautions.append(val)
        result[disease] = precautions

    save_json(result, KNOWLEDGE_DIR / "disease_precautions.json")
    return result