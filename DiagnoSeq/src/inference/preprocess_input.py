import numpy as np
from src.config import MAX_LEN


def normalize_symptom(symptom: str) -> str:
    return symptom.strip().lower().replace(" ", "_")


def preprocess_basic_input(symptoms: list[str], vocab: dict[str, int]) -> np.ndarray:
    symptoms = [normalize_symptom(s) for s in symptoms if s.strip()]
    unk_id = vocab.get("<UNK>", 1)
    encoded = [vocab.get(s, unk_id) for s in symptoms[:MAX_LEN]]

    while len(encoded) < MAX_LEN:
        encoded.append(0)

    return np.array([encoded], dtype=np.int32)


def preprocess_deep_input(symptoms: list[str], vocab: dict[str, int], severity_map: dict[str, int]) -> np.ndarray:
    symptoms = [normalize_symptom(s) for s in symptoms if s.strip()]
    unk_id = vocab.get("<UNK>", 1)

    encoded = []
    for s in symptoms[:MAX_LEN]:
        encoded.append([vocab.get(s, unk_id), severity_map.get(s, 1)])

    while len(encoded) < MAX_LEN:
        encoded.append([0, 0])

    return np.array([encoded], dtype=np.int32)