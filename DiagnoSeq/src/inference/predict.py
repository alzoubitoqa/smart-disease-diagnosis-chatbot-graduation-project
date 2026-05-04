import numpy as np
from tensorflow.keras.models import load_model

from src.paths import MODELS_DIR, INTERIM_DIR, KNOWLEDGE_DIR
from src.utils import load_json
from src.inference.preprocess_input import preprocess_basic_input, preprocess_deep_input


class DiseasePredictor:
    def __init__(self, model_name: str = "deep_lstm"):
        self.model_name = model_name

        self.vocab = load_json(INTERIM_DIR / "symptom_vocab.json")
        self.id_to_disease = load_json(INTERIM_DIR / "id_to_disease.json")
        self.severity_map = load_json(KNOWLEDGE_DIR / "symptom_severity.json")
        self.description_map = load_json(KNOWLEDGE_DIR / "disease_description.json")
        self.precaution_map = load_json(KNOWLEDGE_DIR / "disease_precautions.json")

        if model_name == "rnn":
            self.model = load_model(MODELS_DIR / "rnn_best.keras")
        elif model_name == "vanilla_lstm":
            self.model = load_model(MODELS_DIR / "vanilla_lstm_best.keras")
        else:
            self.model = load_model(MODELS_DIR / "deep_lstm_best.keras")

    def predict(self, symptoms: list[str], top_k: int = 3):
        if self.model_name in ["rnn", "vanilla_lstm"]:
            X = preprocess_basic_input(symptoms, self.vocab)
        else:
            X = preprocess_deep_input(symptoms, self.vocab, self.severity_map)

        probs = self.model.predict(X, verbose=0)[0]
        top_indices = np.argsort(probs)[::-1][:top_k]

        results = []
        for idx in top_indices:
            disease = self.id_to_disease.get(str(idx), self.id_to_disease.get(idx, "unknown_disease"))
            results.append({
                "disease": disease,
                "probability": float(probs[idx]),
                "description": self.description_map.get(disease, "No description available."),
                "precautions": self.precaution_map.get(disease, []),
            })

        return results