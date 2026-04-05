import json
import pickle
import numpy as np
import tensorflow as tf

from src.core.config import (
    BEST_MODEL_PATH,
    SYMPTOM2IDX_PATH,
    SEVERITY2IDX_PATH,
    LABEL_ENCODER_PATH,
    DISEASE_DESCRIPTIONS_PATH,
    DISEASE_PRECAUTIONS_PATH,
)

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


class DiseasePredictor:
    def __init__(self):
        self.model = tf.keras.models.load_model(BEST_MODEL_PATH, compile=False)

        with open(SYMPTOM2IDX_PATH, "rb") as f:
            self.symptom2idx = pickle.load(f)

        with open(SEVERITY2IDX_PATH, "rb") as f:
            self.severity2idx = pickle.load(f)

        with open(LABEL_ENCODER_PATH, "rb") as f:
            self.label_encoder = pickle.load(f)

        with open(DISEASE_DESCRIPTIONS_PATH, "r", encoding="utf-8") as f:
            self.disease_descriptions = json.load(f)

        with open(DISEASE_PRECAUTIONS_PATH, "r", encoding="utf-8") as f:
            self.disease_precautions = json.load(f)

        self.max_len = self.model.input_shape[0][1]

        # ✅ Manual mapping for user-friendly symptoms
        self.symptom_mapping = {
            "fever": "high_fever",
            "temperature": "high_fever",
            "high temperature": "high_fever",
            "sneezing": "continuous_sneezing",
            "runny nose": "runny_nose",
            "cold": "chills",
        }

    def _normalize_symptom(self, symptom: str) -> str:
        symptom = symptom.strip().lower().replace(",", " ").replace(".", " ")
        symptom = " ".join(symptom.split())
        symptom_underscore = symptom.replace(" ", "_")

        # direct mapping by spaced version
        if symptom in self.symptom_mapping:
            return self.symptom_mapping[symptom]

        # direct mapping by underscore version
        if symptom_underscore in self.symptom_mapping:
            return self.symptom_mapping[symptom_underscore]

        return symptom_underscore

    def _prepare_inputs(self, symptoms_with_severity):
        symptom_seq = []
        severity_seq = []

        for item in symptoms_with_severity[:self.max_len]:
            symptom = self._normalize_symptom(item["symptom"])
            severity = int(item["severity"])

            symptom_idx = self.symptom2idx.get(symptom, self.symptom2idx.get(UNK_TOKEN, 1))
            severity_idx = self.severity2idx.get(severity, 1)

            symptom_seq.append(symptom_idx)
            severity_seq.append(severity_idx)

        while len(symptom_seq) < self.max_len:
            symptom_seq.append(self.symptom2idx.get(PAD_TOKEN, 0))
            severity_seq.append(0)

        x_symptoms = np.array([symptom_seq], dtype=np.int32)
        x_severities = np.array([severity_seq], dtype=np.int32)
        return x_symptoms, x_severities

    def predict(self, symptoms_with_severity, top_k=3):
        x_symptoms, x_severities = self._prepare_inputs(symptoms_with_severity)
        probs = self.model.predict([x_symptoms, x_severities], verbose=0)[0]

        top_indices = np.argsort(probs)[::-1][:top_k]
        results = []

        for idx in top_indices:
            disease = self.label_encoder.inverse_transform([idx])[0]
            results.append({
                "disease": disease,
                "confidence": float(probs[idx]),
                "description": self.disease_descriptions.get(disease, ""),
                "precautions": self.disease_precautions.get(disease, {})
            })

        return {
            "top_prediction": results[0],
            "top_k_predictions": results
        }


predictor = DiseasePredictor()