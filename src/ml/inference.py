import os
import numpy as np
import pickle
import tensorflow as tf
import math


# ============================================================================
# Custom Layers (must be defined before loading model)
# ============================================================================

class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='att_weight',
            shape=(input_shape[-1], 1),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='att_bias',
            shape=(input_shape[1], 1),
            initializer='zeros',
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def get_config(self):
        return super().get_config()


class CosineWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, learning_rate_base, warmup_steps, total_steps):
        super().__init__()
        self.learning_rate_base = learning_rate_base
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = step / self.warmup_steps
        cosine = 0.5 * (
            1 + tf.cos(
                math.pi * (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            )
        )
        lr = tf.where(step < self.warmup_steps, warmup, cosine)
        return self.learning_rate_base * lr

    def get_config(self):
        return {
            'learning_rate_base': self.learning_rate_base,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps
        }


# ============================================================================
# Paths
# ============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "artifacts", "models")

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "bilstm_final.keras")


# ============================================================================
# Predictor Class
# ============================================================================

class DiseasePredictor:
    def __init__(self):
        print("🔄 Loading disease predictor...")

        with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), 'rb') as f:
            self.label_encoder = pickle.load(f)

        with open(os.path.join(MODELS_DIR, "symptom2idx.pkl"), 'rb') as f:
            self.symptom2idx = pickle.load(f)

        with open(os.path.join(MODELS_DIR, "severity2idx.pkl"), 'rb') as f:
            self.severity2idx = pickle.load(f)

        custom_objects = {
            'Attention': Attention,
            'CosineWarmup': CosineWarmup
        }

        self.model = tf.keras.models.load_model(
            BEST_MODEL_PATH,
            custom_objects=custom_objects,
            compile=False
        )

        # compile optional, but kept for stability
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.max_len = 17
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.pad_index = self.symptom2idx.get(self.pad_token, 0)
        self.unk_index = self.symptom2idx.get(self.unk_token, 1)
        self.default_severity_token = self.severity2idx.get(1, 1)
        self.pad_severity_token = self.severity2idx.get(0, 0)

        print("✅ Disease predictor loaded successfully.")

    def normalize_symptom(self, symptom):
        if symptom is None:
            return ""

        symptom = str(symptom).strip().lower().replace(" ", "_")
        while "__" in symptom:
            symptom = symptom.replace("__", "_")
        return symptom

    def preprocess_symptoms(self, symptoms_list, severities_list=None):
        if not symptoms_list:
            raise ValueError("symptoms_list is empty.")

        normalized = [self.normalize_symptom(s) for s in symptoms_list if str(s).strip()]
        if not normalized:
            raise ValueError("No valid symptoms were provided after normalization.")

        if severities_list is None:
            severities_list = [1] * len(normalized)
        else:
            severities_list = [int(s) for s in severities_list]

        # Align lengths if any mismatch happens
        min_len = min(len(normalized), len(severities_list))
        normalized = normalized[:min_len]
        severities_list = severities_list[:min_len]

        if len(normalized) > self.max_len:
            normalized = normalized[:self.max_len]
            severities_list = severities_list[:self.max_len]
        else:
            pad_len = self.max_len - len(normalized)
            normalized += [self.pad_token] * pad_len
            severities_list += [0] * pad_len

        symptom_indices = []
        for symptom_name in normalized:
            if symptom_name == self.pad_token:
                symptom_indices.append(self.pad_index)
            else:
                symptom_indices.append(
                    self.symptom2idx.get(symptom_name, self.unk_index)
                )

        severity_indices = []
        for sev in severities_list:
            if sev == 0:
                severity_indices.append(self.pad_severity_token)
            else:
                severity_indices.append(
                    self.severity2idx.get(sev, self.default_severity_token)
                )

        X_symptoms = np.array([symptom_indices], dtype=np.int32)
        X_severities = np.array([severity_indices], dtype=np.int32)

        return X_symptoms, X_severities

    def predict(self, symptoms_list, severities_list=None, top_k=3):
        try:
            X_sym, X_sev = self.preprocess_symptoms(symptoms_list, severities_list)
            probs = self.model.predict([X_sym, X_sev], verbose=0)[0]

            top_k = max(1, min(int(top_k), len(probs)))
            top_indices = np.argsort(probs)[::-1][:top_k]

            results = [
                {
                    "disease": self.label_encoder.classes_[i],
                    "confidence": float(probs[i])
                }
                for i in top_indices
            ]

            return {
                "top_prediction": results[0],
                "top_k_predictions": results
            }

        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Prediction failed: {str(e)}")


# Singleton
predictor = DiseasePredictor()


# Optional test
if __name__ == "__main__":
    test_symptoms = ["vomiting", "headache", "nausea", "spinning_movements", "loss_of_balance", "unsteadiness"]
    test_severities = [5, 3, 5, 6, 4, 4]

    result = predictor.predict(test_symptoms, test_severities, top_k=3)
    print(result)