import os
import sys
import json
import numpy as np
import tensorflow as tf

# ============================================================
# Project paths
# File location: Backend/tests/calculate_final_model_loss.py
# BASE_DIR points to: Backend
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.ml.train_bilstm import Attention, CosineWarmup

MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "models", "bilstm_best.keras")
PROCESSED_DIR = os.path.join(BASE_DIR, "artifacts", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "artifacts", "reports")

os.makedirs(REPORTS_DIR, exist_ok=True)

NUM_CLASSES = 41
BATCH_SIZE = 64


def load_npy(filename):
    path = os.path.join(PROCESSED_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return np.load(path)


def ensure_one_hot(y, num_classes):
    if len(y.shape) == 1:
        return tf.keras.utils.to_categorical(y, num_classes=num_classes), y
    return y, np.argmax(y, axis=1)


def compute_cce_loss(y_true_onehot, y_pred_probs, label_smoothing=0.0):
    loss_fn = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False,
        label_smoothing=label_smoothing
    )
    return float(loss_fn(y_true_onehot, y_pred_probs).numpy())


# ============================================================
# Load data
# ============================================================
X_symptoms_val = load_npy("X_symptoms_val.npy")
X_severities_val = load_npy("X_severities_val.npy")
y_val_raw = load_npy("y_val.npy")

X_symptoms_test = load_npy("X_symptoms_test.npy")
X_severities_test = load_npy("X_severities_test.npy")
y_test_raw = load_npy("y_test.npy")

y_val_onehot, y_val = ensure_one_hot(y_val_raw, NUM_CLASSES)
y_test_onehot, y_test = ensure_one_hot(y_test_raw, NUM_CLASSES)


# ============================================================
# Load saved best model
# ============================================================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

custom_objects = {
    "Attention": Attention,
    "CosineWarmup": CosineWarmup,
}

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects=custom_objects,
    compile=False
)


# ============================================================
# Predict probabilities
# ============================================================
print("\nPredicting validation set...")
val_pred_probs = model.predict(
    [X_symptoms_val, X_severities_val],
    batch_size=BATCH_SIZE,
    verbose=1
)

print("\nPredicting test set...")
test_pred_probs = model.predict(
    [X_symptoms_test, X_severities_test],
    batch_size=BATCH_SIZE,
    verbose=1
)


# ============================================================
# Check output type
# ============================================================
val_prob_sum = np.sum(val_pred_probs, axis=1)
test_prob_sum = np.sum(test_pred_probs, axis=1)

val_outputs_are_probabilities = bool(np.allclose(val_prob_sum, 1.0, atol=1e-3))
test_outputs_are_probabilities = bool(np.allclose(test_prob_sum, 1.0, atol=1e-3))


# ============================================================
# Accuracy and confidence
# ============================================================
val_pred_classes = np.argmax(val_pred_probs, axis=1)
test_pred_classes = np.argmax(test_pred_probs, axis=1)

val_accuracy = float(np.mean(val_pred_classes == y_val))
test_accuracy = float(np.mean(test_pred_classes == y_test))

val_top_conf = np.max(val_pred_probs, axis=1)
test_top_conf = np.max(test_pred_probs, axis=1)


# ============================================================
# Loss calculations
# ============================================================
# 1) Plain CCE: easier to interpret for final classification confidence
val_loss_plain_cce = compute_cce_loss(
    y_val_onehot,
    val_pred_probs,
    label_smoothing=0.0
)

test_loss_plain_cce = compute_cce_loss(
    y_test_onehot,
    test_pred_probs,
    label_smoothing=0.0
)

# 2) Training-compatible CCE: matches train_bilstm.py configuration
val_loss_training_cce = compute_cce_loss(
    y_val_onehot,
    val_pred_probs,
    label_smoothing=0.1
)

test_loss_training_cce = compute_cce_loss(
    y_test_onehot,
    test_pred_probs,
    label_smoothing=0.1
)


# ============================================================
# Save report
# ============================================================
results = {
    "model_path": MODEL_PATH,
    "processed_data_path": PROCESSED_DIR,

    "validation_accuracy": val_accuracy,
    "test_accuracy": test_accuracy,

    "validation_plain_cce_loss_no_label_smoothing": val_loss_plain_cce,
    "test_plain_cce_loss_no_label_smoothing": test_loss_plain_cce,

    "validation_training_cce_loss_label_smoothing_0_1": val_loss_training_cce,
    "test_training_cce_loss_label_smoothing_0_1": test_loss_training_cce,

    "validation_avg_top_confidence": float(np.mean(val_top_conf)),
    "test_avg_top_confidence": float(np.mean(test_top_conf)),

    "validation_min_top_confidence": float(np.min(val_top_conf)),
    "test_min_top_confidence": float(np.min(test_top_conf)),

    "validation_max_top_confidence": float(np.max(val_top_conf)),
    "test_max_top_confidence": float(np.max(test_top_conf)),

    "validation_outputs_sum_to_1": val_outputs_are_probabilities,
    "test_outputs_sum_to_1": test_outputs_are_probabilities,

    "note": (
        "The model output layer uses Softmax, so predictions are probabilities. "
        "Plain CCE loss without label smoothing is reported for final classification confidence interpretation. "
        "Training-compatible CCE with label_smoothing=0.1 is also reported because the original model was trained "
        "with CategoricalCrossentropy(label_smoothing=0.1)."
    )
}

output_path = os.path.join(REPORTS_DIR, "final_model_loss_report.json")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)


# ============================================================
# Print results
# ============================================================
print("\n✅ Final model loss report saved to:")
print(output_path)

print("\n📌 Output Check:")
print(f"Validation outputs sum to 1: {val_outputs_are_probabilities}")
print(f"Test outputs sum to 1      : {test_outputs_are_probabilities}")

print("\n📌 Accuracy:")
print(f"Validation Accuracy: {val_accuracy:.6f}")
print(f"Test Accuracy      : {test_accuracy:.6f}")

print("\n📌 Confidence:")
print(f"Validation Avg Top Confidence: {np.mean(val_top_conf):.6f}")
print(f"Test Avg Top Confidence      : {np.mean(test_top_conf):.6f}")

print("\n📌 Plain CCE Loss without label smoothing:")
print(f"Validation Loss: {val_loss_plain_cce:.6f}")
print(f"Test Loss      : {test_loss_plain_cce:.6f}")

print("\n📌 Training-compatible CCE Loss with label smoothing = 0.1:")
print(f"Validation Loss: {val_loss_training_cce:.6f}")
print(f"Test Loss      : {test_loss_training_cce:.6f}")