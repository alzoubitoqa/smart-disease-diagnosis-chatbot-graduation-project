# ============================================================================
# EVALUATE SAVED BiLSTM MODEL WITHOUT RETRAINING
# Fixes classification_report issue when test set contains fewer classes
# ============================================================================

import os
import sys
import json
import math
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score
)

import tensorflow as tf
from tensorflow.keras.layers import Layer


# ============================================================================
# 1. Custom Objects Needed to Load Saved Model
# ============================================================================

class Attention(Layer):
    """Custom attention layer used in the trained model."""
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
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


class CosineWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule used during training."""
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
                math.pi * (step - self.warmup_steps) /
                (self.total_steps - self.warmup_steps)
            )
        )
        lr = tf.where(step < self.warmup_steps, warmup, cosine)
        return self.learning_rate_base * lr

    def get_config(self):
        return {
            "learning_rate_base": self.learning_rate_base,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps
        }


# ============================================================================
# 2. Paths
# ============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, "artifacts", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "artifacts", "models")
REPORTS_DIR = os.path.join(BASE_DIR, "artifacts", "reports")

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "bilstm_best.keras")
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, "bilstm_final.keras")

os.makedirs(REPORTS_DIR, exist_ok=True)


# ============================================================================
# 3. Load Data and Model
# ============================================================================

def load_test_data():
    print("\n📂 Loading processed test data...")

    X_symptoms_test = np.load(os.path.join(PROCESSED_DIR, "X_symptoms_test.npy"))
    X_severities_test = np.load(os.path.join(PROCESSED_DIR, "X_severities_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

    with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)

    num_classes = len(label_encoder.classes_)

    print(f"   - Test samples: {len(y_test)}")
    print(f"   - Number of total disease classes: {num_classes}")
    print(f"   - Classes present in test set: {len(np.unique(y_test))}")

    return X_symptoms_test, X_severities_test, y_test, label_encoder, num_classes


def load_saved_model():
    print("\n📦 Loading saved trained model...")

    model_path = BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else FINAL_MODEL_PATH

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No saved model found at:\n{BEST_MODEL_PATH}\nor\n{FINAL_MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "Attention": Attention,
            "CosineWarmup": CosineWarmup
        },
        compile=False
    )

    print(f"✅ Model loaded from: {model_path}")
    return model


# ============================================================================
# 4. Evaluation
# ============================================================================

def evaluate_saved_model():
    X_symptoms_test, X_severities_test, y_test, label_encoder, num_classes = load_test_data()
    model = load_saved_model()

    print("\n🔮 Making predictions...")
    y_pred_probs = model.predict([X_symptoms_test, X_severities_test], verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    all_labels = np.arange(num_classes)
    present_labels = np.unique(y_test)

    # Metrics over classes that actually appear in the test set
    test_acc = accuracy_score(y_test, y_pred)

    precision_macro_present = precision_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    recall_macro_present = recall_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    f1_macro_present = f1_score(
        y_test, y_pred, average="macro", zero_division=0
    )

    precision_weighted = precision_score(
        y_test, y_pred, average="weighted", zero_division=0
    )
    recall_weighted = recall_score(
        y_test, y_pred, average="weighted", zero_division=0
    )
    f1_weighted = f1_score(
        y_test, y_pred, average="weighted", zero_division=0
    )

    # Metrics over all 41 classes, including classes absent from test set
    precision_macro_all = precision_score(
        y_test, y_pred, labels=all_labels, average="macro", zero_division=0
    )
    recall_macro_all = recall_score(
        y_test, y_pred, labels=all_labels, average="macro", zero_division=0
    )
    f1_macro_all = f1_score(
        y_test, y_pred, labels=all_labels, average="macro", zero_division=0
    )

    top2_acc = top_k_accuracy_score(y_test, y_pred_probs, k=2, labels=all_labels)
    top3_acc = top_k_accuracy_score(y_test, y_pred_probs, k=3, labels=all_labels)
    top5_acc = top_k_accuracy_score(y_test, y_pred_probs, k=5, labels=all_labels)

    cm = confusion_matrix(y_test, y_pred, labels=all_labels)

    print("\n" + "=" * 70)
    print("📊 SAVED MODEL EVALUATION RESULTS")
    print("=" * 70)

    print("\n1️⃣ BASIC TEST METRICS")
    print("-" * 40)
    print(f"   Test Accuracy: {test_acc * 100:.4f}%")
    print(f"   Precision Macro - Present Test Classes: {precision_macro_present:.6f}")
    print(f"   Recall Macro - Present Test Classes: {recall_macro_present:.6f}")
    print(f"   F1 Macro - Present Test Classes: {f1_macro_present:.6f}")
    print(f"   Precision Weighted: {precision_weighted:.6f}")
    print(f"   Recall Weighted: {recall_weighted:.6f}")
    print(f"   F1 Weighted: {f1_weighted:.6f}")

    print("\n2️⃣ ALL-CLASS MACRO METRICS")
    print("-" * 40)
    print("These include classes absent from the test set with zero support.")
    print(f"   Precision Macro - All 41 Classes: {precision_macro_all:.6f}")
    print(f"   Recall Macro - All 41 Classes: {recall_macro_all:.6f}")
    print(f"   F1 Macro - All 41 Classes: {f1_macro_all:.6f}")

    print("\n3️⃣ TOP-K ACCURACY")
    print("-" * 40)
    print(f"   Top-2 Accuracy: {top2_acc * 100:.4f}%")
    print(f"   Top-3 Accuracy: {top3_acc * 100:.4f}%")
    print(f"   Top-5 Accuracy: {top5_acc * 100:.4f}%")

    correct_predictions = int(np.trace(cm))
    wrong_predictions = int(np.sum(cm) - np.trace(cm))

    print("\n4️⃣ PREDICTION COUNTS")
    print("-" * 40)
    print(f"   Correct Predictions: {correct_predictions}")
    print(f"   Wrong Predictions: {wrong_predictions}")
    print(f"   Total Test Samples: {len(y_test)}")

    print("\n5️⃣ CLASSIFICATION REPORT")
    print("-" * 70)

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=all_labels,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report_dict).transpose()
    report_path = os.path.join(REPORTS_DIR, "classification_report_strict_split.csv")
    report_df.to_csv(report_path)

    print(report_df.round(6).head(45))
    print(f"\n✅ Classification report saved to: {report_path}")

    # Save confusion matrix
    cm_df = pd.DataFrame(
        cm,
        index=label_encoder.classes_,
        columns=label_encoder.classes_
    )
    cm_csv_path = os.path.join(REPORTS_DIR, "confusion_matrix_strict_split.csv")
    cm_df.to_csv(cm_csv_path)

    print(f"✅ Confusion matrix CSV saved to: {cm_csv_path}")

    # Plot confusion matrix for classes present in test only
    present_class_names = label_encoder.classes_[present_labels]
    cm_present = confusion_matrix(y_test, y_pred, labels=present_labels)

    plt.figure(figsize=(20, 16))
    sns.heatmap(
        cm_present,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=present_class_names,
        yticklabels=present_class_names
    )
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.title("Confusion Matrix - Present Test Classes Only")
    plt.tight_layout()

    cm_png_path = os.path.join(REPORTS_DIR, "confusion_matrix_strict_split.png")
    plt.savefig(cm_png_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"✅ Confusion matrix image saved to: {cm_png_path}")

    # Most common errors
    misclassified = np.where(y_test != y_pred)[0]
    error_pairs = [(y_test[i], y_pred[i]) for i in misclassified]
    common_errors = Counter(error_pairs).most_common(15)

    print("\n6️⃣ MOST COMMON MISCLASSIFICATIONS")
    print("-" * 50)
    if not common_errors:
        print("   ✅ No misclassifications found.")
    else:
        for (true, pred), count in common_errors:
            print(
                f"   {label_encoder.classes_[true]:30s} -> "
                f"{label_encoder.classes_[pred]:30s}: {count}"
            )

    # Save JSON report
    report_json = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "split_type": "strict_order_independent_symptom_severity_split",
        "test_samples": int(len(y_test)),
        "total_classes": int(num_classes),
        "classes_present_in_test": int(len(present_labels)),
        "classes_absent_from_test": int(num_classes - len(present_labels)),
        "test_accuracy": float(test_acc),
        "precision_macro_present_test_classes": float(precision_macro_present),
        "recall_macro_present_test_classes": float(recall_macro_present),
        "f1_macro_present_test_classes": float(f1_macro_present),
        "precision_macro_all_classes": float(precision_macro_all),
        "recall_macro_all_classes": float(recall_macro_all),
        "f1_macro_all_classes": float(f1_macro_all),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "top2_accuracy": float(top2_acc),
        "top3_accuracy": float(top3_acc),
        "top5_accuracy": float(top5_acc),
        "correct_predictions": correct_predictions,
        "wrong_predictions": wrong_predictions
    }

    json_path = os.path.join(REPORTS_DIR, "saved_model_evaluation_strict_split.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2, ensure_ascii=False)

    print(f"✅ Evaluation JSON report saved to: {json_path}")

    print("\n" + "=" * 70)
    print("✅ SAVED MODEL EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    evaluate_saved_model()