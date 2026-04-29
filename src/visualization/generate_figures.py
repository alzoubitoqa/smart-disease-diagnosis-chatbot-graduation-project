import os
import json
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model


# =============================================================================
# Custom objects for loading the trained Keras model
# =============================================================================

@tf.keras.saving.register_keras_serializable()
class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True,
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


@tf.keras.saving.register_keras_serializable()
class CosineWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, learning_rate_base, warmup_steps, total_steps):
        super().__init__()
        self.learning_rate_base = learning_rate_base
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)

        warmup = step / tf.cast(self.warmup_steps, tf.float32)

        cosine = 0.5 * (
            1.0
            + tf.cos(
                np.pi
                * (step - tf.cast(self.warmup_steps, tf.float32))
                / tf.cast(self.total_steps - self.warmup_steps, tf.float32)
            )
        )

        return self.learning_rate_base * tf.where(
            step < self.warmup_steps,
            warmup,
            cosine,
        )

    def get_config(self):
        return {
            "learning_rate_base": self.learning_rate_base,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
        }


# =============================================================================
# Paths
# =============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(BASE_DIR, "artifacts", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "artifacts", "models")
REPORTS_DIR = os.path.join(BASE_DIR, "artifacts", "reports")
FIGURES_DIR = os.path.join(BASE_DIR, "artifacts", "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)


# =============================================================================
# Global plotting setup
# =============================================================================

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["savefig.dpi"] = 300


# =============================================================================
# Helper functions
# =============================================================================

def save_figure(fig, filename: str) -> None:
    png_path = os.path.join(FIGURES_DIR, f"{filename}.png")
    pdf_path = os.path.join(FIGURES_DIR, f"{filename}.pdf")
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {png_path}")
    print(f"✅ Saved: {pdf_path}")


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_optional_csv(path: str):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# =============================================================================
# Start
# =============================================================================

print("=" * 70)
print("Generating Chapter 4 figures (Figures 4.1 to 4.7)...")
print("=" * 70)


# =============================================================================
# Load source files
# =============================================================================

dataset = pd.read_csv(os.path.join(DATA_DIR, "dataset.csv"))
severity_df = pd.read_csv(os.path.join(DATA_DIR, "Symptom-severity.csv"))

preprocess_report = load_json(os.path.join(REPORTS_DIR, "preprocessing_report.json"))
training_report = load_json(os.path.join(REPORTS_DIR, "training_complete_report.json"))
classification_report_df = load_optional_csv(os.path.join(REPORTS_DIR, "classification_report.csv"))

X_symptoms_test = np.load(os.path.join(PROCESSED_DIR, "X_symptoms_test.npy"))
X_severities_test = np.load(os.path.join(PROCESSED_DIR, "X_severities_test.npy"))
y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)

model_path = os.path.join(MODELS_DIR, "bilstm_final.keras")
custom_objects = {"Attention": Attention, "CosineWarmup": CosineWarmup}
model = load_model(model_path, custom_objects=custom_objects, compile=False)

symptom_cols = [c for c in dataset.columns if c.startswith("Symptom_")]


# =============================================================================
# Figure 4.1 - Final Performance Metrics
# =============================================================================

fig, ax = plt.subplots(figsize=(9, 5))

metrics_labels = [
    "Train Accuracy",
    "Validation Accuracy",
    "Test Accuracy",
    "Macro F1",
    "Top-3 Accuracy",
]

metrics_values = [
    training_report["training_results"]["train_accuracy"] * 100,
    training_report["training_results"]["val_accuracy"] * 100,
    training_report["test_metrics"]["test_accuracy"] * 100,
    training_report["test_metrics"]["f1_macro"] * 100,
    training_report["test_metrics"]["top3_accuracy"] * 100,
]

bars = ax.bar(metrics_labels, metrics_values, edgecolor="black")
ax.set_ylim(0, 105)
ax.set_ylabel("Percentage (%)")
ax.set_title("Figure 4.1: Final Performance Metrics")

for bar, value in zip(bars, metrics_values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.2f}%",
        ha="center",
        va="bottom",
    )

save_figure(fig, "Figure_4.1_Final_Performance_Metrics")


# =============================================================================
# Figure 4.2 - Dataset Split Distribution
# =============================================================================

fig, ax = plt.subplots(figsize=(8, 6))

split_labels = ["Train", "Validation", "Test"]
split_values = [
    preprocess_report["train_samples"],
    preprocess_report["val_samples"],
    preprocess_report["test_samples"],
]

bars = ax.bar(split_labels, split_values, edgecolor="black")
ax.set_ylabel("Number of Samples")
ax.set_title("Figure 4.2: Dataset Split Distribution")

for bar, value in zip(bars, split_values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 20,
        str(value),
        ha="center",
        va="bottom",
    )

save_figure(fig, "Figure_4.2_Dataset_Split_Distribution")


# =============================================================================
# Figure 4.3 - Top 10 Most Frequent Symptoms
# =============================================================================

all_symptoms = []
for row in dataset[symptom_cols].values:
    for s in row:
        if pd.notna(s) and str(s).strip():
            all_symptoms.append(str(s).strip())

top_10 = Counter(all_symptoms).most_common(10)

fig, ax = plt.subplots(figsize=(10, 6))
labels = [x[0].replace("_", " ") for x in top_10]
values = [x[1] for x in top_10]

bars = ax.barh(labels, values, edgecolor="black")
ax.set_xlabel("Frequency")
ax.set_ylabel("Symptom")
ax.set_title("Figure 4.3: Top 10 Most Frequent Symptoms")
ax.invert_yaxis()

for bar, value in zip(bars, values):
    ax.text(
        value + 5,
        bar.get_y() + bar.get_height() / 2,
        str(value),
        va="center",
    )

save_figure(fig, "Figure_4.3_Top_10_Symptoms")


# =============================================================================
# Figure 4.4 - Distribution of Symptom Severity Levels
# =============================================================================

severity_map = dict(zip(severity_df["Symptom"], severity_df["weight"]))

severity_values = []
for row in dataset[symptom_cols].values:
    for s in row:
        if pd.notna(s) and str(s).strip():
            severity_values.append(severity_map.get(str(s).strip(), 1))

severity_counts = Counter(severity_values)
levels = sorted(severity_counts.keys())
counts = [severity_counts[level] for level in levels]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(levels, counts, edgecolor="black")
ax.set_xlabel("Severity Level")
ax.set_ylabel("Frequency")
ax.set_title("Figure 4.4: Distribution of Symptom Severity Levels")
ax.set_xticks(levels)

for bar, value in zip(bars, counts):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 50,
        str(value),
        ha="center",
        va="bottom",
    )

save_figure(fig, "Figure_4.4_Severity_Distribution")


# =============================================================================
# Figure 4.5 - Distribution of Samples Across Disease Classes
# =============================================================================

disease_counts = dataset["Disease"].value_counts().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(14, 7))
ax.bar(range(len(disease_counts)), disease_counts.values, edgecolor="black")
ax.set_title("Figure 4.5: Distribution of Samples Across Disease Classes")
ax.set_xlabel("Disease Class")
ax.set_ylabel("Number of Samples")
ax.set_xticks(range(len(disease_counts)))
ax.set_xticklabels(
    [d.replace("_", " ") for d in disease_counts.index],
    rotation=90,
    fontsize=8,
)

save_figure(fig, "Figure_4.5_Disease_Class_Distribution")


# =============================================================================
# Figure 4.6 - Confusion Matrix
# =============================================================================

y_pred_probs = model.predict([X_symptoms_test, X_severities_test], verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test, y_pred)
labels = label_encoder.classes_

fig, ax = plt.subplots(figsize=(16, 14))
im = ax.imshow(cm, aspect="auto")
fig.colorbar(im, ax=ax)

ax.set_title("Figure 4.6: Confusion Matrix on the Test Set")
ax.set_xlabel("Predicted Class")
ax.set_ylabel("True Class")
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels([l.replace("_", " ") for l in labels], rotation=90, fontsize=7)
ax.set_yticklabels([l.replace("_", " ") for l in labels], fontsize=7)

save_figure(fig, "Figure_4.6_Confusion_Matrix")


# =============================================================================
# Figure 4.7 - Class-wise Precision, Recall, and F1-score Heatmap
# =============================================================================

if classification_report_df is not None:
    report_df = classification_report_df.copy()

    first_col_original = report_df.columns[0]

    report_df = report_df[
        ~report_df[first_col_original].astype(str).isin(["accuracy", "macro avg", "weighted avg"])
    ]

    report_df = report_df.rename(columns={first_col_original: "class_name"})

    metric_cols = [c for c in report_df.columns if c in ["precision", "recall", "f1-score"]]

    if metric_cols:
        heatmap_data = report_df[metric_cols].astype(float).values
        class_names = report_df["class_name"].astype(str).tolist()

        fig, ax = plt.subplots(figsize=(8, 12))
        im = ax.imshow(heatmap_data, aspect="auto", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax)

        ax.set_title("Figure 4.7: Class-wise Precision, Recall, and F1-score Heatmap")
        ax.set_xticks(range(len(metric_cols)))
        ax.set_xticklabels(metric_cols)
        ax.set_yticks(range(len(class_names)))
        ax.set_yticklabels([c.replace("_", " ") for c in class_names], fontsize=7)

        save_figure(fig, "Figure_4.7_Classwise_Metrics_Heatmap")


print("=" * 70)
print("All figures generated successfully.")
print(f"Saved in: {FIGURES_DIR}")
print("=" * 70)