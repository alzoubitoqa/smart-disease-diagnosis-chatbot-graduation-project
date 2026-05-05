import os
import json
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
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

        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)

        warmup = step / warmup_steps

        denominator = tf.maximum(total_steps - warmup_steps, 1.0)

        cosine = 0.5 * (
            1.0
            + tf.cos(
                np.pi
                * (step - warmup_steps)
                / denominator
            )
        )

        return self.learning_rate_base * tf.where(
            step < warmup_steps,
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


def load_optional_json(path: str):
    if os.path.exists(path):
        return load_json(path)
    return None


def load_optional_csv(path: str):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def normalize_name(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_")


def get_nested_value(dictionary, keys, default=None):
    current = dictionary

    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]

    return current


# =============================================================================
# Start
# =============================================================================

print("=" * 70)
print("Generating Chapter 4 figures (Figures 4.13 to 4.19)...")
print("=" * 70)


# =============================================================================
# Load source files
# =============================================================================

dataset_path = os.path.join(DATA_DIR, "dataset.csv")
severity_path = os.path.join(DATA_DIR, "Symptom-severity.csv")
preprocess_report_path = os.path.join(REPORTS_DIR, "preprocessing_report.json")
training_report_path = os.path.join(REPORTS_DIR, "training_complete_report.json")
strict_eval_report_path = os.path.join(REPORTS_DIR, "saved_model_evaluation_strict_split.json")

classification_report_paths = [
    os.path.join(REPORTS_DIR, "classification_report_strict_split.csv"),
    os.path.join(REPORTS_DIR, "comprehensive_classification_report.csv"),
    os.path.join(REPORTS_DIR, "classification_report.csv"),
]

required_paths = [
    dataset_path,
    severity_path,
    preprocess_report_path,
    os.path.join(PROCESSED_DIR, "X_symptoms_test.npy"),
    os.path.join(PROCESSED_DIR, "X_severities_test.npy"),
    os.path.join(PROCESSED_DIR, "y_test.npy"),
    os.path.join(MODELS_DIR, "label_encoder.pkl"),
]

missing = [path for path in required_paths if not os.path.exists(path)]

if missing:
    print("\n❌ Missing required files:")
    for path in missing:
        print(f"   - {path}")
    raise FileNotFoundError("Please run preprocessing/training/evaluation before generating figures.")

dataset = pd.read_csv(dataset_path)
severity_df = pd.read_csv(severity_path)

preprocess_report = load_json(preprocess_report_path)
training_report = load_optional_json(training_report_path)
strict_eval_report = load_optional_json(strict_eval_report_path)

classification_report_df = None
for path in classification_report_paths:
    classification_report_df = load_optional_csv(path)
    if classification_report_df is not None:
        print(f"✅ Classification report loaded from: {path}")
        break

X_symptoms_test = np.load(os.path.join(PROCESSED_DIR, "X_symptoms_test.npy"))
X_severities_test = np.load(os.path.join(PROCESSED_DIR, "X_severities_test.npy"))
y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)

best_model_path = os.path.join(MODELS_DIR, "bilstm_best.keras")
final_model_path = os.path.join(MODELS_DIR, "bilstm_final.keras")

if os.path.exists(best_model_path):
    model_path = best_model_path
elif os.path.exists(final_model_path):
    model_path = final_model_path
else:
    raise FileNotFoundError(
        f"No trained model found at:\n{best_model_path}\nor\n{final_model_path}"
    )

custom_objects = {
    "Attention": Attention,
    "CosineWarmup": CosineWarmup,
}

model = load_model(
    model_path,
    custom_objects=custom_objects,
    compile=False
)

print(f"✅ Model loaded from: {model_path}")

symptom_cols = [c for c in dataset.columns if c.startswith("Symptom_")]
num_classes = len(label_encoder.classes_)
all_labels = np.arange(num_classes)


# =============================================================================
# Prepare prediction outputs once
# =============================================================================

print("\n🔮 Generating predictions for test set...")
y_pred_probs = model.predict([X_symptoms_test, X_severities_test], verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

if strict_eval_report is None:
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=all_labels,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )

    report_df_generated = pd.DataFrame(report_dict).transpose()
    generated_report_path = os.path.join(REPORTS_DIR, "classification_report_strict_split.csv")
    report_df_generated.to_csv(generated_report_path)
    classification_report_df = pd.read_csv(generated_report_path)

    strict_eval_report = {
        "test_accuracy": float(np.mean(y_test == y_pred)),
        "f1_macro_present_test_classes": float(report_dict["macro avg"]["f1-score"]),
        "f1_weighted": float(report_dict["weighted avg"]["f1-score"]),
        "top3_accuracy": float(
            np.mean([
                y_test[i] in np.argsort(y_pred_probs[i])[-3:]
                for i in range(len(y_test))
            ])
        ),
    }


# =============================================================================
# Metric selection
# =============================================================================

train_accuracy = None
val_accuracy = None

if training_report is not None:
    train_accuracy = get_nested_value(
        training_report,
        ["training_results", "train_accuracy"],
        None,
    )

    val_accuracy = get_nested_value(
        training_report,
        ["training_results", "val_accuracy"],
        None,
    )

test_accuracy = strict_eval_report.get(
    "test_accuracy",
    get_nested_value(training_report or {}, ["test_metrics", "test_accuracy"], 0),
)

macro_f1_present = strict_eval_report.get(
    "f1_macro_present_test_classes",
    strict_eval_report.get(
        "f1_macro_present_classes",
        get_nested_value(training_report or {}, ["test_metrics", "f1_macro_present_classes"], None),
    ),
)

if macro_f1_present is None:
    macro_f1_present = get_nested_value(training_report or {}, ["test_metrics", "f1_macro"], 0)

top3_accuracy = strict_eval_report.get(
    "top3_accuracy",
    get_nested_value(training_report or {}, ["test_metrics", "top3_accuracy"], 0),
)

if train_accuracy is None:
    train_accuracy = get_nested_value(training_report or {}, ["training_results", "train_accuracy"], 0)

if val_accuracy is None:
    val_accuracy = get_nested_value(training_report or {}, ["training_results", "val_accuracy"], 0)


# =============================================================================
# Figure 4.13 - Final Performance Metrics
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 5))

metrics_labels = [
    "Train Accuracy",
    "Validation Accuracy",
    "Test Accuracy",
    "Macro F1\n(Present Classes)",
    "Top-3 Accuracy",
]

metrics_values = [
    train_accuracy * 100,
    val_accuracy * 100,
    test_accuracy * 100,
    macro_f1_present * 100,
    top3_accuracy * 100,
]

bars = ax.bar(metrics_labels, metrics_values, edgecolor="black")
ax.set_ylim(0, 105)
ax.set_ylabel("Percentage (%)")
ax.set_title("Figure 4.13: Final Performance Metrics")

for bar, value in zip(bars, metrics_values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.2f}%",
        ha="center",
        va="bottom",
    )

save_figure(fig, "Figure_4.13_Final_Performance_Metrics")


# =============================================================================
# Figure 4.14 - Dataset Split Distribution
# =============================================================================

fig, ax = plt.subplots(figsize=(8, 6))

split_labels = ["Train", "Validation", "Test"]
split_values = [
    int(preprocess_report["train_samples"]),
    int(preprocess_report["val_samples"]),
    int(preprocess_report["test_samples"]),
]

bars = ax.bar(split_labels, split_values, edgecolor="black")
ax.set_ylabel("Number of Samples")
ax.set_title("Figure 4.14: Dataset Split Distribution")

offset = max(split_values) * 0.015

for bar, value in zip(bars, split_values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + offset,
        str(value),
        ha="center",
        va="bottom",
    )

save_figure(fig, "Figure_4.14_Dataset_Split_Distribution")


# =============================================================================
# Figure 4.15 - Top 10 Most Frequent Symptoms
# =============================================================================

all_symptoms = []

for row in dataset[symptom_cols].values:
    for symptom in row:
        if pd.notna(symptom) and str(symptom).strip():
            all_symptoms.append(normalize_name(symptom))

top_10 = Counter(all_symptoms).most_common(10)

fig, ax = plt.subplots(figsize=(10, 6))

labels = [item[0].replace("_", " ") for item in top_10]
values = [item[1] for item in top_10]

bars = ax.barh(labels, values, edgecolor="black")
ax.set_xlabel("Frequency")
ax.set_ylabel("Symptom")
ax.set_title("Figure 4.15: Top 10 Most Frequent Symptoms")
ax.invert_yaxis()

offset = max(values) * 0.01 if values else 1

for bar, value in zip(bars, values):
    ax.text(
        value + offset,
        bar.get_y() + bar.get_height() / 2,
        str(value),
        va="center",
    )

save_figure(fig, "Figure_4.15_Top_10_Symptoms")


# =============================================================================
# Figure 4.16 - Distribution of Symptom Severity Levels
# =============================================================================

severity_df["Symptom"] = severity_df["Symptom"].apply(normalize_name)
severity_map = dict(zip(severity_df["Symptom"], severity_df["weight"]))

severity_values = []

for row in dataset[symptom_cols].values:
    for symptom in row:
        if pd.notna(symptom) and str(symptom).strip():
            normalized_symptom = normalize_name(symptom)
            severity_values.append(int(severity_map.get(normalized_symptom, 1)))

severity_counts = Counter(severity_values)
levels = sorted(severity_counts.keys())
counts = [severity_counts[level] for level in levels]

fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.bar(levels, counts, edgecolor="black")
ax.set_xlabel("Severity Level")
ax.set_ylabel("Frequency")
ax.set_title("Figure 4.16: Distribution of Symptom Severity Levels")
ax.set_xticks(levels)

offset = max(counts) * 0.015 if counts else 1

for bar, value in zip(bars, counts):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + offset,
        str(value),
        ha="center",
        va="bottom",
    )

save_figure(fig, "Figure_4.16_Severity_Distribution")


# =============================================================================
# Figure 4.17 - Distribution of Samples Across Disease Classes
# =============================================================================

dataset["Disease"] = dataset["Disease"].apply(normalize_name)
disease_counts = dataset["Disease"].value_counts().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(14, 7))

ax.bar(range(len(disease_counts)), disease_counts.values, edgecolor="black")
ax.set_title("Figure 4.17: Distribution of Samples Across Disease Classes")
ax.set_xlabel("Disease Class")
ax.set_ylabel("Number of Samples")
ax.set_xticks(range(len(disease_counts)))
ax.set_xticklabels(
    [disease.replace("_", " ") for disease in disease_counts.index],
    rotation=90,
    fontsize=8,
)

save_figure(fig, "Figure_4.17_Disease_Class_Distribution")


# =============================================================================
# Figure 4.18 - Confusion Matrix
# =============================================================================

cm_full = confusion_matrix(
    y_test,
    y_pred,
    labels=all_labels,
)

present_labels = np.unique(y_test)
present_label_names = label_encoder.classes_[present_labels]

cm_present = confusion_matrix(
    y_test,
    y_pred,
    labels=present_labels,
)

fig, ax = plt.subplots(figsize=(16, 14))

im = ax.imshow(cm_present, aspect="auto")
fig.colorbar(im, ax=ax)

ax.set_title("Figure 4.18: Confusion Matrix on the Test Set")
ax.set_xlabel("Predicted Class")
ax.set_ylabel("True Class")
ax.set_xticks(range(len(present_label_names)))
ax.set_yticks(range(len(present_label_names)))
ax.set_xticklabels(
    [label.replace("_", " ") for label in present_label_names],
    rotation=90,
    fontsize=7,
)
ax.set_yticklabels(
    [label.replace("_", " ") for label in present_label_names],
    fontsize=7,
)

for i in range(cm_present.shape[0]):
    for j in range(cm_present.shape[1]):
        value = cm_present[i, j]
        if value > 0:
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                fontsize=6,
            )

cm_csv_path = os.path.join(FIGURES_DIR, "Figure_4.18_Confusion_Matrix_Full_41_Classes.csv")
pd.DataFrame(
    cm_full,
    index=label_encoder.classes_,
    columns=label_encoder.classes_,
).to_csv(cm_csv_path)

print(f"✅ Saved: {cm_csv_path}")

save_figure(fig, "Figure_4.18_Confusion_Matrix")


# =============================================================================
# Figure 4.19 - Class-wise Precision, Recall, and F1-score Heatmap
# =============================================================================

if classification_report_df is None:
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=all_labels,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    classification_report_df = pd.DataFrame(report_dict).transpose().reset_index()

report_df = classification_report_df.copy()

if "Unnamed: 0" in report_df.columns:
    report_df = report_df.rename(columns={"Unnamed: 0": "class_name"})
elif "index" in report_df.columns:
    report_df = report_df.rename(columns={"index": "class_name"})
else:
    first_col = report_df.columns[0]
    report_df = report_df.rename(columns={first_col: "class_name"})

excluded_rows = [
    "accuracy",
    "macro avg",
    "weighted avg",
    "micro avg",
]

report_df = report_df[
    ~report_df["class_name"].astype(str).isin(excluded_rows)
].copy()

metric_cols = [col for col in ["precision", "recall", "f1-score"] if col in report_df.columns]

if metric_cols:
    if "support" in report_df.columns:
        report_df["support"] = pd.to_numeric(report_df["support"], errors="coerce").fillna(0)
        report_df = report_df[report_df["support"] > 0].copy()

    heatmap_data = report_df[metric_cols].astype(float).values
    class_names = report_df["class_name"].astype(str).tolist()

    fig, ax = plt.subplots(figsize=(8, 12))

    im = ax.imshow(
        heatmap_data,
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    fig.colorbar(im, ax=ax)

    ax.set_title("Figure 4.19: Class-wise Precision, Recall, and F1-score Heatmap")
    ax.set_xticks(range(len(metric_cols)))
    ax.set_xticklabels(metric_cols)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(
        [class_name.replace("_", " ") for class_name in class_names],
        fontsize=7,
    )

    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            ax.text(
                j,
                i,
                f"{heatmap_data[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=6,
            )

    save_figure(fig, "Figure_4.19_Classwise_Metrics_Heatmap")
else:
    print("⚠️ Could not generate Figure 4.19 because metric columns were not found.")


print("=" * 70)
print("All Chapter 4 figures generated successfully.")
print(f"Saved in: {FIGURES_DIR}")
print("=" * 70)