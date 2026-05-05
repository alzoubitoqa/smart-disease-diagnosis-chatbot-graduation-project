import os
import json
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# Generate Charts for 4-Stage GroupKFold Validation Results
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPORTS_DIR = os.path.join(
    BASE_DIR,
    "artifacts",
    "reports",
    "group_kfold_4stage"
)

OUTPUT_DIR = os.path.join(REPORTS_DIR, "charts")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# Load summary + per-stage results
# ---------------------------------------------------------

summary_path = os.path.join(REPORTS_DIR, "group_kfold_4stage_summary.json")

stage_files = [
    os.path.join(REPORTS_DIR, "group_kfold_stage_1_results.json"),
    os.path.join(REPORTS_DIR, "group_kfold_stage_2_results.json"),
    os.path.join(REPORTS_DIR, "group_kfold_stage_3_results.json"),
    os.path.join(REPORTS_DIR, "group_kfold_stage_4_results.json"),
]

if not os.path.exists(summary_path):
    raise FileNotFoundError(
        f"Missing summary file:\n{summary_path}\n\n"
        "Make sure the GroupKFold results are saved inside:\n"
        f"{REPORTS_DIR}"
    )

for file_path in stage_files:
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Missing stage file:\n{file_path}\n\n"
            "Make sure all stage result JSON files are saved inside:\n"
            f"{REPORTS_DIR}"
        )

with open(summary_path, "r", encoding="utf-8") as f:
    summary = json.load(f)

stage_results = []

for file_path in stage_files:
    with open(file_path, "r", encoding="utf-8") as f:
        stage_results.append(json.load(f))

# ---------------------------------------------------------
# Prepare data
# ---------------------------------------------------------

stages = [f"Stage {r['stage']}" for r in stage_results]

test_accuracy = [r["test_accuracy"] * 100 for r in stage_results]
macro_f1 = [r["f1_macro"] * 100 for r in stage_results]
macro_precision = [r["precision_macro"] * 100 for r in stage_results]
macro_recall = [r["recall_macro"] * 100 for r in stage_results]
top3_accuracy = [r["top3_accuracy"] * 100 for r in stage_results]

correct_predictions = [r["correct_predictions"] for r in stage_results]
wrong_predictions = [r["wrong_predictions"] for r in stage_results]
test_classes_present = [r["test_classes_present"] for r in stage_results]

mean_accuracy = summary["mean_test_accuracy"] * 100
std_accuracy = summary["std_test_accuracy"] * 100
mean_macro_precision = summary["mean_macro_precision"] * 100
mean_macro_recall = summary["mean_macro_recall"] * 100
mean_macro_f1 = summary["mean_macro_f1"] * 100
mean_top3 = summary["mean_top3_accuracy"] * 100

# ---------------------------------------------------------
# Helper function
# ---------------------------------------------------------

def save_chart(filename):
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {path}")

# ---------------------------------------------------------
# 1. Test Accuracy Across 4 Stages
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.bar(stages, test_accuracy)
plt.ylabel("Test Accuracy (%)")
plt.xlabel("Validation Stage")
plt.title("4-Stage GroupKFold Test Accuracy")
plt.ylim(0, 105)

for i, value in enumerate(test_accuracy):
    plt.text(i, value + 0.5, f"{value:.2f}%", ha="center")

save_chart("figure_20_groupkfold_test_accuracy.png")

# ---------------------------------------------------------
# 2. Macro F1-score Across 4 Stages
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.bar(stages, macro_f1)
plt.ylabel("Macro F1-score (%)")
plt.xlabel("Validation Stage")
plt.title("4-Stage GroupKFold Macro F1-score")
plt.ylim(0, 105)

for i, value in enumerate(macro_f1):
    plt.text(i, value + 0.5, f"{value:.2f}%", ha="center")

save_chart("figure_21_groupkfold_macro_f1.png")

# ---------------------------------------------------------
# 3. Correct vs Wrong Predictions Across 4 Stages
# ---------------------------------------------------------

x = np.arange(len(stages))
width = 0.35

plt.figure(figsize=(9, 5))
plt.bar(x - width / 2, correct_predictions, width, label="Correct")
plt.bar(x + width / 2, wrong_predictions, width, label="Wrong")

plt.xticks(x, stages)
plt.ylabel("Number of Predictions")
plt.xlabel("Validation Stage")
plt.title("Correct vs Wrong Predictions Across 4 Stages")
plt.legend()

for i, value in enumerate(correct_predictions):
    plt.text(i - width / 2, value + 5, str(value), ha="center")

for i, value in enumerate(wrong_predictions):
    plt.text(i + width / 2, value + 5, str(value), ha="center")

save_chart("figure_22_groupkfold_correct_vs_wrong.png")

# ---------------------------------------------------------
# 4. Test Classes Present Across 4 Stages
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.bar(stages, test_classes_present)
plt.ylabel("Number of Classes Present")
plt.xlabel("Validation Stage")
plt.title("Disease Classes Present in Test Fold Across 4 Stages")
plt.ylim(0, 45)

for i, value in enumerate(test_classes_present):
    plt.text(i, value + 0.5, f"{value}/41", ha="center")

save_chart("figure_23_groupkfold_test_classes_present.png")

# ---------------------------------------------------------
# 5. Mean Cross-Validation Metrics
# ---------------------------------------------------------

metric_names = [
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1",
    "Top-3 Accuracy",
]

metric_values = [
    mean_accuracy,
    mean_macro_precision,
    mean_macro_recall,
    mean_macro_f1,
    mean_top3,
]

plt.figure(figsize=(10, 5))
plt.bar(metric_names, metric_values)
plt.ylabel("Percentage (%)")
plt.xlabel("Metric")
plt.title("Mean Performance Across 4-Stage GroupKFold Validation")
plt.ylim(0, 105)

for i, value in enumerate(metric_values):
    plt.text(i, value + 0.5, f"{value:.2f}%", ha="center")

plt.xticks(rotation=15)
save_chart("figure_24_groupkfold_mean_metrics.png")

# ---------------------------------------------------------
# 6. Accuracy with Mean and Std Annotation
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.bar(stages, test_accuracy)
plt.axhline(
    mean_accuracy,
    linestyle="--",
    label=f"Mean = {mean_accuracy:.2f}%"
)

plt.ylabel("Test Accuracy (%)")
plt.xlabel("Validation Stage")
plt.title(
    f"4-Stage GroupKFold Accuracy\n"
    f"Mean ± Std = {mean_accuracy:.2f}% ± {std_accuracy:.2f}%"
)
plt.ylim(0, 105)
plt.legend()

for i, value in enumerate(test_accuracy):
    plt.text(i, value + 0.5, f"{value:.2f}%", ha="center")

save_chart("figure_25_groupkfold_accuracy_mean_std.png")

print("=" * 70)
print("✅ GroupKFold charts generated successfully!")
print("=" * 70)
print(f"Saved to: {OUTPUT_DIR}")
print("\nGenerated files:")
print("1. figure_20_groupkfold_test_accuracy.png")
print("2. figure_21_groupkfold_macro_f1.png")
print("3. figure_22_groupkfold_correct_vs_wrong.png")
print("4. figure_23_groupkfold_test_classes_present.png")
print("5. figure_24_groupkfold_mean_metrics.png")
print("6. figure_25_groupkfold_accuracy_mean_std.png")