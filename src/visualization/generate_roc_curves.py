import os
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from collections import Counter
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ============================================================================
# Custom Layers
# ============================================================================

class Attention(tf.keras.layers.Layer):
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
            cosine
        )

    def get_config(self):
        return {
            "learning_rate_base": self.learning_rate_base,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps
        }


# ============================================================================
# Paths
# ============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, "artifacts", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "artifacts", "models")
FIGURES_DIR = os.path.join(BASE_DIR, "artifacts", "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================================
# Helper Function
# ============================================================================

def save_figure(fig, filename):
    png_path = os.path.join(FIGURES_DIR, f"{filename}.png")
    pdf_path = os.path.join(FIGURES_DIR, f"{filename}.pdf")

    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ Saved: {png_path}")
    print(f"✅ Saved: {pdf_path}")


# ============================================================================
# Load Data and Model
# ============================================================================

print("=" * 70)
print("Generating ROC and AUC figures: Figures 4.20, 4.21, and 4.22")
print("=" * 70)

X_symptoms_test = np.load(os.path.join(PROCESSED_DIR, "X_symptoms_test.npy"))
X_severities_test = np.load(os.path.join(PROCESSED_DIR, "X_severities_test.npy"))
y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)

num_classes = len(label_encoder.classes_)
all_labels = np.arange(num_classes)

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
    "CosineWarmup": CosineWarmup
}

model = tf.keras.models.load_model(
    model_path,
    custom_objects=custom_objects,
    compile=False
)

print(f"✅ Model loaded from: {model_path}")


# ============================================================================
# Predictions
# ============================================================================

print("\n🔮 Generating predictions...")
y_pred_probs = model.predict([X_symptoms_test, X_severities_test], verbose=0)

# Binarize using all 41 classes to match model output
y_test_bin = label_binarize(y_test, classes=all_labels)

present_classes = np.unique(y_test)

print(f"   - Total classes in dataset: {num_classes}")
print(f"   - Classes present in test set: {len(present_classes)}")


# ============================================================================
# ROC / AUC Calculation
# ============================================================================

fpr = {}
tpr = {}
roc_auc = {}

# Calculate ROC only for classes present in the test set.
# This avoids undefined ROC curves for classes with zero support.
for class_idx in present_classes:
    fpr[class_idx], tpr[class_idx], _ = roc_curve(
        y_test_bin[:, class_idx],
        y_pred_probs[:, class_idx]
    )
    roc_auc[class_idx] = auc(fpr[class_idx], tpr[class_idx])

# Micro-average over all class columns
fpr_micro, tpr_micro, _ = roc_curve(
    y_test_bin.ravel(),
    y_pred_probs.ravel()
)
roc_auc_micro = auc(fpr_micro, tpr_micro)

# Macro-average over classes present in test set only
all_fpr = np.unique(
    np.concatenate([fpr[class_idx] for class_idx in present_classes])
)

mean_tpr = np.zeros_like(all_fpr)

for class_idx in present_classes:
    mean_tpr += np.interp(all_fpr, fpr[class_idx], tpr[class_idx])

mean_tpr /= len(present_classes)
roc_auc_macro = auc(all_fpr, mean_tpr)

auc_list = [roc_auc[class_idx] for class_idx in present_classes]

print(f"\nMicro-average AUC: {roc_auc_micro:.4f}")
print(f"Macro-average AUC for present test classes: {roc_auc_macro:.4f}")
print(f"Mean AUC for present test classes: {np.mean(auc_list):.4f}")
print(f"Min AUC for present test classes: {np.min(auc_list):.4f}")
print(f"Max AUC for present test classes: {np.max(auc_list):.4f}")


# ============================================================================
# Figure 4.20: Micro vs Macro ROC
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 8))

ax.plot(
    fpr_micro,
    tpr_micro,
    linewidth=3,
    label=f"Micro-average ROC (AUC = {roc_auc_micro:.4f})"
)

ax.plot(
    all_fpr,
    mean_tpr,
    linewidth=3,
    linestyle="--",
    label=f"Macro-average ROC - Present Classes (AUC = {roc_auc_macro:.4f})"
)

ax.plot(
    [0, 1],
    [0, 1],
    linestyle=":",
    linewidth=1.5,
    color="gray",
    label="Random classifier"
)

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Figure 4.20: Micro-average and Macro-average ROC Curves")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)

save_figure(fig, "Figure_4.20_Micro_Macro_ROC")


# ============================================================================
# Figure 4.21: ROC Curves for Top 10 Most Common Diseases
# ============================================================================

disease_counts = Counter(y_test)
top_10_indices = [idx for idx, _ in disease_counts.most_common(10)]
top_10_names = [
    label_encoder.classes_[idx].replace("_", " ")
    for idx in top_10_indices
]

fig, ax = plt.subplots(figsize=(12, 9))

colors = plt.cm.tab10(np.linspace(0, 1, len(top_10_indices)))

for color, class_idx, class_name in zip(colors, top_10_indices, top_10_names):
    ax.plot(
        fpr[class_idx],
        tpr[class_idx],
        linewidth=2.2,
        color=color,
        label=f"{class_name} (AUC = {roc_auc[class_idx]:.4f})"
    )

ax.plot(
    [0, 1],
    [0, 1],
    linestyle=":",
    linewidth=1.5,
    color="gray",
    label="Random classifier"
)

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Figure 4.21: ROC Curves for the Top 10 Most Common Diseases")
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, alpha=0.3)

save_figure(fig, "Figure_4.21_Top10_ROC")


# ============================================================================
# Figure 4.22: AUC Summary for Top 10 Most Common Diseases
# ============================================================================

top_10_auc = [roc_auc[idx] for idx in top_10_indices]

fig, ax = plt.subplots(figsize=(11, 7))

bars = ax.barh(
    range(len(top_10_indices)),
    top_10_auc,
    edgecolor="black"
)

ax.set_yticks(range(len(top_10_indices)))
ax.set_yticklabels(top_10_names)
ax.set_xlim(0, 1.05)
ax.set_xlabel("AUC Score")
ax.set_ylabel("Disease")
ax.set_title("Figure 4.22: AUC Scores for the Top 10 Most Common Diseases")
ax.invert_yaxis()

ax.axvline(
    roc_auc_macro,
    linestyle="--",
    linewidth=2,
    color="red",
    label=f"Macro-average AUC = {roc_auc_macro:.4f}"
)

ax.legend()

for bar, value in zip(bars, top_10_auc):
    ax.text(
        value + 0.005,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.4f}",
        va="center"
    )

save_figure(fig, "Figure_4.22_Top10_AUC")


print("=" * 70)
print("ROC and AUC figures generated successfully.")
print(f"Saved in: {FIGURES_DIR}")
print("=" * 70)