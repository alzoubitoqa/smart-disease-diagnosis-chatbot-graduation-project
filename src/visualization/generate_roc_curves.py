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
            1 + tf.cos(np.pi * (step - self.warmup_steps) / (self.total_steps - self.warmup_steps))
        )
        return self.learning_rate_base * tf.where(step < self.warmup_steps, warmup, cosine)

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
PROCESSED_DIR = os.path.join(BASE_DIR, "artifacts", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "artifacts", "models")
FIGURES_DIR = os.path.join(BASE_DIR, "artifacts", "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================================
# Load Data and Model
# ============================================================================

print("=" * 70)
print("Generating ROC and AUC figures...")
print("=" * 70)

X_symptoms_test = np.load(os.path.join(PROCESSED_DIR, "X_symptoms_test.npy"))
X_severities_test = np.load(os.path.join(PROCESSED_DIR, "X_severities_test.npy"))
y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)

num_classes = len(label_encoder.classes_)

model_path = os.path.join(MODELS_DIR, "bilstm_final.keras")
custom_objects = {"Attention": Attention, "CosineWarmup": CosineWarmup}
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)

# ============================================================================
# Predictions
# ============================================================================

y_pred_probs = model.predict([X_symptoms_test, X_severities_test], verbose=0)
y_test_bin = label_binarize(y_test, classes=range(num_classes))

# ============================================================================
# ROC / AUC Calculation
# ============================================================================

fpr = {}
tpr = {}
roc_auc = {}

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Micro-average
fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_pred_probs.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)

# Macro-average
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(num_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= num_classes
roc_auc_macro = auc(all_fpr, mean_tpr)

auc_list = [roc_auc[i] for i in range(num_classes)]

print(f"Micro-average AUC: {roc_auc_micro:.4f}")
print(f"Macro-average AUC: {roc_auc_macro:.4f}")
print(f"Mean AUC: {np.mean(auc_list):.4f}")
print(f"Min AUC: {np.min(auc_list):.4f}")
print(f"Max AUC: {np.max(auc_list):.4f}")

# ============================================================================
# Figure 4.7: Micro vs Macro ROC
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(fpr_micro, tpr_micro, linewidth=3, label=f"Micro-average ROC (AUC = {roc_auc_micro:.4f})")
ax.plot(all_fpr, mean_tpr, linewidth=3, linestyle="--", label=f"Macro-average ROC (AUC = {roc_auc_macro:.4f})")
ax.plot([0, 1], [0, 1], linestyle=":", linewidth=1.5, color="gray", label="Random classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Figure 4.7: Micro-average and Macro-average ROC Curves")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "Figure_4.7_Micro_Macro_ROC.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(FIGURES_DIR, "Figure_4.7_Micro_Macro_ROC.pdf"), bbox_inches="tight")
plt.close()

# ============================================================================
# Figure 4.8: ROC Curves for Top 10 Most Common Diseases
# ============================================================================

disease_counts = Counter(y_test)
top_10_indices = [idx for idx, _ in disease_counts.most_common(10)]
top_10_names = [label_encoder.classes_[i].replace("_", " ") for i in top_10_indices]

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

ax.plot([0, 1], [0, 1], linestyle=":", linewidth=1.5, color="gray", label="Random classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Figure 4.8: ROC Curves for the Top 10 Most Common Diseases")
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "Figure_4.8_Top10_ROC.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(FIGURES_DIR, "Figure_4.8_Top10_ROC.pdf"), bbox_inches="tight")
plt.close()

# ============================================================================
# Figure 4.9: AUC Summary for Top 10 Most Common Diseases
# ============================================================================

top_10_auc = [roc_auc[i] for i in top_10_indices]

fig, ax = plt.subplots(figsize=(11, 7))
bars = ax.barh(range(len(top_10_indices)), top_10_auc, edgecolor="black")
ax.set_yticks(range(len(top_10_indices)))
ax.set_yticklabels(top_10_names)
ax.set_xlim(0, 1.05)
ax.set_xlabel("AUC Score")
ax.set_ylabel("Disease")
ax.set_title("Figure 4.9: AUC Scores for the Top 10 Most Common Diseases")
ax.invert_yaxis()
ax.axvline(roc_auc_macro, linestyle="--", linewidth=2, color="red", label=f"Macro-average AUC = {roc_auc_macro:.4f}")
ax.legend()

for bar, value in zip(bars, top_10_auc):
    ax.text(value + 0.005, bar.get_y() + bar.get_height()/2, f"{value:.4f}", va="center")

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "Figure_4.9_Top10_AUC.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(FIGURES_DIR, "Figure_4.9_Top10_AUC.pdf"), bbox_inches="tight")
plt.close()

print("=" * 70)
print("ROC and AUC figures generated successfully.")
print(f"Saved in: {FIGURES_DIR}")
print("=" * 70)