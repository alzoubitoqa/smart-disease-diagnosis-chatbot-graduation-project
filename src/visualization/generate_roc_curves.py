"""
File: src/visualization/generate_roc_curves.py
Purpose: Generate ROC curves for multi-class classification (41 diseases)
"""

import os
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ============================================================================
# Custom Layers (for loading model)
# ============================================================================

class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(input_shape[1], 1),
                                 initializer='zeros', trainable=True)
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
        cosine = 0.5 * (1 + tf.cos(np.pi * (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)))
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
PROCESSED_DIR = os.path.join(BASE_DIR, "artifacts", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "artifacts", "models")
FIGURES_DIR = os.path.join(BASE_DIR, "artifacts", "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================================
# Load Data and Model
# ============================================================================

print("="*70)
print("📊 GENERATING ROC CURVES FOR 41 DISEASES")
print("="*70)

print("\n📂 Loading preprocessed data...")
X_symptoms_test = np.load(os.path.join(PROCESSED_DIR, "X_symptoms_test.npy"))
X_severities_test = np.load(os.path.join(PROCESSED_DIR, "X_severities_test.npy"))
y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), 'rb') as f:
    label_encoder = pickle.load(f)

num_classes = len(label_encoder.classes_)
print(f"   Test samples: {len(X_symptoms_test)}")
print(f"   Number of diseases: {num_classes}")

print("\n🤖 Loading trained model...")
model_path = os.path.join(MODELS_DIR, "bilstm_final.keras")
custom_objects = {'Attention': Attention, 'CosineWarmup': CosineWarmup}
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
print("✅ Model loaded")

# ============================================================================
# Get Predictions
# ============================================================================

print("\n📊 Getting predictions...")
y_pred_probs = model.predict([X_symptoms_test, X_severities_test], verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# Convert labels to one-hot for ROC
y_test_bin = label_binarize(y_test, classes=range(num_classes))

print(f"   Prediction shape: {y_pred_probs.shape}")
print(f"   Test labels shape: {y_test_bin.shape}")

# ============================================================================
# Calculate ROC and AUC for each class
# ============================================================================

print("\n📊 Calculating ROC curves and AUC scores...")

fpr = {}
tpr = {}
roc_auc = {}

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Calculate micro-average (overall performance)
fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_pred_probs.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)

# Calculate macro-average (average of all classes)
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(num_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= num_classes
roc_auc_macro = auc(all_fpr, mean_tpr)

print(f"   Micro-average AUC: {roc_auc_micro:.4f}")
print(f"   Macro-average AUC: {roc_auc_macro:.4f}")

# Print AUC for each class
print("\n📊 AUC per disease:")
auc_list = []
for i in range(num_classes):
    disease_name = label_encoder.classes_[i].replace("_", " ")
    print(f"   {i+1:2d}. {disease_name[:35]:35} AUC = {roc_auc[i]:.4f}")
    auc_list.append(roc_auc[i])

print(f"\n   Min AUC: {min(auc_list):.4f}")
print(f"   Max AUC: {max(auc_list):.4f}")
print(f"   Mean AUC: {np.mean(auc_list):.4f}")
print(f"   Std AUC: {np.std(auc_list):.4f}")

# ============================================================================
# Figure 1: ROC Curves (All 41 classes - zoomed view)
# ============================================================================

print("\n📊 Generating Figure: ROC Curves (All 41 Diseases)...")

fig, ax = plt.subplots(figsize=(12, 10))

# Plot each class with low alpha
colors = plt.cm.viridis(np.linspace(0, 1, num_classes))
for i in range(num_classes):
    ax.plot(fpr[i], tpr[i], color=colors[i], lw=1, alpha=0.4, 
            label=f'{label_encoder.classes_[i][:20]}' if i < 10 else "")

# Plot micro and macro averages
ax.plot(fpr_micro, tpr_micro, color='darkred', lw=3, linestyle='-', 
        label=f'Micro-average (AUC = {roc_auc_micro:.4f})')
ax.plot(all_fpr, mean_tpr, color='navy', lw=3, linestyle='--', 
        label=f'Macro-average (AUC = {roc_auc_macro:.4f})')

# Diagonal line (random classifier)
ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle=':', label='Random (AUC = 0.5)')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
ax.set_title('Figure 4.X: ROC Curves for 41 Disease Classes', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "Figure_ROC_Curves_All_41.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIGURES_DIR, "Figure_ROC_Curves_All_41.pdf"), bbox_inches='tight')
plt.close()
print("   ✅ Saved: Figure_ROC_Curves_All_41.png/pdf")

# ============================================================================
# Figure 2: ROC Curves (Top 10 Most Common Diseases - cleaner view)
# ============================================================================

print("\n📊 Generating Figure: ROC Curves (Top 10 Most Common Diseases)...")

# Get top 10 most frequent diseases in test set
from collections import Counter
disease_counts = Counter(y_test)
top_10_indices = [idx for idx, _ in disease_counts.most_common(10)]
top_10_names = [label_encoder.classes_[i].replace("_", " ") for i in top_10_indices]

fig, ax = plt.subplots(figsize=(12, 10))

# Plot top 10 classes with distinct colors
top_colors = plt.cm.tab10(np.linspace(0, 1, 10))
for idx, (class_idx, color) in enumerate(zip(top_10_indices, top_colors)):
    ax.plot(fpr[class_idx], tpr[class_idx], color=color, lw=2.5, alpha=0.8,
            label=f'{top_10_names[idx]} (AUC = {roc_auc[class_idx]:.4f})')

# Diagonal line
ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle=':', label='Random (AUC = 0.5)')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
ax.set_title('Figure 4.X: ROC Curves (Top 10 Most Common Diseases)', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "Figure_ROC_Curves_Top_10.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIGURES_DIR, "Figure_ROC_Curves_Top_10.pdf"), bbox_inches='tight')
plt.close()
print("   ✅ Saved: Figure_ROC_Curves_Top_10.png/pdf")

# ============================================================================
# Figure 3: AUC Bar Chart
# ============================================================================

print("\n📊 Generating Figure: AUC Bar Chart...")

fig, ax = plt.subplots(figsize=(14, 8))
disease_names_short = [label_encoder.classes_[i].replace("_", " ")[:30] for i in range(num_classes)]
colors_bar = plt.cm.RdYlGn(np.linspace(0.2, 0.8, num_classes))

bars = ax.barh(range(num_classes), auc_list, color=colors_bar, edgecolor='black', linewidth=0.5)
ax.set_xlabel('AUC Score', fontsize=12)
ax.set_ylabel('Disease', fontsize=12)
ax.set_title('Figure 4.X: AUC Score per Disease', fontsize=14, fontweight='bold')
ax.set_yticks(range(num_classes))
ax.set_yticklabels(disease_names_short, fontsize=8)
ax.set_xlim(0, 1.05)
ax.axvline(x=roc_auc_macro, color='red', linestyle='--', linewidth=2, label=f'Macro-average AUC = {roc_auc_macro:.4f}')
ax.legend()
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "Figure_AUC_Bar_Chart.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIGURES_DIR, "Figure_AUC_Bar_Chart.pdf"), bbox_inches='tight')
plt.close()
print("   ✅ Saved: Figure_AUC_Bar_Chart.png/pdf")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("✅ ROC Analysis Complete!")
print("="*70)
print(f"\n📊 Final AUC Summary:")
print(f"   Micro-average AUC: {roc_auc_micro:.4f}")
print(f"   Macro-average AUC: {roc_auc_macro:.4f}")
print(f"   Mean AUC across all diseases: {np.mean(auc_list):.4f}")
print(f"   Standard deviation: {np.std(auc_list):.4f}")
print(f"\n📁 Figures saved in: {FIGURES_DIR}")
print("="*70)


import os
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ============================================================================
# Custom Layers (for loading model)
# ============================================================================

class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(input_shape[1], 1),
                                 initializer='zeros', trainable=True)
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
        cosine = 0.5 * (1 + tf.cos(np.pi * (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)))
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
PROCESSED_DIR = os.path.join(BASE_DIR, "artifacts", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "artifacts", "models")
FIGURES_DIR = os.path.join(BASE_DIR, "artifacts", "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================================
# Load Data and Model
# ============================================================================

print("="*70)
print("📊 GENERATING ROC CURVES FOR 41 DISEASES")
print("="*70)

print("\n📂 Loading preprocessed data...")
X_symptoms_test = np.load(os.path.join(PROCESSED_DIR, "X_symptoms_test.npy"))
X_severities_test = np.load(os.path.join(PROCESSED_DIR, "X_severities_test.npy"))
y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), 'rb') as f:
    label_encoder = pickle.load(f)

num_classes = len(label_encoder.classes_)
print(f"   Test samples: {len(X_symptoms_test)}")
print(f"   Number of diseases: {num_classes}")

print("\n🤖 Loading trained model...")
model_path = os.path.join(MODELS_DIR, "bilstm_final.keras")
custom_objects = {'Attention': Attention, 'CosineWarmup': CosineWarmup}
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
print("✅ Model loaded")

# ============================================================================
# Get Predictions
# ============================================================================

print("\n📊 Getting predictions...")
y_pred_probs = model.predict([X_symptoms_test, X_severities_test], verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# Convert labels to one-hot for ROC
y_test_bin = label_binarize(y_test, classes=range(num_classes))

print(f"   Prediction shape: {y_pred_probs.shape}")
print(f"   Test labels shape: {y_test_bin.shape}")

# ============================================================================
# Calculate ROC and AUC for each class
# ============================================================================

print("\n📊 Calculating ROC curves and AUC scores...")

fpr = {}
tpr = {}
roc_auc = {}

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Calculate micro-average (overall performance)
fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_pred_probs.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)

# Calculate macro-average (average of all classes)
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(num_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= num_classes
roc_auc_macro = auc(all_fpr, mean_tpr)

print(f"   Micro-average AUC: {roc_auc_micro:.4f}")
print(f"   Macro-average AUC: {roc_auc_macro:.4f}")

# Print AUC for each class
print("\n📊 AUC per disease:")
auc_list = []
for i in range(num_classes):
    disease_name = label_encoder.classes_[i].replace("_", " ")
    print(f"   {i+1:2d}. {disease_name[:35]:35} AUC = {roc_auc[i]:.4f}")
    auc_list.append(roc_auc[i])

print(f"\n   Min AUC: {min(auc_list):.4f}")
print(f"   Max AUC: {max(auc_list):.4f}")
print(f"   Mean AUC: {np.mean(auc_list):.4f}")
print(f"   Std AUC: {np.std(auc_list):.4f}")

# ============================================================================
# Figure 1: ROC Curves (All 41 classes - zoomed view)
# ============================================================================

print("\n📊 Generating Figure: ROC Curves (All 41 Diseases)...")

fig, ax = plt.subplots(figsize=(12, 10))

# Plot each class with low alpha
colors = plt.cm.viridis(np.linspace(0, 1, num_classes))
for i in range(num_classes):
    ax.plot(fpr[i], tpr[i], color=colors[i], lw=1, alpha=0.4, 
            label=f'{label_encoder.classes_[i][:20]}' if i < 10 else "")

# Plot micro and macro averages
ax.plot(fpr_micro, tpr_micro, color='darkred', lw=3, linestyle='-', 
        label=f'Micro-average (AUC = {roc_auc_micro:.4f})')
ax.plot(all_fpr, mean_tpr, color='navy', lw=3, linestyle='--', 
        label=f'Macro-average (AUC = {roc_auc_macro:.4f})')

# Diagonal line (random classifier)
ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle=':', label='Random (AUC = 0.5)')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
ax.set_title('Figure 4.X: ROC Curves for 41 Disease Classes', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "Figure_ROC_Curves_All_41.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIGURES_DIR, "Figure_ROC_Curves_All_41.pdf"), bbox_inches='tight')
plt.close()
print("   ✅ Saved: Figure_ROC_Curves_All_41.png/pdf")

# ============================================================================
# Figure 2: ROC Curves (Top 10 Most Common Diseases - cleaner view)
# ============================================================================

print("\n📊 Generating Figure: ROC Curves (Top 10 Most Common Diseases)...")

# Get top 10 most frequent diseases in test set
from collections import Counter
disease_counts = Counter(y_test)
top_10_indices = [idx for idx, _ in disease_counts.most_common(10)]
top_10_names = [label_encoder.classes_[i].replace("_", " ") for i in top_10_indices]

fig, ax = plt.subplots(figsize=(12, 10))

# Plot top 10 classes with distinct colors
top_colors = plt.cm.tab10(np.linspace(0, 1, 10))
for idx, (class_idx, color) in enumerate(zip(top_10_indices, top_colors)):
    ax.plot(fpr[class_idx], tpr[class_idx], color=color, lw=2.5, alpha=0.8,
            label=f'{top_10_names[idx]} (AUC = {roc_auc[class_idx]:.4f})')

# Diagonal line
ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle=':', label='Random (AUC = 0.5)')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
ax.set_title('Figure 4.X: ROC Curves (Top 10 Most Common Diseases)', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "Figure_ROC_Curves_Top_10.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIGURES_DIR, "Figure_ROC_Curves_Top_10.pdf"), bbox_inches='tight')
plt.close()
print("   ✅ Saved: Figure_ROC_Curves_Top_10.png/pdf")

# ============================================================================
# Figure 3: AUC Bar Chart
# ============================================================================

print("\n📊 Generating Figure: AUC Bar Chart...")

fig, ax = plt.subplots(figsize=(14, 8))
disease_names_short = [label_encoder.classes_[i].replace("_", " ")[:30] for i in range(num_classes)]
colors_bar = plt.cm.RdYlGn(np.linspace(0.2, 0.8, num_classes))

bars = ax.barh(range(num_classes), auc_list, color=colors_bar, edgecolor='black', linewidth=0.5)
ax.set_xlabel('AUC Score', fontsize=12)
ax.set_ylabel('Disease', fontsize=12)
ax.set_title('Figure 4.X: AUC Score per Disease', fontsize=14, fontweight='bold')
ax.set_yticks(range(num_classes))
ax.set_yticklabels(disease_names_short, fontsize=8)
ax.set_xlim(0, 1.05)
ax.axvline(x=roc_auc_macro, color='red', linestyle='--', linewidth=2, label=f'Macro-average AUC = {roc_auc_macro:.4f}')
ax.legend()
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "Figure_AUC_Bar_Chart.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIGURES_DIR, "Figure_AUC_Bar_Chart.pdf"), bbox_inches='tight')
plt.close()
print("   ✅ Saved: Figure_AUC_Bar_Chart.png/pdf")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("✅ ROC Analysis Complete!")
print("="*70)
print(f"\n📊 Final AUC Summary:")
print(f"   Micro-average AUC: {roc_auc_micro:.4f}")
print(f"   Macro-average AUC: {roc_auc_macro:.4f}")
print(f"   Mean AUC across all diseases: {np.mean(auc_list):.4f}")
print(f"   Standard deviation: {np.std(auc_list):.4f}")
print(f"\n📁 Figures saved in: {FIGURES_DIR}")
print("="*70)