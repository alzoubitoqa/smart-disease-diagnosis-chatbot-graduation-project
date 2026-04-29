# for preprocessing
import os
import sys
import numpy as np
import pickle
import json
import math
import tensorflow as tf

from sklearn.metrics import accuracy_score, f1_score, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from scipy.stats import ks_2samp


# ============================================================================
# 0. Custom Objects Needed to Load the Keras Model
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
# 1. Set Paths
# ============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "artifacts", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "artifacts", "models")
REPORTS_DIR = os.path.join(BASE_DIR, "artifacts", "reports")


# ============================================================================
# 2. Load Data and Model
# ============================================================================

print("=" * 70)
print("🔬 COMPREHENSIVE TESTING - Overfitting & Data Leakage")
print("=" * 70)

print("\n📂 Loading preprocessed data...")

required_files = [
    "X_symptoms_train.npy",
    "X_symptoms_val.npy",
    "X_symptoms_test.npy",
    "X_severities_train.npy",
    "X_severities_val.npy",
    "X_severities_test.npy",
    "y_train.npy",
    "y_val.npy",
    "y_test.npy"
]

missing_files = []
for file in required_files:
    if not os.path.exists(os.path.join(PROCESSED_DIR, file)):
        missing_files.append(file)

if missing_files:
    print(f"\n❌ Missing files: {missing_files}")
    print("   Please run preprocessing first:")
    print("   python src/preprocessing/preprocessor.py")
    sys.exit(1)

X_symptoms_train = np.load(os.path.join(PROCESSED_DIR, "X_symptoms_train.npy"))
X_symptoms_val = np.load(os.path.join(PROCESSED_DIR, "X_symptoms_val.npy"))
X_symptoms_test = np.load(os.path.join(PROCESSED_DIR, "X_symptoms_test.npy"))

X_severities_train = np.load(os.path.join(PROCESSED_DIR, "X_severities_train.npy"))
X_severities_val = np.load(os.path.join(PROCESSED_DIR, "X_severities_val.npy"))
X_severities_test = np.load(os.path.join(PROCESSED_DIR, "X_severities_test.npy"))

y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
y_val = np.load(os.path.join(PROCESSED_DIR, "y_val.npy"))
y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

print(f"   - Train: {len(X_symptoms_train)} samples")
print(f"   - Validation: {len(X_symptoms_val)} samples")
print(f"   - Test: {len(X_symptoms_test)} samples")
print(f"   - Sequence length: {X_symptoms_train.shape[1]}")

print("\n🏷️ Loading label encoder...")

label_encoder_path = os.path.join(MODELS_DIR, "label_encoder.pkl")

if not os.path.exists(label_encoder_path):
    print(f"❌ Label encoder not found at: {label_encoder_path}")
    print("   Please run preprocessing first")
    sys.exit(1)

with open(label_encoder_path, "rb") as f:
    le = pickle.load(f)

print("\n🤖 Loading trained model...")

model_path = os.path.join(MODELS_DIR, "bilstm_best.keras")

if not os.path.exists(model_path):
    print(f"❌ Model not found at: {model_path}")
    print("   Please run training first: python src/ml/train_bilstm.py")
    sys.exit(1)

model = load_model(
    model_path,
    custom_objects={
        "Attention": Attention,
        "CosineWarmup": CosineWarmup
    },
    compile=False
)

print("✅ Model loaded successfully!")

num_classes = len(le.classes_)
y_train_cat = to_categorical(y_train, num_classes)
y_val_cat = to_categorical(y_val, num_classes)
y_test_cat = to_categorical(y_test, num_classes)


# ============================================================================
# 3. Test 1: Overfitting Test
# ============================================================================

print("\n" + "=" * 70)
print("📈 1. OVERFITTING TEST")
print("=" * 70)

print("\n🔄 Making predictions...")

y_train_pred_probs = model.predict([X_symptoms_train, X_severities_train], verbose=0)
y_val_pred_probs = model.predict([X_symptoms_val, X_severities_val], verbose=0)
y_test_pred_probs = model.predict([X_symptoms_test, X_severities_test], verbose=0)

y_train_pred = np.argmax(y_train_pred_probs, axis=1)
y_val_pred = np.argmax(y_val_pred_probs, axis=1)
y_test_pred = np.argmax(y_test_pred_probs, axis=1)

train_acc = accuracy_score(y_train, y_train_pred)
val_acc = accuracy_score(y_val, y_val_pred)
test_acc = accuracy_score(y_test, y_test_pred)

train_f1 = f1_score(y_train, y_train_pred, average="macro")
val_f1 = f1_score(y_val, y_val_pred, average="macro")
test_f1 = f1_score(y_test, y_test_pred, average="macro")

print("\n📊 Performance Comparison:")
print(f"   {'Set':<12} {'Accuracy':<15} {'F1-Score':<15}")
print(f"   {'-' * 42}")
print(f"   {'Train':<12} {train_acc * 100:>6.2f}%{'':<6} {train_f1 * 100:>6.2f}%")
print(f"   {'Validation':<12} {val_acc * 100:>6.2f}%{'':<6} {val_f1 * 100:>6.2f}%")
print(f"   {'Test':<12} {test_acc * 100:>6.2f}%{'':<6} {test_f1 * 100:>6.2f}%")

train_val_gap = (train_acc - val_acc) * 100
train_test_gap = (train_acc - test_acc) * 100

print("\n📊 Gaps:")
print(f"   Train - Validation Gap: {train_val_gap:.2f}%")
print(f"   Train - Test Gap: {train_test_gap:.2f}%")

print("\n✅ Overfitting Evaluation:")

if train_test_gap < 5:
    print(f"   ✅ NO OVERFITTING! (Gap = {train_test_gap:.2f}%)")
    print("   The model generalizes well to new data.")
elif train_test_gap < 10:
    print(f"   ⚠️ Mild Overfitting detected (Gap = {train_test_gap:.2f}%)")
else:
    print(f"   ❌ Severe Overfitting detected (Gap = {train_test_gap:.2f}%)")


# ============================================================================
# 4. Test 2: Pattern Overlap Analysis
# ============================================================================

print("\n" + "=" * 70)
print("🔬 2. PATTERN OVERLAP ANALYSIS")
print("=" * 70)


def get_patterns(X):
    patterns = []
    for row in X:
        pattern = tuple(sorted([s for s in row if s != 0]))
        patterns.append(pattern)
    return set(patterns)


train_patterns = get_patterns(X_symptoms_train)
val_patterns = get_patterns(X_symptoms_val)
test_patterns = get_patterns(X_symptoms_test)

print("\n📊 Unique Symptom Patterns:")
print(f"   Train: {len(train_patterns)} patterns")
print(f"   Validation: {len(val_patterns)} patterns")
print(f"   Test: {len(test_patterns)} patterns")

pattern_train_val = len(train_patterns.intersection(val_patterns))
pattern_train_test = len(train_patterns.intersection(test_patterns))

print("\n📊 Pattern Overlaps (Same Symptoms):")
print(f"   Train ∩ Validation: {pattern_train_val}")
print(f"   Train ∩ Test: {pattern_train_test}")

print("\n📖 INTERPRETATION:")
print("   Pattern overlap here checks symptom sets after sorting.")
print("   Exact leakage is tested more strictly in the next section.")


# ============================================================================
# 5. Test 3: Exact Sample Overlap
# ============================================================================

print("\n" + "=" * 70)
print("🔒 3. EXACT SAMPLE OVERLAP TEST")
print("=" * 70)


def get_sample_id(symptoms, severities):
    pairs = tuple(sorted(zip(symptoms, severities)))
    return pairs


train_ids = set(
    get_sample_id(X_symptoms_train[i], X_severities_train[i])
    for i in range(len(X_symptoms_train))
)

val_ids = set(
    get_sample_id(X_symptoms_val[i], X_severities_val[i])
    for i in range(len(X_symptoms_val))
)

test_ids = set(
    get_sample_id(X_symptoms_test[i], X_severities_test[i])
    for i in range(len(X_symptoms_test))
)

train_val_overlap = len(train_ids.intersection(val_ids))
train_test_overlap = len(train_ids.intersection(test_ids))
val_test_overlap = len(val_ids.intersection(test_ids))

print("\n📊 Exact Sample Overlap (Symptoms + Severity):")
print(f"   Train ∩ Validation: {train_val_overlap}")
print(f"   Train ∩ Test: {train_test_overlap}")
print(f"   Validation ∩ Test: {val_test_overlap}")

print("\n📖 INTERPRETATION:")

if train_test_overlap == 0:
    print("   ✅ NO EXACT SAMPLE OVERLAP!")
    print("   No identical symptom-severity sample appears in both Train and Test.")
else:
    print(f"   ⚠️ NOTE: {train_test_overlap} exact samples appear in both Train and Test.")


# ============================================================================
# 6. Test 4: Feature Distribution Test
# ============================================================================

print("\n" + "=" * 70)
print("📊 4. FEATURE DISTRIBUTION TEST")
print("=" * 70)

train_mean = np.mean(X_symptoms_train, axis=0)
val_mean = np.mean(X_symptoms_val, axis=0)
test_mean = np.mean(X_symptoms_test, axis=0)

print("\n📈 Mean Symptom Index per Set:")
print(f"   Train Mean: {np.mean(train_mean):.4f}")
print(f"   Validation Mean: {np.mean(val_mean):.4f}")
print(f"   Test Mean: {np.mean(test_mean):.4f}")

ks_train_val = ks_2samp(train_mean, val_mean)
ks_train_test = ks_2samp(train_mean, test_mean)

print("\n📊 Kolmogorov-Smirnov Test Results:")
print(f"   Train vs Validation: p-value = {ks_train_val.pvalue:.4f}")
print(f"   Train vs Test: p-value = {ks_train_test.pvalue:.4f}")

print("\n✅ Distribution Analysis:")

if ks_train_val.pvalue > 0.05 and ks_train_test.pvalue > 0.05:
    print("   ✅ Distributions are similar - No major split bias detected.")
else:
    print("   ⚠️ Distributions differ - Possible split distribution difference detected.")


# ============================================================================
# 7. Test 5: Per-Class Performance
# ============================================================================

print("\n" + "=" * 70)
print("📋 5. PER-CLASS PERFORMANCE")
print("=" * 70)

print("\n📊 Classification Report (Test Set):")
print(
    classification_report(
        y_test,
        y_test_pred,
        target_names=le.classes_,
        zero_division=0
    )
)


# ============================================================================
# 8. Load Preprocessing Report
# ============================================================================

print("\n" + "=" * 70)
print("📁 6. PREPROCESSING REPORT")
print("=" * 70)

report_path = os.path.join(REPORTS_DIR, "preprocessing_report.json")

if os.path.exists(report_path):
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    print("\n📊 Dataset Statistics:")
    print(f"   - Total samples: {report.get('total_samples', 'N/A')}")
    print(f"   - Train samples: {report.get('train_samples', 'N/A')}")
    print(f"   - Validation samples: {report.get('val_samples', 'N/A')}")
    print(f"   - Test samples: {report.get('test_samples', 'N/A')}")
    print(f"   - Unique symptoms: {report.get('unique_symptoms', 'N/A')}")
    print(f"   - Severity vocabulary size: {report.get('severity_vocab_size', 'N/A')}")
    print(f"   - Avg symptoms per sample: {float(report.get('avg_symptoms_per_sample', 0)):.2f}")
    print(f"   - Avg severity: {float(report.get('avg_severity', 0)):.2f}")
    print(f"   - Unique patterns: {report.get('unique_patterns', 'N/A')}")
    print(f"   - Pattern overlap (Train ∩ Test): {report.get('pattern_overlap', 'N/A')}")
else:
    print("\n⚠️ Preprocessing report not found")


# ============================================================================
# 9. Load Training Report
# ============================================================================

print("\n" + "=" * 70)
print("📈 7. TRAINING REPORT")
print("=" * 70)

training_report_path = os.path.join(REPORTS_DIR, "training_complete_report.json")

if os.path.exists(training_report_path):
    with open(training_report_path, "r", encoding="utf-8") as f:
        training_report = json.load(f)

    data_info = training_report.get("data_info", {})
    training_results = training_report.get("training_results", {})
    test_metrics = training_report.get("test_metrics", {})

    print("\n📊 Training Statistics:")
    print(f"   - Train samples before augmentation: {data_info.get('train_samples_before_aug', 'N/A')}")
    print(f"   - Train samples after augmentation: {data_info.get('train_samples_after_aug', 'N/A')}")
    print(f"   - Validation samples: {data_info.get('val_samples', 'N/A')}")
    print(f"   - Test samples: {data_info.get('test_samples', 'N/A')}")
    print(f"   - Severity vocabulary size: {data_info.get('severity_vocab_size', 'N/A')}")
    print(f"   - Train Accuracy: {float(training_results.get('train_accuracy', 0)) * 100:.2f}%")
    print(f"   - Validation Accuracy: {float(training_results.get('val_accuracy', 0)) * 100:.2f}%")
    print(f"   - Test Accuracy: {float(training_results.get('test_accuracy', 0)) * 100:.2f}%")
    print(f"   - F1-Score Macro: {float(test_metrics.get('f1_macro', 0)) * 100:.2f}%")
    print(f"   - Top-3 Accuracy: {float(test_metrics.get('top3_accuracy', 0)) * 100:.2f}%")
    print(f"   - Correct Predictions: {test_metrics.get('correct_predictions', 'N/A')}")
    print(f"   - Wrong Predictions: {test_metrics.get('wrong_predictions', 'N/A')}")
    print(f"   - Epochs completed: {training_results.get('epochs_completed', 'N/A')}")
else:
    print("\n⚠️ Training report not found")


# ============================================================================
# 10. FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("🎯 FINAL SUMMARY")
print("=" * 70)

print(f"""
┌─────────────────────────────────────────────────────────────────┐
│                    TEST RESULTS                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📈 OVERFITTING TEST:                                           │
│     - Train Accuracy: {train_acc * 100:.2f}%                    │
│     - Validation Accuracy: {val_acc * 100:.2f}%                 │
│     - Test Accuracy: {test_acc * 100:.2f}%                      │
│     - Train-Test Gap: {train_test_gap:.2f}%                     │
│     - Status: {'✅ NO OVERFITTING' if train_test_gap < 5 else '⚠️ OVERFITTING'}       
│                                                                 │
│  🔬 PATTERN OVERLAP:                                            │
│     - Train ∩ Test: {pattern_train_test} patterns               │
│                                                                 │
│  🔒 EXACT SAMPLE OVERLAP:                                       │
│     - Train ∩ Test: {train_test_overlap} samples                │
│     - Status: {'✅ NO DATA LEAKAGE' if train_test_overlap == 0 else '⚠️ POSSIBLE DUPLICATES'}   
│                                                                 │
│  📊 PERFORMANCE:                                                │
│     - Test Accuracy: {test_acc * 100:.2f}%                      │
│     - Test F1-Score Macro: {test_f1 * 100:.2f}%                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
""")


# ============================================================================
# 11. FINAL VERDICT
# ============================================================================

print("\n" + "=" * 70)
print("🏆 FINAL VERDICT")
print("=" * 70)

if train_test_gap < 5 and train_test_overlap == 0:
    print(f"""
    ✅✅✅ MODEL PASSED THE TESTING CHECKS ✅✅✅

    ✓ No overfitting detected (Gap = {train_test_gap:.2f}%)
    ✓ No exact sample overlap between Train and Test
    ✓ Test accuracy: {test_acc * 100:.2f}%
    ✓ Test F1-score macro: {test_f1 * 100:.2f}%

    Note:
    Perfect benchmark performance should still be interpreted within the
    structured and controlled nature of the dataset.
    """)
elif train_test_gap < 5:
    print(f"""
    ✅ MODEL PERFORMANCE IS STRONG

    ✓ No overfitting detected (Gap = {train_test_gap:.2f}%)
    ✓ Test accuracy: {test_acc * 100:.2f}%

    ⚠️ However, exact sample overlap was detected.
    Please review the dataset split if needed.
    """)
else:
    print(f"""
    ⚠️ Model may need improvement:

    - Overfitting gap: {train_test_gap:.2f}%
    - Consider additional regularization or reviewing the split.
    """)

print("\n" + "=" * 70)
print("✅ TESTING COMPLETED")
print("=" * 70)