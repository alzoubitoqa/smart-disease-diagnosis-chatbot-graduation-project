"""
File: tests/test_overfitting.py
Purpose: Standalone overfitting detection test for the trained BiLSTM model
"""

import os
import sys
import numpy as np
import pickle
import tensorflow as tf
import math

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ============================================================================
# Custom Layers (needed for loading the model)
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
        cosine = 0.5 * (1 + tf.cos(math.pi * (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)))
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "artifacts", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "artifacts", "models")

def to_one_hot(y, num_classes):
    return tf.keras.utils.to_categorical(y, num_classes)

# ============================================================================
# Load Data and Model
# ============================================================================

print("="*70)
print("🔬 OVERFITTING DETECTION TEST")
print("="*70)

print("\n📂 Loading preprocessed data...")
X_symptoms_train = np.load(os.path.join(PROCESSED_DIR, "X_symptoms_train.npy"))
X_symptoms_val = np.load(os.path.join(PROCESSED_DIR, "X_symptoms_val.npy"))
X_symptoms_test = np.load(os.path.join(PROCESSED_DIR, "X_symptoms_test.npy"))
X_severities_train = np.load(os.path.join(PROCESSED_DIR, "X_severities_train.npy"))
X_severities_val = np.load(os.path.join(PROCESSED_DIR, "X_severities_val.npy"))
X_severities_test = np.load(os.path.join(PROCESSED_DIR, "X_severities_test.npy"))
y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
y_val = np.load(os.path.join(PROCESSED_DIR, "y_val.npy"))
y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

print(f"   Train: {len(X_symptoms_train)} samples")
print(f"   Validation: {len(X_symptoms_val)} samples")
print(f"   Test: {len(X_symptoms_test)} samples")

with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), 'rb') as f:
    label_encoder = pickle.load(f)
num_classes = len(label_encoder.classes_)

print("\n🤖 Loading trained model...")
model_path = os.path.join(MODELS_DIR, "bilstm_final.keras")
custom_objects = {'Attention': Attention, 'CosineWarmup': CosineWarmup}
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("✅ Model loaded successfully")

# ============================================================================
# Overfitting Test
# ============================================================================

print("\n" + "="*70)
print("📊 PERFORMING OVERFITTING ANALYSIS")
print("="*70)

# Convert labels to one-hot
y_train_onehot = to_one_hot(y_train, num_classes)
y_val_onehot = to_one_hot(y_val, num_classes)
y_test_onehot = to_one_hot(y_test, num_classes)

# Evaluate on all sets
train_loss, train_acc = model.evaluate([X_symptoms_train, X_severities_train], y_train_onehot, verbose=0)
val_loss, val_acc = model.evaluate([X_symptoms_val, X_severities_val], y_val_onehot, verbose=0)
test_loss, test_acc = model.evaluate([X_symptoms_test, X_severities_test], y_test_onehot, verbose=0)

print(f"\n📊 Performance Comparison:")
print(f"   Train Accuracy: {train_acc*100:.2f}%   Train Loss: {train_loss:.6f}")
print(f"   Val Accuracy:   {val_acc*100:.2f}%   Val Loss:   {val_loss:.6f}")
print(f"   Test Accuracy:  {test_acc*100:.2f}%   Test Loss:  {test_loss:.6f}")

# Calculate gaps
train_val_gap = (train_acc - val_acc) * 100
train_test_gap = (train_acc - test_acc) * 100
val_test_gap = (val_acc - test_acc) * 100

print(f"\n📊 Performance Gaps:")
print(f"   Train - Val:  {train_val_gap:+.2f}%")
print(f"   Train - Test: {train_test_gap:+.2f}%")
print(f"   Val - Test:   {val_test_gap:+.2f}%")

# Loss analysis
loss_ratio = val_loss / train_loss
print(f"\n📊 Loss Ratio (Val/Train): {loss_ratio:.4f}")

# Diagnosis
print("\n📊 Overfitting Diagnosis:")
overfitting_score = 0
if train_val_gap > 5:
    print("   ⚠️ Large train-val gap (>5%) - possible overfitting")
    overfitting_score += 1
elif train_val_gap < 1:
    print("   ✅ Very small train-val gap (<1%) - good generalization")
else:
    print("   🟡 Moderate train-val gap (1-5%) - acceptable")

if loss_ratio > 2.0:
    print("   ⚠️ Validation loss much higher than training loss - overfitting likely")
    overfitting_score += 1
elif loss_ratio < 1.2:
    print("   ✅ Loss ratio close to 1 - well generalized")
else:
    print("   🟡 Loss ratio moderately elevated")

if train_acc > 0.99 and test_acc > 0.99:
    print("   ✅ Both train and test accuracy very high - excellent performance")
elif train_acc > 0.99 and test_acc < 0.95:
    print("   ⚠️ High train but lower test accuracy - overfitting suspected")
    overfitting_score += 1

# Final verdict
print("\n" + "="*70)
print("🎯 FINAL VERDICT")
print("="*70)
if overfitting_score >= 2:
    print("❌ Model shows signs of overfitting.")
    print("   Suggestions: Increase dropout, reduce model complexity, add more data, or use stronger regularization.")
elif overfitting_score == 1:
    print("🟡 Mild overfitting detected. Acceptable but monitor performance on new data.")
else:
    print("✅ No significant overfitting detected. Model generalizes well.")
print("="*70)

# Summary dictionary
results = {
    'train_acc': float(train_acc),
    'val_acc': float(val_acc),
    'test_acc': float(test_acc),
    'train_loss': float(train_loss),
    'val_loss': float(val_loss),
    'test_loss': float(test_loss),
    'train_val_gap': float(train_val_gap),
    'loss_ratio': float(loss_ratio),
    'overfitting_score': overfitting_score
}