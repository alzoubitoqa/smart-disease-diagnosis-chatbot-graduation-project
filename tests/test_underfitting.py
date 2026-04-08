"""
File: tests/test_underfitting.py
Purpose: Standalone underfitting detection test
"""

import os
import sys
import numpy as np
import pickle
import tensorflow as tf
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Custom layers (same as before)
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
        cosine = 0.5 * (1 + tf.cos(math.pi * (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)))
        lr = tf.where(step < self.warmup_steps, warmup, cosine)
        return self.learning_rate_base * lr
    def get_config(self):
        return {
            'learning_rate_base': self.learning_rate_base,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps
        }

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "artifacts", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "artifacts", "models")

def to_one_hot(y, num_classes):
    return tf.keras.utils.to_categorical(y, num_classes)

print("="*70)
print("🔬 UNDERFITTING DETECTION TEST")
print("="*70)

# Load data
print("\n📂 Loading preprocessed data...")
X_train = np.load(os.path.join(PROCESSED_DIR, "X_symptoms_train.npy"))
X_val = np.load(os.path.join(PROCESSED_DIR, "X_symptoms_val.npy"))
X_test = np.load(os.path.join(PROCESSED_DIR, "X_symptoms_test.npy"))
X_sev_train = np.load(os.path.join(PROCESSED_DIR, "X_severities_train.npy"))
X_sev_val = np.load(os.path.join(PROCESSED_DIR, "X_severities_val.npy"))
X_sev_test = np.load(os.path.join(PROCESSED_DIR, "X_severities_test.npy"))
y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
y_val = np.load(os.path.join(PROCESSED_DIR, "y_val.npy"))
y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), 'rb') as f:
    le = pickle.load(f)
num_classes = len(le.classes_)

print(f"   Train: {len(X_train)} samples")
print(f"   Val: {len(X_val)} samples")
print(f"   Test: {len(X_test)} samples")

# Load model
print("\n🤖 Loading trained model...")
model_path = os.path.join(MODELS_DIR, "bilstm_final.keras")
custom_objects = {'Attention': Attention, 'CosineWarmup': CosineWarmup}
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("✅ Model loaded")

# Convert to one-hot
y_train_oh = to_one_hot(y_train, num_classes)
y_val_oh = to_one_hot(y_val, num_classes)
y_test_oh = to_one_hot(y_test, num_classes)

# Evaluate
train_loss, train_acc = model.evaluate([X_train, X_sev_train], y_train_oh, verbose=0)
val_loss, val_acc = model.evaluate([X_val, X_sev_val], y_val_oh, verbose=0)
test_loss, test_acc = model.evaluate([X_test, X_sev_test], y_test_oh, verbose=0)

print(f"\n📊 Performance:")
print(f"   Train Accuracy: {train_acc*100:.2f}%   Train Loss: {train_loss:.4f}")
print(f"   Val Accuracy:   {val_acc*100:.2f}%   Val Loss:   {val_loss:.4f}")
print(f"   Test Accuracy:  {test_acc*100:.2f}%   Test Loss:  {test_loss:.4f}")

# Underfitting conditions
print("\n📊 Underfitting Analysis:")
underfitting_score = 0

if train_acc < 0.70:
    print("   ⚠️ Low training accuracy (<70%) - model failed to learn training data")
    underfitting_score += 1
elif train_acc < 0.85:
    print("   🟡 Moderate training accuracy (70-85%) - possible underfitting")
    underfitting_score += 1
else:
    print("   ✅ High training accuracy (>85%) - model can learn training data")

if train_acc < 0.85 and (val_acc - train_acc) < 0.05:
    print("   ⚠️ Training accuracy low but validation similar - underfitting likely")
    underfitting_score += 1

if train_loss > 1.0 and train_acc < 0.90:
    print("   ⚠️ High training loss with low accuracy - underfitting")
    underfitting_score += 1

print("\n" + "="*70)
print("🎯 FINAL VERDICT")
print("="*70)
if underfitting_score >= 2:
    print("❌ Model shows signs of underfitting.")
    print("   Suggestions: Increase model complexity, add more layers/units, train longer, or reduce regularization.")
elif underfitting_score == 1:
    print("🟡 Mild underfitting detected. Consider slightly increasing model capacity.")
else:
    print("✅ No underfitting detected. Model successfully learns training data.")
print("="*70)