# ============================================================================
# BiLSTM MODEL WITH ADVANCED IMPROVEMENTS (FINAL VERSION)
# Merged from Shaden, Tuqa, Ghida + Attention + Residual + Cosine Decay + Gate
# ============================================================================

import os
import sys
import json
import math
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, top_k_accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Embedding, Bidirectional, LSTM, Dense, Dropout,
    BatchNormalization, Input, Concatenate, Add, Multiply,
    Layer, Activation
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import AdamW

# ============================================================================
# 1. Custom Layers (Attention, CosineWarmup)
# ============================================================================

class Attention(Layer):
    """Custom attention layer for sequence weighting"""
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
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

class CosineWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Cosine annealing with warmup"""
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
# 2. Configuration (Enhanced)
# ============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, "artifacts", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "artifacts", "models")
REPORTS_DIR = os.path.join(BASE_DIR, "artifacts", "reports")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Enhanced hyperparameters
EMBEDDING_DIM = 256
LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64
DROPOUT_RATE = 0.4
RECURRENT_DROPOUT = 0.3
L2_REG = 5e-4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
EPOCHS = 200
PATIENCE = 20
REDUCE_LR_PATIENCE = 8
MIN_LR = 1e-6
LABEL_SMOOTHING = 0.1
GRADIENT_CLIP_NORM = 1.0

# Data augmentation
AUGMENTATION_FACTOR = 0.5

# Checkpoint paths
BEST_WEIGHTS_PATH = os.path.join(MODELS_DIR, "bilstm_best.weights.h5")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "bilstm_best.keras")
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, "bilstm_final.keras")

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# 3. Data Loading
# ============================================================================

def load_data():
    """Load processed data from artifacts"""
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

    with open(os.path.join(MODELS_DIR, "symptom2idx.pkl"), 'rb') as f:
        symptom2idx = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "severity2idx.pkl"), 'rb') as f:
        severity2idx = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), 'rb') as f:
        label_encoder = pickle.load(f)

    print(f"   - Train: {len(X_symptoms_train)} samples")
    print(f"   - Validation: {len(X_symptoms_val)} samples")
    print(f"   - Test: {len(X_symptoms_test)} samples")
    print(f"   - Sequence length: {X_symptoms_train.shape[1]}")
    print(f"   - Vocabulary size: {len(symptom2idx)}")
    print(f"   - Severity vocabulary size: {len(severity2idx)}")
    print(f"   - Number of diseases: {len(label_encoder.classes_)}")

    return {
        'X_symptoms_train': X_symptoms_train,
        'X_symptoms_val': X_symptoms_val,
        'X_symptoms_test': X_symptoms_test,
        'X_severities_train': X_severities_train,
        'X_severities_val': X_severities_val,
        'X_severities_test': X_severities_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'symptom2idx': symptom2idx,
        'severity2idx': severity2idx,
        'label_encoder': label_encoder,
        'vocab_size': len(symptom2idx),
        'severity_vocab_size': len(severity2idx),
        'num_classes': len(label_encoder.classes_),
        'max_len': X_symptoms_train.shape[1]
    }

# ============================================================================
# 4. Enhanced Data Augmentation (Realistic symptom sampling)
# ============================================================================

def compute_symptom_distribution(X_symptoms, vocab_size):
    """Compute empirical distribution of symptom indices (ignoring padding 0)"""
    all_symptoms = X_symptoms.flatten()
    all_symptoms = all_symptoms[all_symptoms != 0]
    counts = np.bincount(all_symptoms, minlength=vocab_size)
    probs = counts.astype(np.float32) / counts.sum()
    return probs

def data_augmentation(X_symptoms, X_severities, y, aug_factor=0.5):
    n_samples = len(X_symptoms)
    n_augmented = int(n_samples * aug_factor)
    print(f"\n🔄 Performing enhanced data augmentation...")
    print(f"   - Original samples: {n_samples}")
    print(f"   - Augmented samples: {n_augmented}")

    vocab_size = np.max(X_symptoms) + 1
    severity_vocab_size = np.max(X_severities) + 1
    symptom_probs = compute_symptom_distribution(X_symptoms, vocab_size)

    X_symptoms_aug = list(X_symptoms)
    X_severities_aug = list(X_severities)
    y_aug = list(y)

    for _ in range(n_augmented):
        idx = np.random.randint(0, n_samples)
        symptom_seq = X_symptoms[idx].copy()
        severity_seq = X_severities[idx].copy()

        symptom_positions = np.where(symptom_seq != 0)[0]
        if len(symptom_positions) > 0:
            n_modify = np.random.randint(1, min(4, len(symptom_positions)))
            modify_idx = np.random.choice(symptom_positions, n_modify, replace=False)
            new_symptoms = np.random.choice(vocab_size, size=n_modify, p=symptom_probs)
            symptom_seq[modify_idx] = new_symptoms

            for pos in modify_idx:
                current_sev = severity_seq[pos]
                if current_sev > 1:
                    delta = np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2])
                    new_sev = max(1, min(severity_vocab_size - 1, current_sev + delta))
                    severity_seq[pos] = new_sev

        X_symptoms_aug.append(symptom_seq)
        X_severities_aug.append(severity_seq)
        y_aug.append(y[idx])

    X_symptoms_aug = np.array(X_symptoms_aug, dtype=np.int32)
    X_severities_aug = np.array(X_severities_aug, dtype=np.int32)
    y_aug = np.array(y_aug, dtype=np.int32)

    print(f"   - After augmentation: {len(X_symptoms_aug)} samples")
    print(f"   - Increase: {(len(X_symptoms_aug)/n_samples - 1)*100:.1f}%")
    return X_symptoms_aug, X_severities_aug, y_aug

# ============================================================================
# 5. Build Enhanced BiLSTM Model
# ============================================================================

def build_enhanced_bilstm_model(data):
    vocab_size = data['vocab_size']
    severity_vocab_size = data['severity_vocab_size']
    max_len = data['max_len']
    num_classes = data['num_classes']

    symptom_input = Input(shape=(max_len,), name='symptom_input')
    severity_input = Input(shape=(max_len,), name='severity_input')

    symptom_emb = Embedding(
        input_dim=vocab_size,
        output_dim=EMBEDDING_DIM,
        input_length=max_len,
        mask_zero=True,
        embeddings_regularizer=regularizers.l2(L2_REG),
        name='symptom_embedding'
    )(symptom_input)

    severity_emb = Embedding(
        input_dim=severity_vocab_size,
        output_dim=16,
        input_length=max_len,
        mask_zero=True,
        embeddings_regularizer=regularizers.l2(L2_REG),
        name='severity_embedding'
    )(severity_input)

    # Multiplicative gate: severity modulates symptom embedding
    gate = Dense(EMBEDDING_DIM, activation='sigmoid', name='severity_gate')(severity_emb)
    combined = Multiply(name='gated_symptoms')([symptom_emb, gate])

    # First BiLSTM with residual connection
    lstm1 = Bidirectional(
        LSTM(LSTM_UNITS_1, dropout=DROPOUT_RATE, recurrent_dropout=RECURRENT_DROPOUT,
             return_sequences=True, kernel_regularizer=regularizers.l2(L2_REG)),
        name='bilstm_1'
    )(combined)
    lstm1 = Dropout(DROPOUT_RATE, name='dropout_1')(lstm1)

    shortcut = Dense(lstm1.shape[-1], name='shortcut')(combined)
    lstm1 = Add(name='residual_1')([lstm1, shortcut])

    # Second BiLSTM
    lstm2 = Bidirectional(
        LSTM(LSTM_UNITS_2, dropout=DROPOUT_RATE, recurrent_dropout=RECURRENT_DROPOUT,
             return_sequences=True, kernel_regularizer=regularizers.l2(L2_REG)),
        name='bilstm_2'
    )(lstm1)
    lstm2 = Dropout(DROPOUT_RATE, name='dropout_2')(lstm2)

    # Attention layer
    attention_output = Attention(name='attention')(lstm2)
    bn = BatchNormalization(name='batch_norm')(attention_output)

    dense1 = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(L2_REG), name='dense_1')(bn)
    dense1 = Dropout(DROPOUT_RATE + 0.1, name='dropout_3')(dense1)

    dense2 = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(L2_REG), name='dense_2')(dense1)
    dense2 = Dropout(DROPOUT_RATE, name='dropout_4')(dense2)

    output = Dense(num_classes, activation='softmax', name='output')(dense2)

    model = Model(inputs=[symptom_input, severity_input], outputs=output)

    # Cosine decay with warmup
    total_steps = (len(data['X_symptoms_train']) // BATCH_SIZE) * EPOCHS
    warmup_steps = int(0.1 * total_steps)
    lr_schedule = CosineWarmup(LEARNING_RATE, warmup_steps, total_steps)

    optimizer = AdamW(
        learning_rate=lr_schedule,
        weight_decay=WEIGHT_DECAY,
        clipnorm=GRADIENT_CLIP_NORM
    )

    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    return model

# ============================================================================
# 6. Callbacks Setup
# ============================================================================

def setup_callbacks():
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=REDUCE_LR_PATIENCE,
        min_lr=MIN_LR,
        verbose=1
    )
    checkpoint = ModelCheckpoint(
        BEST_WEIGHTS_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,
        mode='max',
        verbose=1
    )
    return [early_stop, reduce_lr, checkpoint]

# ============================================================================
# 7. Helper for one-hot conversion
# ============================================================================

def to_one_hot(y, num_classes):
    return tf.keras.utils.to_categorical(y, num_classes)

# ============================================================================
# 8. Training Visualization
# ============================================================================

def plot_training_history(history):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history.history['accuracy'], label='Train Acc', linewidth=2)
    axes[0, 1].plot(history.history['val_accuracy'], label='Val Acc', linewidth=2)
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    if 'lr' in history.history:
        axes[1, 0].plot(history.history['lr'], label='Learning Rate')
        axes[1, 0].set_xlabel('Epochs')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate over Time')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)

    if len(history.history['accuracy']) > 5:
        train_acc_last = np.array(history.history['accuracy'][-5:])
        val_acc_last = np.array(history.history['val_accuracy'][-5:])
        gap = train_acc_last - val_acc_last
        axes[1, 1].bar(range(5), gap)
        axes[1, 1].set_xlabel('Last 5 Epochs')
        axes[1, 1].set_ylabel('Gap (Train - Val)')
        axes[1, 1].set_title('Train-Val Gap (Last 5 Epochs)')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].grid(True, alpha=0.3)
        avg_gap = np.mean(gap)
        axes[1, 1].text(2, max(gap)/2 if len(gap)>0 else 0, f'Avg Gap: {avg_gap:.4f}',
                        bbox=dict(boxstyle="round", facecolor="wheat"))

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "training_curves.png"), dpi=100, bbox_inches='tight')
    plt.close()
    print(f"\n📈 Training curves saved to: {os.path.join(REPORTS_DIR, 'training_curves.png')}")

# ============================================================================
# 9. Evaluation Functions
# ============================================================================

def evaluate_model(model, data):
    print("\n" + "="*70)
    print("📊 COMPREHENSIVE MODEL EVALUATION")
    print("="*70)

    X_test_symptoms = data['X_symptoms_test']
    X_test_severities = data['X_severities_test']
    y_test = data['y_test']
    label_encoder = data['label_encoder']
    num_classes = data['num_classes']

    y_pred_probs = model.predict([X_test_symptoms, X_test_severities], verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    test_acc = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print("\n1️⃣ BASIC METRICS:")
    print("-" * 40)
    print(f"   Test Accuracy: {test_acc*100:.4f}%")
    print(f"   Precision (Macro): {precision_macro:.6f}")
    print(f"   Recall (Macro): {recall_macro:.6f}")
    print(f"   F1-Score (Macro): {f1_macro:.6f}")
    print(f"   Precision (Weighted): {precision_weighted:.6f}")
    print(f"   Recall (Weighted): {recall_weighted:.6f}")
    print(f"   F1-Score (Weighted): {f1_weighted:.6f}")

    top2_acc = top_k_accuracy_score(y_test, y_pred_probs, k=2, labels=range(num_classes))
    top3_acc = top_k_accuracy_score(y_test, y_pred_probs, k=3, labels=range(num_classes))
    top5_acc = top_k_accuracy_score(y_test, y_pred_probs, k=5, labels=range(num_classes))

    print("\n2️⃣ TOP-K ACCURACY:")
    print("-" * 40)
    print(f"   Top-2 Accuracy: {top2_acc*100:.4f}%")
    print(f"   Top-3 Accuracy: {top3_acc*100:.4f}%")
    print(f"   Top-5 Accuracy: {top5_acc*100:.4f}%")

    print("\n3️⃣ CLASSIFICATION REPORT:")
    print("-" * 70)
    report_dict = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    print(report_df.round(6).head(20))
    print(f"\n   ... and {len(label_encoder.classes_) - 20} more classes")
    report_df.to_csv(os.path.join(REPORTS_DIR, "classification_report.csv"))

    cm = confusion_matrix(y_test, y_pred)
    class_freq = np.sum(cm, axis=1)
    sorted_idx = np.argsort(class_freq)[::-1]
    cm_sorted = cm[sorted_idx][:, sorted_idx]
    labels_sorted = label_encoder.classes_[sorted_idx]

    plt.figure(figsize=(24, 20))
    cm_percent = cm_sorted.astype('float') / cm_sorted.sum(axis=1)[:, np.newaxis] * 100
    annot = np.empty_like(cm_sorted, dtype=object)
    for i in range(cm_sorted.shape[0]):
        for j in range(cm_sorted.shape[1]):
            annot[i, j] = f"{cm_sorted[i,j]}\n({cm_percent[i,j]:.1f}%)"

    sns.heatmap(cm_sorted, annot=annot, fmt='', cmap='Blues',
                xticklabels=labels_sorted, yticklabels=labels_sorted,
                annot_kws={'size': 7}, cbar_kws={'label': 'Count'})
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.title('Confusion Matrix (Sorted by Class Frequency)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "confusion_matrix_sorted.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Sorted confusion matrix saved to: {os.path.join(REPORTS_DIR, 'confusion_matrix_sorted.png')}")

    misclassified = np.where(y_test != y_pred)[0]
    error_pairs = [(y_test[i], y_pred[i]) for i in misclassified]
    common_errors = Counter(error_pairs).most_common(15)
    print("\n4️⃣ MOST COMMON MISCLASSIFICATIONS:")
    print("-" * 50)
    for (true, pred), count in common_errors:
        print(f"   {label_encoder.classes_[true]:30s} -> {label_encoder.classes_[pred]:30s} : {count} times")

    per_class = []
    for i, cls in enumerate(label_encoder.classes_):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp+fp) > 0 else 0
        rec = tp / (tp + fn) if (tp+fn) > 0 else 0
        per_class.append((cls, prec, rec))
    per_class.sort(key=lambda x: x[1])
    print("\n5️⃣ WORST PERFORMING CLASSES (by precision):")
    print("-" * 50)
    for cls, prec, rec in per_class[:10]:
        print(f"   {cls:30s} precision={prec:.3f}, recall={rec:.3f}")

    return {
        'test_accuracy': float(test_acc),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'f1_weighted': float(f1_weighted),
        'top2_accuracy': float(top2_acc),
        'top3_accuracy': float(top3_acc),
        'top5_accuracy': float(top5_acc),
        'correct_predictions': int(np.trace(cm)),
        'wrong_predictions': int(np.sum(cm) - np.trace(cm))
    }

# ============================================================================
# 10. Main Training Pipeline
# ============================================================================

def main():
    print("="*70)
    print("🚀 ENHANCED BiLSTM TRAINING (Attention + Residual + Cosine Decay + Gate)")
    print("="*70)

    data = load_data()

    # Data augmentation
    X_symptoms_train, X_severities_train, y_train = data_augmentation(
        data['X_symptoms_train'],
        data['X_severities_train'],
        data['y_train'],
        aug_factor=AUGMENTATION_FACTOR
    )
    data['X_symptoms_train'] = X_symptoms_train
    data['X_severities_train'] = X_severities_train
    data['y_train'] = y_train

    # Class weights
    print("\n⚖️ Computing class weights...")
    class_weights = compute_class_weight('balanced', classes=np.unique(data['y_train']), y=data['y_train'])
    class_weight_dict = dict(enumerate(class_weights))

    # Build model
    print("\n🏗️ Building enhanced BiLSTM model...")
    model = build_enhanced_bilstm_model(data)
    model.summary()

    callbacks = setup_callbacks()

    print("\n🎯 Starting training...")
    print(f"   - Epochs: {EPOCHS}")
    print(f"   - Batch size: {BATCH_SIZE}")
    print(f"   - Label smoothing: {LABEL_SMOOTHING}")
    print(f"   - Gradient clipping: {GRADIENT_CLIP_NORM}")

    # Convert labels to one-hot
    y_train_one_hot = to_one_hot(data['y_train'], data['num_classes'])
    y_val_one_hot = to_one_hot(data['y_val'], data['num_classes'])

    history = model.fit(
        [data['X_symptoms_train'], data['X_severities_train']], y_train_one_hot,
        validation_data=([data['X_symptoms_val'], data['X_severities_val']], y_val_one_hot),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    # Load best weights
    print("\n📦 Loading best checkpoint weights...")
    if os.path.exists(BEST_WEIGHTS_PATH):
        model.load_weights(BEST_WEIGHTS_PATH)
        print(f"✅ Best weights loaded from: {BEST_WEIGHTS_PATH}")
    else:
        print(f"⚠️ Best weights file not found, using current model.")

    # Save models
    model.save(BEST_MODEL_PATH)
    model.save(FINAL_MODEL_PATH)
    print(f"✅ Best model saved to: {BEST_MODEL_PATH}")
    print(f"✅ Final model saved to: {FINAL_MODEL_PATH}")

    # Plot training history
    plot_training_history(history)

    # Evaluate
    metrics = evaluate_model(model, data)

    # Summary
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    gap = train_acc - val_acc

    print("\n" + "="*70)
    print("📈 TRAINING SUMMARY")
    print("="*70)
    print(f"\n   Train Accuracy: {train_acc*100:.4f}%")
    print(f"   Validation Accuracy: {val_acc*100:.4f}%")
    print(f"   Test Accuracy: {metrics['test_accuracy']*100:.4f}%")
    print(f"   Train-Val Gap: {gap*100:.4f}%")
    if abs(gap) < 0.03:
        print("   ✅ Excellent generalization (small gap)")
    elif gap > 0.1:
        print("   ⚠️ Possible overfitting detected")
    elif gap < -0.1:
        print("   ⚠️ Validation set might be easier than training")

    print(f"\n   F1-Score (Macro): {metrics['f1_macro']*100:.4f}%")
    print(f"   F1-Score (Weighted): {metrics['f1_weighted']*100:.4f}%")
    print(f"   Top-3 Accuracy: {metrics['top3_accuracy']*100:.4f}%")

    # Save full report
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_config": {
            "embedding_dim": EMBEDDING_DIM,
            "lstm_units_1": LSTM_UNITS_1,
            "lstm_units_2": LSTM_UNITS_2,
            "dropout_rate": DROPOUT_RATE,
            "recurrent_dropout": RECURRENT_DROPOUT,
            "l2_regularization": L2_REG,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "augmentation_factor": AUGMENTATION_FACTOR,
            "label_smoothing": LABEL_SMOOTHING,
            "gradient_clip_norm": GRADIENT_CLIP_NORM
        },
        "data_info": {
            "train_samples_before_aug": len(data['X_symptoms_train']) // 2,
            "train_samples_after_aug": len(data['X_symptoms_train']),
            "val_samples": len(data['X_symptoms_val']),
            "test_samples": len(data['X_symptoms_test']),
            "vocab_size": data['vocab_size'],
            "severity_vocab_size": data['severity_vocab_size'],
            "num_classes": data['num_classes'],
            "max_sequence_length": data['max_len']
        },
        "training_results": {
            "train_accuracy": float(train_acc),
            "val_accuracy": float(val_acc),
            "test_accuracy": metrics['test_accuracy'],
            "train_val_gap": float(gap),
            "epochs_completed": len(history.history['accuracy'])
        },
        "test_metrics": metrics
    }

    with open(os.path.join(REPORTS_DIR, "training_complete_report.json"), 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n📁 Complete report saved to: {os.path.join(REPORTS_DIR, 'training_complete_report.json')}")
    print("\n" + "="*70)
    print("✅ ENHANCED TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)

if __name__ == "__main__":
    main()