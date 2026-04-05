import os
import sys
import json
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
    BatchNormalization, Input, Concatenate
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import AdamW

# ============================================================================
# 1. Configuration
# ============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, "artifacts", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "artifacts", "models")
REPORTS_DIR = os.path.join(BASE_DIR, "artifacts", "reports")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Model hyperparameters
EMBEDDING_DIM = 128
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
DROPOUT_RATE = 0.3
RECURRENT_DROPOUT = 0.2
L2_REG = 1e-4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
EPOCHS = 150
PATIENCE = 20
REDUCE_LR_PATIENCE = 8
MIN_LR = 1e-6

# Data augmentation
AUGMENTATION_FACTOR = 0.5  # 50% increase

# Checkpoint paths
BEST_WEIGHTS_PATH = os.path.join(MODELS_DIR, "bilstm_best.weights.h5")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "bilstm_best.keras")
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, "bilstm_final.keras")

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# 2. Data Loading Functions
# ============================================================================

def load_data():
    """Load processed data from artifacts"""
    print("\n📂 Loading preprocessed data...")

    # Load symptom sequences
    X_symptoms_train = np.load(os.path.join(PROCESSED_DIR, "X_symptoms_train.npy"))
    X_symptoms_val = np.load(os.path.join(PROCESSED_DIR, "X_symptoms_val.npy"))
    X_symptoms_test = np.load(os.path.join(PROCESSED_DIR, "X_symptoms_test.npy"))

    # Load severity sequences
    X_severities_train = np.load(os.path.join(PROCESSED_DIR, "X_severities_train.npy"))
    X_severities_val = np.load(os.path.join(PROCESSED_DIR, "X_severities_val.npy"))
    X_severities_test = np.load(os.path.join(PROCESSED_DIR, "X_severities_test.npy"))

    # Load labels
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    y_val = np.load(os.path.join(PROCESSED_DIR, "y_val.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

    # Load vocabularies
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
# 3. Data Augmentation (from Tuqa's approach)
# ============================================================================

def data_augmentation(X_symptoms, X_severities, y, aug_factor=0.5):
    """
    Enhanced data augmentation - creates variations of existing samples
    """
    n_samples = len(X_symptoms)
    n_augmented = int(n_samples * aug_factor)

    print(f"\n🔄 Performing data augmentation...")
    print(f"   - Original samples: {n_samples}")
    print(f"   - Augmented samples: {n_augmented}")

    X_symptoms_aug = list(X_symptoms)
    X_severities_aug = list(X_severities)
    y_aug = list(y)

    vocab_size = np.max(X_symptoms) + 1
    severity_vocab_size = np.max(X_severities) + 1

    for _ in range(n_augmented):
        idx = np.random.randint(0, n_samples)
        symptom_seq = X_symptoms[idx].copy()
        severity_seq = X_severities[idx].copy()

        symptom_positions = np.where(symptom_seq != 0)[0]

        if len(symptom_positions) > 0:
            n_modify = np.random.randint(1, min(4, len(symptom_positions)))
            modify_idx = np.random.choice(symptom_positions, n_modify, replace=False)

            new_symptoms = np.random.randint(1, vocab_size, size=n_modify)
            symptom_seq[modify_idx] = new_symptoms

            for pos in modify_idx:
                current_sev = severity_seq[pos]
                if current_sev > 1:
                    delta = np.random.choice([-1, 0, 1])
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
# 4. Build Optimized BiLSTM Model
# ============================================================================

def build_optimized_bilstm_model(data):
    vocab_size = data['vocab_size']
    severity_vocab_size = data['severity_vocab_size']
    max_len = data['max_len']
    num_classes = data['num_classes']

    symptom_input = Input(shape=(max_len,), name='symptom_input')
    symptom_embedding = Embedding(
        input_dim=vocab_size,
        output_dim=EMBEDDING_DIM,
        input_length=max_len,
        mask_zero=True,
        embeddings_regularizer=regularizers.l2(L2_REG),
        name='symptom_embedding'
    )(symptom_input)

    severity_input = Input(shape=(max_len,), name='severity_input')
    severity_embedding = Embedding(
        input_dim=severity_vocab_size,
        output_dim=16,
        input_length=max_len,
        mask_zero=True,
        embeddings_regularizer=regularizers.l2(L2_REG),
        name='severity_embedding'
    )(severity_input)

    combined = Concatenate(name='concat_layer')([symptom_embedding, severity_embedding])

    lstm1 = Bidirectional(
        LSTM(
            LSTM_UNITS_1,
            dropout=DROPOUT_RATE,
            recurrent_dropout=RECURRENT_DROPOUT,
            return_sequences=True,
            kernel_regularizer=regularizers.l2(L2_REG)
        ),
        name='bilstm_1'
    )(combined)
    lstm1 = Dropout(DROPOUT_RATE, name='dropout_1')(lstm1)

    lstm2 = Bidirectional(
        LSTM(
            LSTM_UNITS_2,
            dropout=DROPOUT_RATE,
            recurrent_dropout=RECURRENT_DROPOUT,
            return_sequences=False,
            kernel_regularizer=regularizers.l2(L2_REG)
        ),
        name='bilstm_2'
    )(lstm1)
    lstm2 = Dropout(DROPOUT_RATE, name='dropout_2')(lstm2)

    bn = BatchNormalization(name='batch_norm')(lstm2)

    dense1 = Dense(
        64,
        activation='relu',
        kernel_regularizer=regularizers.l2(L2_REG),
        name='dense_1'
    )(bn)
    dense1 = Dropout(DROPOUT_RATE + 0.1, name='dropout_3')(dense1)

    dense2 = Dense(
        32,
        activation='relu',
        kernel_regularizer=regularizers.l2(L2_REG),
        name='dense_2'
    )(dense1)
    dense2 = Dropout(DROPOUT_RATE, name='dropout_4')(dense2)

    output = Dense(num_classes, activation='softmax', name='output')(dense2)

    model = Model(inputs=[symptom_input, severity_input], outputs=output)

    optimizer = AdamW(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# ============================================================================
# 5. Callbacks Setup
# ============================================================================

def setup_callbacks():
    """Setup training callbacks"""

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
# 6. Visualization Functions
# ============================================================================

def plot_training_history(history):
    """Plot training curves"""

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
        axes[1, 1].text(
            2, max(gap) / 2 if len(gap) > 0 else 0,
            f'Avg Gap: {avg_gap:.4f}',
            bbox=dict(boxstyle="round", facecolor="wheat")
        )

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "training_curves.png"), dpi=100, bbox_inches='tight')
    plt.close()
    print(f"\n📈 Training curves saved to: {os.path.join(REPORTS_DIR, 'training_curves.png')}")

# ============================================================================
# 7. Evaluation Functions
# ============================================================================

def evaluate_model(model, data):
    """Comprehensive model evaluation"""

    print("\n" + "="*70)
    print("📊 COMPREHENSIVE MODEL EVALUATION")
    print("="*70)

    X_test_symptoms = data['X_symptoms_test']
    X_test_severities = data['X_severities_test']
    y_test = data['y_test']
    label_encoder = data['label_encoder']

    y_pred_prob = model.predict([X_test_symptoms, X_test_severities], verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    print("\n1️⃣ BASIC METRICS:")
    print("-" * 40)

    test_acc = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

    precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"   Test Accuracy: {test_acc*100:.4f}%")
    print(f"   Precision (Macro): {precision_macro:.6f}")
    print(f"   Recall (Macro): {recall_macro:.6f}")
    print(f"   F1-Score (Macro): {f1_macro:.6f}")
    print(f"   Precision (Weighted): {precision_weighted:.6f}")
    print(f"   Recall (Weighted): {recall_weighted:.6f}")
    print(f"   F1-Score (Weighted): {f1_weighted:.6f}")

    print("\n2️⃣ TOP-K ACCURACY:")
    print("-" * 40)

    top2_acc = top_k_accuracy_score(y_test, y_pred_prob, k=2, labels=range(len(label_encoder.classes_)))
    top3_acc = top_k_accuracy_score(y_test, y_pred_prob, k=3, labels=range(len(label_encoder.classes_)))
    top5_acc = top_k_accuracy_score(y_test, y_pred_prob, k=5, labels=range(len(label_encoder.classes_)))

    print(f"   Top-2 Accuracy: {top2_acc*100:.4f}%")
    print(f"   Top-3 Accuracy: {top3_acc*100:.4f}%")
    print(f"   Top-5 Accuracy: {top5_acc*100:.4f}%")

    print("\n3️⃣ CLASSIFICATION REPORT:")
    print("-" * 70)

    report_dict = classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report_dict).transpose()
    print(report_df.round(6).head(20))
    print(f"\n   ... and {len(label_encoder.classes_) - 20} more classes")

    report_df.to_csv(os.path.join(REPORTS_DIR, "classification_report.csv"))
    print(f"\n✅ Full report saved to: {os.path.join(REPORTS_DIR, 'classification_report.csv')}")

    print("\n4️⃣ CONFUSION MATRIX:")
    print("-" * 40)

    cm = confusion_matrix(y_test, y_pred)
    diagonal_sum = np.trace(cm)
    off_diagonal_sum = np.sum(cm) - diagonal_sum

    print(f"   Total samples: {np.sum(cm)}")
    print(f"   Correct predictions: {diagonal_sum}")
    print(f"   Wrong predictions: {off_diagonal_sum}")

    plt.figure(figsize=(20, 16))

    if len(label_encoder.classes_) <= 30:
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_
        )
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
    else:
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')

    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"), dpi=100, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved to: {os.path.join(REPORTS_DIR, 'confusion_matrix.png')}")

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
        'correct_predictions': int(diagonal_sum),
        'wrong_predictions': int(off_diagonal_sum)
    }

# ============================================================================
# 8. Main Training Function
# ============================================================================

def main():
    """Main training pipeline"""

    print("="*70)
    print("🚀 OPTIMIZED BiLSTM TRAINING - Merged from Shaden, Tuqa, Ghida")
    print("="*70)
    print("\n📋 Features Integrated:")
    print("   - From Shaden: EarlyStopping, clean architecture")
    print("   - From Tuqa: Data augmentation, class weights, L2 regularization, AdamW")
    print("   - From Ghida: Bidirectional LSTM, two-layer architecture, dropout")
    print("   - Project: Severity levels, zero data leakage")

    data = load_data()

    X_symptoms_train, X_severities_train, y_train = data_augmentation(
        data['X_symptoms_train'],
        data['X_severities_train'],
        data['y_train'],
        aug_factor=AUGMENTATION_FACTOR
    )

    data['X_symptoms_train'] = X_symptoms_train
    data['X_severities_train'] = X_severities_train
    data['y_train'] = y_train

    print("\n⚖️ Computing class weights...")
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(data['y_train']),
        y=data['y_train']
    )
    class_weight_dict = dict(enumerate(class_weights))

    print("\n🏗️ Building optimized BiLSTM model...")
    model = build_optimized_bilstm_model(data)
    model.summary()

    callbacks = setup_callbacks()

    print("\n🎯 Starting training...")
    print(f"   - Epochs: {EPOCHS}")
    print(f"   - Batch size: {BATCH_SIZE}")
    print(f"   - Patience: {PATIENCE}")

    history = model.fit(
        [data['X_symptoms_train'], data['X_severities_train']],
        data['y_train'],
        validation_data=([data['X_symptoms_val'], data['X_severities_val']], data['y_val']),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    print("\n📦 Loading best checkpoint weights...")
    if os.path.exists(BEST_WEIGHTS_PATH):
        model.load_weights(BEST_WEIGHTS_PATH)
        print(f"✅ Best weights loaded from: {BEST_WEIGHTS_PATH}")
    else:
        print(f"⚠️ Best weights file not found: {BEST_WEIGHTS_PATH}")
        print("   Continuing with current in-memory model weights.")

    print("\n💾 Saving best model...")
    model.save(BEST_MODEL_PATH)
    print(f"✅ Best model saved to: {BEST_MODEL_PATH}")

    print("\n💾 Saving final model...")
    model.save(FINAL_MODEL_PATH)
    print(f"✅ Final model saved to: {FINAL_MODEL_PATH}")

    plot_training_history(history)

    metrics = evaluate_model(model, data)

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
            "augmentation_factor": AUGMENTATION_FACTOR
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
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\n📊 Final Results:")
    print(f"   - Test Accuracy: {metrics['test_accuracy']*100:.4f}%")
    print(f"   - F1-Score (Macro): {metrics['f1_macro']*100:.4f}%")
    print(f"   - F1-Score (Weighted): {metrics['f1_weighted']*100:.4f}%")
    print(f"   - Top-3 Accuracy: {metrics['top3_accuracy']*100:.4f}%")
    print(f"\n📁 Best model saved in: {BEST_MODEL_PATH}")
    print(f"📁 Final model saved in: {FINAL_MODEL_PATH}")
    print(f"📁 Reports saved in: {REPORTS_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()