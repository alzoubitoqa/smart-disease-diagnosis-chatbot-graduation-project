# ============================================================================
# 4-STAGE GROUPKFOLD CROSS VALIDATION
# STRICT ORDER-INDEPENDENT SYMPTOM-SEVERITY GROUPS
# Medical Diagnosis Assistant
# ============================================================================

import os
import json
import pickle
import math
import re
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    BatchNormalization,
    Add,
    Multiply,
    Layer
)
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ============================================================================
# 1. Paths
# ============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
REPORTS_DIR = os.path.join(ARTIFACTS_DIR, "reports")
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")

GROUP_KFOLD_DIR = os.path.join(REPORTS_DIR, "group_kfold_4stage")

os.makedirs(GROUP_KFOLD_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ============================================================================
# 2. Configuration
# ============================================================================

RANDOM_STATE = 42
N_SPLITS = 4

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

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
LABEL_SMOOTHING = 0.1
GRADIENT_CLIP_NORM = 1.0

AUGMENTATION_FACTOR = 0.5

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================================
# 3. Text Processing
# ============================================================================

STOP_WORDS = set([
    "i", "have", "and", "my", "with", "feeling", "am", "the", "a", "of", "in",
    "for", "on", "at", "to", "is", "was", "are", "were", "been", "has", "had"
])


def clean_text_advanced(text):
    if pd.isna(text):
        return ""

    text = str(text).strip().lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(words)


def normalize_symptom(symptom):
    if pd.isna(symptom):
        return np.nan

    symptom = str(symptom).strip().lower()
    symptom = symptom.replace(" ", "_")

    while "__" in symptom:
        symptom = symptom.replace("__", "_")

    return symptom


def pad_symptoms_fixed(row, symptom_cols, max_len):
    symptoms = [
        s for s in row[symptom_cols].values
        if s != PAD_TOKEN and pd.notna(s)
    ]

    if len(symptoms) < max_len:
        symptoms.extend([PAD_TOKEN] * (max_len - len(symptoms)))
    else:
        symptoms = symptoms[:max_len]

    return pd.Series(symptoms, index=symptom_cols)


def shuffle_symptoms_randomly(df, symptom_cols, random_state=42):
    np.random.seed(random_state)
    df_shuffled = df.copy()

    for idx in df_shuffled.index:
        symptoms = df_shuffled.loc[idx, symptom_cols].values

        non_pad = [
            s for s in symptoms
            if s != PAD_TOKEN and pd.notna(s)
        ]

        if len(non_pad) > 1:
            np.random.shuffle(non_pad)

            new_symptoms = []
            i = 0

            for s in symptoms:
                if s != PAD_TOKEN and pd.notna(s):
                    new_symptoms.append(non_pad[i])
                    i += 1
                else:
                    new_symptoms.append(s)

            df_shuffled.loc[idx, symptom_cols] = new_symptoms

    return df_shuffled


# ============================================================================
# 4. Strict Canonical Pattern
# ============================================================================

def create_canonical_symptom_severity_pattern(row, symptom_cols, severity_dict):
    """
    Creates an order-independent symptom-severity pattern.

    Example:
    fever(5), cough(4), fatigue(3)

    becomes:
    ((cough, 4), (fatigue, 3), (fever, 5))

    Therefore:
    fever(5), cough(4)
    and
    cough(4), fever(5)

    are treated as the same group.
    """
    pairs = []

    for symptom in row[symptom_cols].values:
        if pd.notna(symptom) and symptom != PAD_TOKEN:
            sev = int(severity_dict.get(symptom, 1))
            pairs.append((symptom, sev))

    return tuple(sorted(pairs))


# ============================================================================
# 5. Custom Layers
# ============================================================================

class Attention(Layer):
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
            1.0 + tf.cos(math.pi * (step - warmup_steps) / denominator)
        )

        lr = tf.where(step < warmup_steps, warmup, cosine)
        return self.learning_rate_base * lr

    def get_config(self):
        return {
            "learning_rate_base": self.learning_rate_base,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps
        }


# ============================================================================
# 6. Build Enhanced BiLSTM Model
# ============================================================================

def build_enhanced_bilstm_model(
    vocab_size,
    severity_vocab_size,
    max_len,
    num_classes,
    train_size
):
    symptom_input = Input(shape=(max_len,), name="symptom_input")
    severity_input = Input(shape=(max_len,), name="severity_input")

    symptom_emb = Embedding(
        input_dim=vocab_size,
        output_dim=EMBEDDING_DIM,
        input_length=max_len,
        mask_zero=True,
        embeddings_regularizer=regularizers.l2(L2_REG),
        name="symptom_embedding"
    )(symptom_input)

    severity_emb = Embedding(
        input_dim=severity_vocab_size,
        output_dim=16,
        input_length=max_len,
        mask_zero=True,
        embeddings_regularizer=regularizers.l2(L2_REG),
        name="severity_embedding"
    )(severity_input)

    gate = Dense(
        EMBEDDING_DIM,
        activation="sigmoid",
        name="severity_gate"
    )(severity_emb)

    combined = Multiply(name="gated_symptoms")([symptom_emb, gate])

    lstm1 = Bidirectional(
        LSTM(
            LSTM_UNITS_1,
            dropout=DROPOUT_RATE,
            recurrent_dropout=RECURRENT_DROPOUT,
            return_sequences=True,
            kernel_regularizer=regularizers.l2(L2_REG)
        ),
        name="bilstm_1"
    )(combined)

    lstm1 = Dropout(DROPOUT_RATE, name="dropout_1")(lstm1)

    shortcut = Dense(lstm1.shape[-1], name="shortcut")(combined)
    lstm1 = Add(name="residual_1")([lstm1, shortcut])

    lstm2 = Bidirectional(
        LSTM(
            LSTM_UNITS_2,
            dropout=DROPOUT_RATE,
            recurrent_dropout=RECURRENT_DROPOUT,
            return_sequences=True,
            kernel_regularizer=regularizers.l2(L2_REG)
        ),
        name="bilstm_2"
    )(lstm1)

    lstm2 = Dropout(DROPOUT_RATE, name="dropout_2")(lstm2)

    attention_output = Attention(name="attention")(lstm2)
    bn = BatchNormalization(name="batch_norm")(attention_output)

    dense1 = Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(L2_REG),
        name="dense_1"
    )(bn)

    dense1 = Dropout(DROPOUT_RATE + 0.1, name="dropout_3")(dense1)

    dense2 = Dense(
        64,
        activation="relu",
        kernel_regularizer=regularizers.l2(L2_REG),
        name="dense_2"
    )(dense1)

    dense2 = Dropout(DROPOUT_RATE, name="dropout_4")(dense2)

    output = Dense(
        num_classes,
        activation="softmax",
        name="output"
    )(dense2)

    model = Model(
        inputs=[symptom_input, severity_input],
        outputs=output
    )

    total_steps = max(1, (train_size // BATCH_SIZE) * EPOCHS)
    warmup_steps = max(1, int(0.1 * total_steps))

    lr_schedule = CosineWarmup(
        LEARNING_RATE,
        warmup_steps,
        total_steps
    )

    optimizer = AdamW(
        learning_rate=lr_schedule,
        weight_decay=WEIGHT_DECAY,
        clipnorm=GRADIENT_CLIP_NORM
    )

    loss = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=LABEL_SMOOTHING
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"]
    )

    return model


# ============================================================================
# 7. Data Augmentation
# ============================================================================

def compute_symptom_distribution(X_symptoms, vocab_size):
    all_symptoms = X_symptoms.flatten()
    all_symptoms = all_symptoms[all_symptoms != 0]

    counts = np.bincount(all_symptoms, minlength=vocab_size)

    if counts.sum() == 0:
        probs = np.ones(vocab_size, dtype=np.float32) / vocab_size
    else:
        probs = counts.astype(np.float32) / counts.sum()

    probs[0] = 0.0
    probs = probs / probs.sum()

    return probs


def data_augmentation(X_symptoms, X_severities, y, aug_factor=0.5):
    n_samples = len(X_symptoms)
    n_augmented = int(n_samples * aug_factor)

    vocab_size = int(np.max(X_symptoms)) + 1
    severity_vocab_size = int(np.max(X_severities)) + 1
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
            upper_bound = min(4, len(symptom_positions) + 1)
            n_modify = np.random.randint(1, upper_bound)

            modify_idx = np.random.choice(
                symptom_positions,
                n_modify,
                replace=False
            )

            new_symptoms = np.random.choice(
                vocab_size,
                size=n_modify,
                p=symptom_probs
            )

            symptom_seq[modify_idx] = new_symptoms

            for pos in modify_idx:
                current_sev = severity_seq[pos]

                if current_sev > 1:
                    delta = np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2])
                    new_sev = max(
                        1,
                        min(severity_vocab_size - 1, current_sev + delta)
                    )
                    severity_seq[pos] = new_sev

        X_symptoms_aug.append(symptom_seq)
        X_severities_aug.append(severity_seq)
        y_aug.append(y[idx])

    return (
        np.array(X_symptoms_aug, dtype=np.int32),
        np.array(X_severities_aug, dtype=np.int32),
        np.array(y_aug, dtype=np.int32)
    )


# ============================================================================
# 8. Load and Prepare Data
# ============================================================================

def load_and_prepare_data():
    print("=" * 70)
    print("📦 Loading and preparing data for 4-Stage GroupKFold")
    print("=" * 70)

    dataset = pd.read_csv(os.path.join(DATA_DIR, "dataset.csv"))
    severity = pd.read_csv(os.path.join(DATA_DIR, "Symptom-severity.csv"))

    dataset["Disease"] = dataset["Disease"].apply(normalize_symptom)

    symptom_cols = [col for col in dataset.columns if col.startswith("Symptom_")]
    max_len = len(symptom_cols)

    for col in symptom_cols:
        dataset[col] = dataset[col].apply(normalize_symptom)

    severity["Symptom"] = severity["Symptom"].apply(normalize_symptom)

    dataset[symptom_cols] = dataset[symptom_cols].fillna(PAD_TOKEN)

    dataset[symptom_cols] = dataset.apply(
        lambda row: pad_symptoms_fixed(row, symptom_cols, max_len),
        axis=1
    )

    dataset = shuffle_symptoms_randomly(
        dataset,
        symptom_cols,
        random_state=RANDOM_STATE
    )

    dataset = dataset[
        ~dataset[symptom_cols].apply(
            lambda row: all(s == PAD_TOKEN for s in row.values),
            axis=1
        )
    ].reset_index(drop=True)

    dataset["Symptoms_Text"] = dataset[symptom_cols].apply(
        lambda row: " ".join([s for s in row.values if s != PAD_TOKEN]),
        axis=1
    )

    dataset["Symptoms_Text"] = dataset["Symptoms_Text"].apply(clean_text_advanced)

    all_symptoms = [
        symptom
        for row in dataset[symptom_cols].values
        for symptom in row
        if symptom != PAD_TOKEN
    ]

    unique_symptoms_list = sorted(set(all_symptoms))

    symptom2idx = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1
    }

    for symptom in unique_symptoms_list:
        if symptom not in symptom2idx:
            symptom2idx[symptom] = len(symptom2idx)

    severity_dict = dict(zip(severity["Symptom"], severity["weight"]))

    severity2idx = {
        0: 0
    }

    for symptom in unique_symptoms_list:
        sev = int(severity_dict.get(symptom, 1))

        if sev not in severity2idx:
            severity2idx[sev] = len(severity2idx)

    X_symptoms = []
    X_severities = []

    for _, row in dataset.iterrows():
        symptom_seq = []
        severity_seq = []

        for symptom in row[symptom_cols].values:
            symptom_idx = symptom2idx.get(symptom, symptom2idx[UNK_TOKEN])
            symptom_seq.append(symptom_idx)

            if symptom == PAD_TOKEN:
                sev = 0
            else:
                sev = int(severity_dict.get(symptom, 1))

            severity_idx = severity2idx.get(sev, 1)
            severity_seq.append(severity_idx)

        X_symptoms.append(symptom_seq)
        X_severities.append(severity_seq)

    X_symptoms = np.array(X_symptoms, dtype=np.int32)
    X_severities = np.array(X_severities, dtype=np.int32)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(dataset["Disease"].values)

    patterns = dataset.apply(
        lambda row: create_canonical_symptom_severity_pattern(
            row,
            symptom_cols,
            severity_dict
        ),
        axis=1
    )

    unique_pattern_list = sorted(set(patterns), key=str)
    pattern_to_group = {
        pattern: idx
        for idx, pattern in enumerate(unique_pattern_list)
    }

    groups = np.array([pattern_to_group[p] for p in patterns], dtype=np.int32)

    print("\n📊 Prepared Data:")
    print(f"   Total samples: {len(dataset)}")
    print(f"   Disease classes: {len(label_encoder.classes_)}")
    print(f"   Unique symptoms: {len(unique_symptoms_list)}")
    print(f"   Symptom vocab size: {len(symptom2idx)}")
    print(f"   Severity vocab size: {len(severity2idx)}")
    print(f"   Max sequence length: {max_len}")
    print(f"   Unique strict symptom-severity groups: {len(np.unique(groups))}")

    return {
        "dataset": dataset,
        "X_symptoms": X_symptoms,
        "X_severities": X_severities,
        "y": y,
        "groups": groups,
        "label_encoder": label_encoder,
        "symptom2idx": symptom2idx,
        "severity2idx": severity2idx,
        "max_len": max_len,
        "num_classes": len(label_encoder.classes_),
        "vocab_size": len(symptom2idx),
        "severity_vocab_size": len(severity2idx),
    }


# ============================================================================
# 9. Run 4-Stage GroupKFold
# ============================================================================

def run_group_kfold():
    data = load_and_prepare_data()

    X_symptoms = data["X_symptoms"]
    X_severities = data["X_severities"]
    y = data["y"]
    groups = data["groups"]

    num_classes = data["num_classes"]
    all_labels = np.arange(num_classes)

    gkf = GroupKFold(n_splits=N_SPLITS)

    fold_results = []

    print("\n" + "=" * 70)
    print("🚀 Starting 4-Stage GroupKFold Cross Validation")
    print("=" * 70)

    for fold_number, (train_val_idx, test_idx) in enumerate(
        gkf.split(X_symptoms, y, groups),
        start=1
    ):
        print("\n" + "=" * 70)
        print(f"🔁 STAGE {fold_number}/{N_SPLITS}")
        print("=" * 70)

        train_val_groups = groups[train_val_idx]

        group_split = GroupShuffleSplit(
            n_splits=1,
            test_size=0.15,
            random_state=RANDOM_STATE + fold_number
        )

        train_inner_relative, val_relative = next(
            group_split.split(
                X_symptoms[train_val_idx],
                y[train_val_idx],
                groups=train_val_groups
            )
        )

        train_idx = train_val_idx[train_inner_relative]
        val_idx = train_val_idx[val_relative]

        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        test_groups = set(groups[test_idx])

        train_val_overlap = len(train_groups & val_groups)
        train_test_overlap = len(train_groups & test_groups)
        val_test_overlap = len(val_groups & test_groups)

        print("\n📊 Group Overlap Check:")
        print(f"   Train ∩ Validation: {train_val_overlap}")
        print(f"   Train ∩ Test: {train_test_overlap}")
        print(f"   Validation ∩ Test: {val_test_overlap}")

        if train_val_overlap != 0 or train_test_overlap != 0 or val_test_overlap != 0:
            raise ValueError("Group leakage detected. Stop training.")

        X_sym_train = X_symptoms[train_idx]
        X_sev_train = X_severities[train_idx]
        y_train = y[train_idx]

        X_sym_val = X_symptoms[val_idx]
        X_sev_val = X_severities[val_idx]
        y_val = y[val_idx]

        X_sym_test = X_symptoms[test_idx]
        X_sev_test = X_severities[test_idx]
        y_test = y[test_idx]

        print("\n📊 Stage Split:")
        print(f"   Train samples: {len(train_idx)}")
        print(f"   Validation samples: {len(val_idx)}")
        print(f"   Test samples: {len(test_idx)}")
        print(f"   Train groups: {len(train_groups)}")
        print(f"   Validation groups: {len(val_groups)}")
        print(f"   Test groups: {len(test_groups)}")
        print(f"   Test classes present: {len(np.unique(y_test))}/{num_classes}")

        print("\n🔄 Applying training-only augmentation...")
        X_sym_train_aug, X_sev_train_aug, y_train_aug = data_augmentation(
            X_sym_train,
            X_sev_train,
            y_train,
            aug_factor=AUGMENTATION_FACTOR
        )

        print(f"   Train before augmentation: {len(X_sym_train)}")
        print(f"   Train after augmentation: {len(X_sym_train_aug)}")

        y_train_cat = tf.keras.utils.to_categorical(y_train_aug, num_classes)
        y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes)

        unique_train_classes = np.unique(y_train_aug)

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=unique_train_classes,
            y=y_train_aug
        )

        class_weight_dict = {
            int(cls): float(weight)
            for cls, weight in zip(unique_train_classes, class_weights)
        }

        tf.keras.backend.clear_session()

        model = build_enhanced_bilstm_model(
            vocab_size=data["vocab_size"],
            severity_vocab_size=data["severity_vocab_size"],
            max_len=data["max_len"],
            num_classes=num_classes,
            train_size=len(X_sym_train_aug)
        )

        fold_model_path = os.path.join(
            GROUP_KFOLD_DIR,
            f"group_kfold_stage_{fold_number}_best.weights.h5"
        )

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                fold_model_path,
                monitor="val_accuracy",
                save_best_only=True,
                save_weights_only=True,
                mode="max",
                verbose=1
            )
        ]

        history = model.fit(
            [X_sym_train_aug, X_sev_train_aug],
            y_train_cat,
            validation_data=(
                [X_sym_val, X_sev_val],
                y_val_cat
            ),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )

        if os.path.exists(fold_model_path):
            model.load_weights(fold_model_path)

        y_pred_probs = model.predict(
            [X_sym_test, X_sev_test],
            verbose=0
        )

        y_pred = np.argmax(y_pred_probs, axis=1)

        test_accuracy = accuracy_score(y_test, y_pred)

        precision_macro = precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        recall_macro = recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        f1_macro = f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        precision_weighted = precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        recall_weighted = recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        f1_weighted = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        top2_accuracy = top_k_accuracy_score(
            y_test,
            y_pred_probs,
            k=2,
            labels=all_labels
        )

        top3_accuracy = top_k_accuracy_score(
            y_test,
            y_pred_probs,
            k=3,
            labels=all_labels
        )

        top5_accuracy = top_k_accuracy_score(
            y_test,
            y_pred_probs,
            k=5,
            labels=all_labels
        )

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=all_labels
        )

        correct_predictions = int(np.trace(cm))
        wrong_predictions = int(np.sum(cm) - np.trace(cm))

        print("\n📊 Stage Results:")
        print(f"   Test Accuracy: {test_accuracy * 100:.2f}%")
        print(f"   Macro Precision: {precision_macro:.4f}")
        print(f"   Macro Recall: {recall_macro:.4f}")
        print(f"   Macro F1: {f1_macro:.4f}")
        print(f"   Weighted F1: {f1_weighted:.4f}")
        print(f"   Top-2 Accuracy: {top2_accuracy * 100:.2f}%")
        print(f"   Top-3 Accuracy: {top3_accuracy * 100:.2f}%")
        print(f"   Top-5 Accuracy: {top5_accuracy * 100:.2f}%")
        print(f"   Correct Predictions: {correct_predictions}")
        print(f"   Wrong Predictions: {wrong_predictions}")

        report = classification_report(
            y_test,
            y_pred,
            labels=all_labels,
            target_names=data["label_encoder"].classes_,
            output_dict=True,
            zero_division=0
        )

        report_path = os.path.join(
            GROUP_KFOLD_DIR,
            f"group_kfold_stage_{fold_number}_classification_report.csv"
        )

        pd.DataFrame(report).transpose().to_csv(report_path)

        fold_result = {
            "stage": int(fold_number),
            "train_samples": int(len(train_idx)),
            "validation_samples": int(len(val_idx)),
            "test_samples": int(len(test_idx)),
            "train_samples_after_augmentation": int(len(X_sym_train_aug)),
            "train_groups": int(len(train_groups)),
            "validation_groups": int(len(val_groups)),
            "test_groups": int(len(test_groups)),
            "train_validation_group_overlap": int(train_val_overlap),
            "train_test_group_overlap": int(train_test_overlap),
            "validation_test_group_overlap": int(val_test_overlap),
            "test_classes_present": int(len(np.unique(y_test))),
            "total_classes": int(num_classes),
            "epochs_completed": int(len(history.history["accuracy"])),
            "best_validation_accuracy": float(max(history.history["val_accuracy"])),
            "final_validation_accuracy": float(history.history["val_accuracy"][-1]),
            "test_accuracy": float(test_accuracy),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_macro": float(f1_macro),
            "precision_weighted": float(precision_weighted),
            "recall_weighted": float(recall_weighted),
            "f1_weighted": float(f1_weighted),
            "top2_accuracy": float(top2_accuracy),
            "top3_accuracy": float(top3_accuracy),
            "top5_accuracy": float(top5_accuracy),
            "correct_predictions": int(correct_predictions),
            "wrong_predictions": int(wrong_predictions)
        }

        fold_results.append(fold_result)

        fold_json_path = os.path.join(
            GROUP_KFOLD_DIR,
            f"group_kfold_stage_{fold_number}_results.json"
        )

        with open(fold_json_path, "w", encoding="utf-8") as f:
            json.dump(fold_result, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Stage {fold_number} report saved to:")
        print(f"   {fold_json_path}")

    # ========================================================================
    # Final Cross-Validation Summary
    # ========================================================================

    accuracy_values = [r["test_accuracy"] for r in fold_results]
    precision_values = [r["precision_macro"] for r in fold_results]
    recall_values = [r["recall_macro"] for r in fold_results]
    f1_values = [r["f1_macro"] for r in fold_results]
    top3_values = [r["top3_accuracy"] for r in fold_results]

    total_correct = sum(r["correct_predictions"] for r in fold_results)
    total_wrong = sum(r["wrong_predictions"] for r in fold_results)
    total_test_samples = sum(r["test_samples"] for r in fold_results)

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "validation_type": "4-stage GroupKFold cross-validation",
        "grouping_strategy": "canonical order-independent symptom-severity patterns",
        "n_splits": int(N_SPLITS),
        "total_samples": int(len(X_symptoms)),
        "total_classes": int(num_classes),
        "total_unique_groups": int(len(np.unique(groups))),
        "stages": fold_results,
        "mean_test_accuracy": float(np.mean(accuracy_values)),
        "std_test_accuracy": float(np.std(accuracy_values)),
        "mean_macro_precision": float(np.mean(precision_values)),
        "std_macro_precision": float(np.std(precision_values)),
        "mean_macro_recall": float(np.mean(recall_values)),
        "std_macro_recall": float(np.std(recall_values)),
        "mean_macro_f1": float(np.mean(f1_values)),
        "std_macro_f1": float(np.std(f1_values)),
        "mean_top3_accuracy": float(np.mean(top3_values)),
        "std_top3_accuracy": float(np.std(top3_values)),
        "total_test_samples_across_stages": int(total_test_samples),
        "total_correct_predictions_across_stages": int(total_correct),
        "total_wrong_predictions_across_stages": int(total_wrong)
    }

    summary_path = os.path.join(
        GROUP_KFOLD_DIR,
        "group_kfold_4stage_summary.json"
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    summary_csv_path = os.path.join(
        GROUP_KFOLD_DIR,
        "group_kfold_4stage_summary.csv"
    )

    pd.DataFrame(fold_results).to_csv(summary_csv_path, index=False)

    print("\n" + "=" * 70)
    print("🏁 4-STAGE GROUPKFOLD CROSS VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Mean Test Accuracy: {np.mean(accuracy_values) * 100:.2f}%")
    print(f"Std Test Accuracy: {np.std(accuracy_values) * 100:.4f}%")
    print(f"Mean Macro Precision: {np.mean(precision_values):.4f}")
    print(f"Mean Macro Recall: {np.mean(recall_values):.4f}")
    print(f"Mean Macro F1: {np.mean(f1_values):.4f}")
    print(f"Mean Top-3 Accuracy: {np.mean(top3_values) * 100:.2f}%")
    print(f"Total Test Samples Across Stages: {total_test_samples}")
    print(f"Total Correct Predictions Across Stages: {total_correct}")
    print(f"Total Wrong Predictions Across Stages: {total_wrong}")
    print("\n📁 Reports saved in:")
    print(f"   {GROUP_KFOLD_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    run_group_kfold()