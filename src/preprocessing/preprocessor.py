# ============================================================================
# COMPLETE PREPROCESSING CODE
# STRICT DATA LEAKAGE FIX:
# SPLIT BY UNIQUE ORDER-INDEPENDENT SYMPTOM-SEVERITY PATTERNS
# ============================================================================

import os
import numpy as np
import pandas as pd
import pickle
import json
import re
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

# ============================================================================
# 1. Set Paths
# ============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(BASE_DIR, "artifacts", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "artifacts", "models")
REPORTS_DIR = os.path.join(BASE_DIR, "artifacts", "reports")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ============================================================================
# 2. Text Processing Functions
# ============================================================================

STOP_WORDS = set([
    "i", "have", "and", "my", "with", "feeling", "am", "the", "a", "of", "in",
    "for", "on", "at", "to", "is", "was", "are", "were", "been", "has", "had"
])

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def clean_text_advanced(text):
    """Advanced text processing with stop words removal."""
    if pd.isna(text):
        return ""

    text = str(text).strip().lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(words)


def normalize_symptom(symptom):
    """Normalize symptom and disease names."""
    if pd.isna(symptom):
        return np.nan

    symptom = str(symptom).strip().lower()
    symptom = symptom.replace(" ", "_")

    while "__" in symptom:
        symptom = symptom.replace("__", "_")

    return symptom


def pad_symptoms_fixed(row, symptom_cols, max_len):
    """Pad symptoms to fixed length."""
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
    """
    Shuffle symptom order for each row to reduce positional bias.

    Note:
    This does not affect strict splitting because the split later uses
    order-independent symptom-severity canonical patterns.
    """
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
# 3. Strict Pattern Creation and Data Splitting
# ============================================================================

def create_canonical_symptom_severity_pattern(row, symptom_cols, severity_dict):
    """
    Create an order-independent symptom-severity pattern.

    This function:
    - Ignores PAD tokens.
    - Links every symptom with its severity value.
    - Sorts the symptom-severity pairs.
    - Returns a tuple that is independent of symptom order.

    Example:
    [fever, cough] with severities [5, 4]
    becomes:
    (("cough", 4), ("fever", 5))

    This prevents the same symptom-severity combination from appearing
    across train, validation, and test even if symptom order differs.
    """
    pairs = []

    for symptom in row[symptom_cols].values:
        if pd.notna(symptom) and symptom != PAD_TOKEN:
            sev = severity_dict.get(symptom, 1)
            pairs.append((symptom, int(sev)))

    return tuple(sorted(pairs))


def split_data_properly(
    df,
    symptom_cols,
    severity_dict,
    test_size=0.15,
    val_size=0.15,
    random_state=42
):
    """
    Strict leakage-aware split.

    Split data by UNIQUE ORDER-INDEPENDENT SYMPTOM-SEVERITY PATTERNS.

    Each unique symptom-severity pattern is assigned entirely to one subset:
    train, validation, or test.

    This is stricter than:
    - random row-level splitting
    - order-dependent sequence splitting

    It reduces leakage caused by repeated symptom-severity combinations.
    """
    np.random.seed(random_state)

    df_copy = df.copy()

    # Create strict canonical pattern
    df_copy["pattern"] = df_copy.apply(
        lambda row: create_canonical_symptom_severity_pattern(
            row,
            symptom_cols,
            severity_dict
        ),
        axis=1
    )

    unique_patterns = df_copy["pattern"].unique()
    np.random.shuffle(unique_patterns)

    n_patterns = len(unique_patterns)
    n_test = max(1, int(n_patterns * test_size))
    n_val = max(1, int(n_patterns * val_size))
    n_train = n_patterns - n_test - n_val

    train_patterns = set(unique_patterns[:n_train])
    val_patterns = set(unique_patterns[n_train:n_train + n_val])
    test_patterns = set(unique_patterns[n_train + n_val:])

    train_indices = df_copy[df_copy["pattern"].isin(train_patterns)].index.tolist()
    val_indices = df_copy[df_copy["pattern"].isin(val_patterns)].index.tolist()
    test_indices = df_copy[df_copy["pattern"].isin(test_patterns)].index.tolist()

    train_val_overlap = len(train_patterns & val_patterns)
    train_test_overlap = len(train_patterns & test_patterns)
    val_test_overlap = len(val_patterns & test_patterns)

    print("\n" + "=" * 70)
    print("🔍 STRICT Data Splitting by Unique Symptom-Severity Patterns")
    print("=" * 70)
    print(f"\n📊 Total unique symptom-severity patterns: {n_patterns}")

    print("\n📊 Split Results:")
    print(f"   Train: {len(train_indices)} samples, {len(train_patterns)} patterns")
    print(f"   Val:   {len(val_indices)} samples, {len(val_patterns)} patterns")
    print(f"   Test:  {len(test_indices)} samples, {len(test_patterns)} patterns")

    print("\n📊 Pattern Overlap Checks:")
    print(f"   Train ∩ Val:  {train_val_overlap}")
    print(f"   Train ∩ Test: {train_test_overlap}")
    print(f"   Val ∩ Test:   {val_test_overlap}")

    if train_test_overlap == 0 and train_val_overlap == 0 and val_test_overlap == 0:
        print("\n✅ Strict pattern overlap check passed.")
        print("✅ No identical symptom-severity pattern appears across splits.")
    else:
        print("\n⚠️ Warning: pattern overlap still detected.")

    return train_indices, val_indices, test_indices, train_patterns, val_patterns, test_patterns


# ============================================================================
# 4. Main Preprocessing Function
# ============================================================================

def preprocess():
    print("=" * 70)
    print("📦 DATA PREPROCESSING - Disease Prediction Chatbot")
    print("🔒 STRICT LEAKAGE-AWARE SPLIT ENABLED")
    print("=" * 70)

    # Check required files exist
    required_files = [
        "dataset.csv",
        "Symptom-severity.csv",
        "symptom_Description.csv",
        "symptom_precaution.csv"
    ]

    for file_name in required_files:
        file_path = os.path.join(DATA_DIR, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Missing required file: {file_name} in {DATA_DIR}"
            )

    # Load data
    print("\n📂 Loading data...")
    dataset = pd.read_csv(os.path.join(DATA_DIR, "dataset.csv"))
    severity = pd.read_csv(os.path.join(DATA_DIR, "Symptom-severity.csv"))
    description = pd.read_csv(os.path.join(DATA_DIR, "symptom_Description.csv"))
    precaution = pd.read_csv(os.path.join(DATA_DIR, "symptom_precaution.csv"))

    print(f"   - dataset: {len(dataset)} samples")
    print(f"   - severity: {len(severity)} symptoms")
    print(f"   - description: {len(description)} diseases")
    print(f"   - precaution: {len(precaution)} diseases")

    # Clean data
    print("\n🧹 Cleaning data...")
    dataset["Disease"] = dataset["Disease"].apply(normalize_symptom)

    symptom_cols = [c for c in dataset.columns if c.startswith("Symptom_")]
    MAX_SEQUENCE_LENGTH = len(symptom_cols)

    print(f"   - Number of symptom columns: {MAX_SEQUENCE_LENGTH}")

    for col in symptom_cols:
        dataset[col] = dataset[col].apply(normalize_symptom)

    # Normalize severity symptom names too
    severity["Symptom"] = severity["Symptom"].apply(normalize_symptom)

    # Fill missing symptoms
    dataset[symptom_cols] = dataset[symptom_cols].fillna(PAD_TOKEN)

    # Ensure fixed symptom length
    dataset[symptom_cols] = dataset.apply(
        lambda row: pad_symptoms_fixed(row, symptom_cols, MAX_SEQUENCE_LENGTH),
        axis=1
    )

    # Shuffle symptom order to reduce positional bias
    print("\n🔀 Shuffling symptom order randomly to reduce positional bias...")
    dataset = shuffle_symptoms_randomly(dataset, symptom_cols, random_state=42)

    # Remove samples with no valid symptoms
    before = len(dataset)
    dataset = dataset[
        ~dataset[symptom_cols].apply(
            lambda row: all(s == PAD_TOKEN for s in row.values),
            axis=1
        )
    ]

    # Reset index to keep dataframe indices aligned with numpy arrays
    dataset = dataset.reset_index(drop=True)

    print(f"   - Removed {before - len(dataset)} samples with no symptoms")

    # Advanced text processing
    print("\n📝 Advanced text processing...")
    dataset["Symptoms_Text"] = dataset[symptom_cols].apply(
        lambda row: " ".join([s for s in row.values if s != PAD_TOKEN]),
        axis=1
    )
    dataset["Symptoms_Text"] = dataset["Symptoms_Text"].apply(clean_text_advanced)

    # Analyze dataset
    print("\n📊 Dataset Analysis:")
    all_symptoms = [
        symptom
        for row in dataset[symptom_cols].values
        for symptom in row
        if symptom != PAD_TOKEN
    ]

    unique_symptoms = len(set(all_symptoms))
    avg_symptoms = len(all_symptoms) / len(dataset)

    print(f"   - Unique symptoms: {unique_symptoms}")
    print(f"   - Average symptoms per sample: {avg_symptoms:.2f}")

    # Build vocabularies
    print("\n📚 Building vocabularies...")
    unique_symptoms_list = sorted(set(all_symptoms))

    symptom2idx = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1
    }

    for symptom in unique_symptoms_list:
        if symptom not in symptom2idx:
            symptom2idx[symptom] = len(symptom2idx)

    vocab_size = len(symptom2idx)
    print(f"   - Symptom vocabulary size: {vocab_size}")

    # Severity mapping
    severity_dict = dict(zip(severity["Symptom"], severity["weight"]))

    severity2idx = {0: 0}

    for symptom in unique_symptoms_list:
        sev = int(severity_dict.get(symptom, 1))
        if sev not in severity2idx:
            severity2idx[sev] = len(severity2idx)

    severity_vocab_size = len(severity2idx)
    print(f"   - Severity vocabulary size: {severity_vocab_size}")

    # Severity statistics
    severities = [
        int(severity_dict.get(symptom, 1))
        for symptom in all_symptoms
    ]

    print("\n⚡ Severity Statistics:")
    print(f"   - Average severity: {np.mean(severities):.2f}")
    print(f"   - Max severity: {np.max(severities)}")
    print(f"   - Min severity: {np.min(severities)}")

    # Convert symptoms and severities to indices
    print("\n🔄 Converting symptoms and severities to indices...")
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

    print(f"   - X_symptoms shape: {X_symptoms.shape}")
    print(f"   - X_severities shape: {X_severities.shape}")

    # Strict split by order-independent symptom-severity patterns
    print("\n✂️ Splitting data with strict leakage-aware strategy...")
    train_idx, val_idx, test_idx, train_patterns, val_patterns, test_patterns = split_data_properly(
        dataset,
        symptom_cols,
        severity_dict,
        test_size=0.15,
        val_size=0.15,
        random_state=42
    )

    X_symptoms_train = X_symptoms[train_idx]
    X_symptoms_val = X_symptoms[val_idx]
    X_symptoms_test = X_symptoms[test_idx]

    X_severities_train = X_severities[train_idx]
    X_severities_val = X_severities[val_idx]
    X_severities_test = X_severities[test_idx]

    y_train = dataset.loc[train_idx, "Disease"].values
    y_val = dataset.loc[val_idx, "Disease"].values
    y_test = dataset.loc[test_idx, "Disease"].values

    # Encode labels
    print("\n🏷️ Encoding labels...")
    label_encoder = LabelEncoder()

    y_train_enc = label_encoder.fit_transform(y_train)
    y_val_enc = label_encoder.transform(y_val)
    y_test_enc = label_encoder.transform(y_test)

    num_classes = len(label_encoder.classes_)

    print(f"   - Number of diseases: {num_classes}")
    print(f"   - First diseases: {', '.join(list(label_encoder.classes_)[:5])}...")

    # Save processed data
    print("\n💾 Saving processed data...")

    np.save(os.path.join(PROCESSED_DIR, "X_symptoms_train.npy"), X_symptoms_train)
    np.save(os.path.join(PROCESSED_DIR, "X_symptoms_val.npy"), X_symptoms_val)
    np.save(os.path.join(PROCESSED_DIR, "X_symptoms_test.npy"), X_symptoms_test)

    np.save(os.path.join(PROCESSED_DIR, "X_severities_train.npy"), X_severities_train)
    np.save(os.path.join(PROCESSED_DIR, "X_severities_val.npy"), X_severities_val)
    np.save(os.path.join(PROCESSED_DIR, "X_severities_test.npy"), X_severities_test)

    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train_enc)
    np.save(os.path.join(PROCESSED_DIR, "y_val.npy"), y_val_enc)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test_enc)

    # Save vocabularies and label encoder
    with open(os.path.join(MODELS_DIR, "symptom2idx.pkl"), "wb") as f:
        pickle.dump(symptom2idx, f)

    with open(os.path.join(MODELS_DIR, "severity2idx.pkl"), "wb") as f:
        pickle.dump(severity2idx, f)

    with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)

    # Save reports and knowledge base outputs
    print("\n📁 Saving reports and knowledge base files...")

    # Disease descriptions
    disease_info = {}

    for _, row in description.iterrows():
        disease_name = normalize_symptom(row["Disease"])
        disease_info[disease_name] = row["Description"]

    with open(
        os.path.join(REPORTS_DIR, "disease_descriptions.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(disease_info, f, ensure_ascii=False, indent=2)

    # Disease precautions
    precaution_info = {}

    for _, row in precaution.iterrows():
        disease_name = normalize_symptom(row["Disease"])

        precaution_info[disease_name] = {
            "precaution_1": row.get("Precaution_1", None),
            "precaution_2": row.get("Precaution_2", None),
            "precaution_3": row.get("Precaution_3", None),
            "precaution_4": row.get("Precaution_4", None)
        }

    with open(
        os.path.join(REPORTS_DIR, "disease_precautions.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(precaution_info, f, ensure_ascii=False, indent=2)

    # Overlap checks for report
    train_val_overlap = len(train_patterns & val_patterns)
    train_test_overlap = len(train_patterns & test_patterns)
    val_test_overlap = len(val_patterns & test_patterns)

    preprocessing_report = {
        "total_samples": int(len(dataset)),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "test_samples": int(len(test_idx)),
        "num_classes": int(num_classes),
        "vocab_size": int(vocab_size),
        "severity_vocab_size": int(severity_vocab_size),
        "max_sequence_length": int(MAX_SEQUENCE_LENGTH),
        "unique_symptoms": int(unique_symptoms),
        "avg_symptoms_per_sample": float(avg_symptoms),
        "avg_severity": float(np.mean(severities)),
        "unique_symptom_severity_patterns_total": int(
            len(train_patterns) + len(val_patterns) + len(test_patterns)
        ),
        "train_unique_patterns": int(len(train_patterns)),
        "val_unique_patterns": int(len(val_patterns)),
        "test_unique_patterns": int(len(test_patterns)),
        "pattern_overlap_train_val": int(train_val_overlap),
        "pattern_overlap_train_test": int(train_test_overlap),
        "pattern_overlap_val_test": int(val_test_overlap),
        "symptom_order_shuffled": True,
        "data_leakage_reduction": True,
        "strict_order_independent_split": True,
        "split_method": "unique_order_independent_symptom_severity_patterns",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(
        os.path.join(REPORTS_DIR, "preprocessing_report.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(preprocessing_report, f, ensure_ascii=False, indent=2)

    # Final summary
    print("\n" + "=" * 70)
    print("✅ PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 70)

    print("\n📊 Final Results:")
    print(f"   - Train samples: {len(train_idx)}")
    print(f"   - Validation samples: {len(val_idx)}")
    print(f"   - Test samples: {len(test_idx)}")
    print(f"   - Symptom vocabulary size: {vocab_size}")
    print(f"   - Severity vocabulary size: {severity_vocab_size}")
    print(f"   - Number of diseases: {num_classes}")
    print(f"   - Train unique patterns: {len(train_patterns)}")
    print(f"   - Validation unique patterns: {len(val_patterns)}")
    print(f"   - Test unique patterns: {len(test_patterns)}")

    print("\n📊 Strict Pattern Overlap:")
    print(f"   - Train ∩ Validation: {train_val_overlap}")
    print(f"   - Train ∩ Test: {train_test_overlap}")
    print(f"   - Validation ∩ Test: {val_test_overlap}")

    print("\n✅ Symptom order was shuffled to reduce positional bias.")
    print("✅ Strict split used symptom-severity canonical patterns.")
    print("✅ Same unordered symptom-severity pattern cannot appear in more than one split.")

    print(f"\n📁 Processed files saved in: {PROCESSED_DIR}")
    print(f"📁 Model helper files saved in: {MODELS_DIR}")
    print(f"📁 Reports saved in: {REPORTS_DIR}")


if __name__ == "__main__":
    preprocess()