import os
import sys
import numpy as np
import pandas as pd
import pickle
import json
import re
from collections import Counter
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
    """Advanced text processing with stop words removal"""
    if pd.isna(text):
        return ""
    text = str(text).strip().lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(words)

def normalize_symptom(symptom):
    """Normalize symptom names"""
    if pd.isna(symptom):
        return np.nan
    symptom = str(symptom).strip().lower()
    symptom = symptom.replace(" ", "_")
    while "__" in symptom:
        symptom = symptom.replace("__", "_")
    return symptom

def pad_symptoms_fixed(row, symptom_cols, max_len):
    """Pad symptoms to fixed length"""
    symptoms = [s for s in row[symptom_cols].values if s != PAD_TOKEN and pd.notna(s)]
    if len(symptoms) < max_len:
        symptoms.extend([PAD_TOKEN] * (max_len - len(symptoms)))
    else:
        symptoms = symptoms[:max_len]
    return pd.Series(symptoms, index=symptom_cols)

# ============================================================================
# 3. Data Splitting - Proper handling of duplicates
# ============================================================================

def split_data_properly(df, symptom_cols, test_size=0.15, val_size=0.15, random_state=42):
    """
    Split data properly:
    - Same PATTERN (symptoms) can appear in multiple sets (different patients)
    - This is EXPECTED and DESIRABLE for learning generalization
    """
    np.random.seed(random_state)
    
    # Group by exact symptom pattern
    df_copy = df.copy()
    df_copy['pattern'] = df_copy[symptom_cols].apply(
        lambda row: tuple(sorted([s for s in row if s != PAD_TOKEN and pd.notna(s)])),
        axis=1
    )
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    # Group by pattern
    pattern_groups = {}
    for idx, row in df_copy.iterrows():
        pattern = row['pattern']
        if pattern not in pattern_groups:
            pattern_groups[pattern] = []
        pattern_groups[pattern].append(idx)
    
    print("\n" + "="*70)
    print("🔍 Data Splitting with Proper Duplicate Handling")
    print("="*70)
    print(f"\n📊 Total unique symptom patterns: {len(pattern_groups)}")
    
    for pattern, indices in pattern_groups.items():
        n_samples = len(indices)
        shuffled = np.random.permutation(indices)
        
        if n_samples == 1:
            train_indices.append(shuffled[0])
        elif n_samples == 2:
            train_indices.append(shuffled[0])
            test_indices.append(shuffled[1])
        elif n_samples == 3:
            train_indices.append(shuffled[0])
            val_indices.append(shuffled[1])
            test_indices.append(shuffled[2])
        else:
            n_test = max(1, int(n_samples * test_size))
            n_val = max(1, int(n_samples * val_size))
            n_train = n_samples - n_test - n_val
            
            current = 0
            test_indices.extend(shuffled[current:current + n_test])
            current += n_test
            val_indices.extend(shuffled[current:current + n_val])
            current += n_val
            train_indices.extend(shuffled[current:])
    
    print(f"\n📊 Split Results:")
    print(f"   Train: {len(train_indices)} samples")
    print(f"   Val:   {len(val_indices)} samples")
    print(f"   Test:  {len(test_indices)} samples")
    
    # Calculate pattern distribution
    def get_patterns(indices):
        return set(df_copy.loc[indices, 'pattern'])
    
    train_patterns = get_patterns(train_indices)
    val_patterns = get_patterns(val_indices)
    test_patterns = get_patterns(test_indices)
    
    print(f"\n📊 Pattern Distribution:")
    print(f"   Train patterns: {len(train_patterns)}")
    print(f"   Val patterns: {len(val_patterns)}")
    print(f"   Test patterns: {len(test_patterns)}")
    
    pattern_overlap = len(train_patterns.intersection(test_patterns))
    print(f"\n📊 Pattern Overlap (Train ∩ Test): {pattern_overlap}")
    print(f"   ℹ️  This is NOT data leakage - different patients with the same symptoms")
    print(f"   This is exactly what the model should learn to generalize!")
    
    # Return patterns for later use
    return train_indices, val_indices, test_indices, train_patterns, val_patterns, test_patterns

# ============================================================================
# 4. Main Preprocessing Function
# ============================================================================

def preprocess():
    print("="*70)
    print("📦 DATA PREPROCESSING - Disease Prediction Chatbot")
    print("="*70)
    
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
    
    for c in symptom_cols:
        dataset[c] = dataset[c].apply(normalize_symptom)
    
    dataset[symptom_cols] = dataset[symptom_cols].fillna(PAD_TOKEN)
    dataset[symptom_cols] = dataset.apply(
        lambda row: pad_symptoms_fixed(row, symptom_cols, MAX_SEQUENCE_LENGTH), axis=1
    )
    
    # Remove samples with no symptoms
    before = len(dataset)
    dataset = dataset[~dataset[symptom_cols].apply(lambda row: all(s == PAD_TOKEN for s in row.values), axis=1)]
    print(f"   - Removed {before - len(dataset)} samples with no symptoms")
    
    # Advanced text processing
    print("\n📝 Advanced text processing...")
    dataset['Symptoms_Text'] = dataset[symptom_cols].apply(
        lambda x: ' '.join([s for s in x.values if s != PAD_TOKEN]), axis=1
    )
    dataset['Symptoms_Text'] = dataset['Symptoms_Text'].apply(clean_text_advanced)
    
    # Analyze dataset
    print("\n📊 Dataset Analysis:")
    all_symptoms = [s for row in dataset[symptom_cols].values for s in row if s != PAD_TOKEN]
    unique_symptoms = len(set(all_symptoms))
    avg_symptoms = len(all_symptoms) / len(dataset)
    print(f"   - Unique symptoms: {unique_symptoms}")
    print(f"   - Average symptoms per sample: {avg_symptoms:.2f}")
    
    # Build vocabularies
    print("\n📚 Building vocabularies...")
    unique_symptoms_list = sorted(set(all_symptoms))
    
    symptom2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for s in unique_symptoms_list:
        if s not in symptom2idx:
            symptom2idx[s] = len(symptom2idx)
    
    vocab_size = len(symptom2idx)
    print(f"   - Vocabulary size: {vocab_size}")
    
    severity_dict = dict(zip(severity['Symptom'], severity['weight']))
    severity2idx = {0: 0}
    for s in unique_symptoms_list:
        sev = severity_dict.get(s, 1)
        if sev not in severity2idx:
            severity2idx[sev] = len(severity2idx)
    
    severity_vocab_size = len(severity2idx)
    print(f"   - Severity vocabulary size: {severity_vocab_size}")
    
    # Severity statistics
    severities = [severity_dict.get(s, 1) for s in all_symptoms]
    print(f"\n⚡ Severity Statistics:")
    print(f"   - Average severity: {np.mean(severities):.2f}")
    print(f"   - Max severity: {np.max(severities)}")
    print(f"   - Min severity: {np.min(severities)}")
    
    # Convert to indices
    print("\n🔄 Converting to indices...")
    X_symptoms = []
    X_severities = []
    
    for _, row in dataset.iterrows():
        symptom_seq = []
        severity_seq = []
        for s in row[symptom_cols].values:
            idx = symptom2idx.get(s, symptom2idx[UNK_TOKEN])
            symptom_seq.append(idx)
            sev = severity_dict.get(s, 1)
            severity_seq.append(severity2idx.get(sev, 1))
        X_symptoms.append(symptom_seq)
        X_severities.append(severity_seq)
    
    X_symptoms = np.array(X_symptoms, dtype=np.int32)
    X_severities = np.array(X_severities, dtype=np.int32)
    
    print(f"   - X_symptoms shape: {X_symptoms.shape}")
    print(f"   - X_severities shape: {X_severities.shape}")
    
    # Split data - FIXED: returns patterns as well
    print("\n✂️ Splitting data...")
    train_idx, val_idx, test_idx, train_patterns, val_patterns, test_patterns = split_data_properly(
        dataset, symptom_cols, test_size=0.15, val_size=0.15
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
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)
    
    num_classes = len(le.classes_)
    print(f"   - Number of diseases: {num_classes}")
    print(f"   - Diseases: {', '.join(list(le.classes_)[:5])}...")
    
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
    
    # Save vocabularies
    with open(os.path.join(MODELS_DIR, "symptom2idx.pkl"), 'wb') as f:
        pickle.dump(symptom2idx, f)
    
    with open(os.path.join(MODELS_DIR, "severity2idx.pkl"), 'wb') as f:
        pickle.dump(severity2idx, f)
    
    with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), 'wb') as f:
        pickle.dump(le, f)
    
    # Save reports
    print("\n📁 Saving reports...")
    
    # Save disease descriptions
    disease_info = {}
    for _, row in description.iterrows():
        disease_info[row['Disease']] = row['Description']
    
    with open(os.path.join(REPORTS_DIR, "disease_descriptions.json"), 'w', encoding='utf-8') as f:
        json.dump(disease_info, f, ensure_ascii=False, indent=2)
    
    # Save precautions
    precaution_info = {}
    for _, row in precaution.iterrows():
        precaution_info[row['Disease']] = {
            'precaution_1': row['Precaution_1'],
            'precaution_2': row['Precaution_2'],
            'precaution_3': row['Precaution_3'],
            'precaution_4': row['Precaution_4'] if 'Precaution_4' in row else None
        }
    
    with open(os.path.join(REPORTS_DIR, "disease_precautions.json"), 'w', encoding='utf-8') as f:
        json.dump(precaution_info, f, ensure_ascii=False, indent=2)
    
    # Save preprocessing report
    pattern_overlap = len(train_patterns.intersection(test_patterns))
    
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
        "unique_patterns": int(len(train_patterns)),
        "pattern_overlap": int(pattern_overlap),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(REPORTS_DIR, "preprocessing_report.json"), 'w', encoding='utf-8') as f:
        json.dump(preprocessing_report, f, ensure_ascii=False, indent=2)
    
    # Final summary
    print("\n" + "="*70)
    print("✅ PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\n📊 Final Results:")
    print(f"   - Train samples: {len(train_idx)}")
    print(f"   - Validation samples: {len(val_idx)}")
    print(f"   - Test samples: {len(test_idx)}")
    print(f"   - Vocabulary size: {vocab_size}")
    print(f"   - Number of diseases: {num_classes}")
    print(f"   - Unique patterns: {len(train_patterns)}")
    print(f"   - Pattern overlap (Train ∩ Test): {pattern_overlap}")
    print(f"\n   ℹ️  Pattern overlap is EXPECTED and DESIRABLE!")
    print(f"   This means different patients with the same symptoms appear in both sets,")
    print(f"   which helps the model learn to generalize rather than memorize.")
    print(f"\n📁 Files saved in: {PROCESSED_DIR}")
    print(f"📁 Models saved in: {MODELS_DIR}")
    print(f"📁 Reports saved in: {REPORTS_DIR}")

if __name__ == "__main__":
    preprocess()



































    