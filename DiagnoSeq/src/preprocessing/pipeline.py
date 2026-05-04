import numpy as np
import pandas as pd

from src.paths import ensure_directories, INTERIM_DIR, PROCESSED_DIR, ENCODERS_DIR
from src.utils import save_json, save_pickle
from src.config import TARGET_COLUMN
from src.preprocessing.load_data import (
    load_main_dataset,
    load_description_dataset,
    load_precaution_dataset,
    load_severity_dataset,
)
from src.preprocessing.clean_symptoms import clean_dataset, extract_symptom_sequence
from src.preprocessing.build_vocab import build_symptom_vocab, build_disease_mapping
from src.preprocessing.severity_mapper import build_severity_map
from src.preprocessing.sequence_builder import encode_sequence, encode_sequence_with_severity
from src.preprocessing.grouped_split import build_group_key, split_indices_grouped
from src.preprocessing.save_processed import save_split_arrays
from src.preprocessing.knowledge_base_builder import build_description_map, build_precaution_map


def run_preprocessing_pipeline() -> None:
    ensure_directories()

    main_df = load_main_dataset()
    desc_df = load_description_dataset()
    precaution_df = load_precaution_dataset()
    severity_df = load_severity_dataset()

    main_df = clean_dataset(main_df)
    main_df.to_csv(INTERIM_DIR / "cleaned_dataset.csv", index=False)

    sequences = [extract_symptom_sequence(row) for _, row in main_df.iterrows()]
    labels = main_df[TARGET_COLUMN].tolist()

    vocab = build_symptom_vocab(sequences)
    disease_to_id, id_to_disease = build_disease_mapping(labels)
    severity_map = build_severity_map(severity_df)

    save_json(vocab, INTERIM_DIR / "symptom_vocab.json")
    save_json(disease_to_id, INTERIM_DIR / "disease_to_id.json")
    save_json(id_to_disease, INTERIM_DIR / "id_to_disease.json")
    save_json(severity_map, INTERIM_DIR / "severity_map.json")

    build_description_map(desc_df)
    build_precaution_map(precaution_df)

    group_keys = [build_group_key(seq) for seq in sequences]
    train_idx, val_idx, test_idx = split_indices_grouped(group_keys)

    X_basic = np.array([encode_sequence(seq, vocab) for seq in sequences], dtype=np.int32)
    X_severity = np.array(
        [encode_sequence_with_severity(seq, vocab, severity_map) for seq in sequences],
        dtype=np.int32,
    )
    y = np.array([disease_to_id[label] for label in labels], dtype=np.int32)

    def select_split(X, y, indices):
        return X[indices], y[indices]

    # RNN
    X_train, y_train = select_split(X_basic, y, train_idx)
    X_val, y_val = select_split(X_basic, y, val_idx)
    X_test, y_test = select_split(X_basic, y, test_idx)
    save_split_arrays(PROCESSED_DIR / "rnn", X_train, X_val, X_test, y_train, y_val, y_test)

    # Vanilla LSTM
    X_train, y_train = select_split(X_basic, y, train_idx)
    X_val, y_val = select_split(X_basic, y, val_idx)
    X_test, y_test = select_split(X_basic, y, test_idx)
    save_split_arrays(PROCESSED_DIR / "vanilla_lstm", X_train, X_val, X_test, y_train, y_val, y_test)

    # Deep LSTM
    X_train, y_train = select_split(X_severity, y, train_idx)
    X_val, y_val = select_split(X_severity, y, val_idx)
    X_test, y_test = select_split(X_severity, y, test_idx)
    save_split_arrays(PROCESSED_DIR / "deep_lstm", X_train, X_val, X_test, y_train, y_val, y_test)

    metadata = {
        "num_classes": len(disease_to_id),
        "vocab_size": len(vocab),
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_idx),
    }
    save_pickle(metadata, ENCODERS_DIR / "metadata.pkl")