from collections import Counter
from src.config import PAD_TOKEN, UNK_TOKEN


def build_symptom_vocab(all_sequences: list[list[str]]) -> dict[str, int]:
    counter = Counter()
    for seq in all_sequences:
        counter.update(seq)

    vocab = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    for symptom in sorted(counter.keys()):
        if symptom not in vocab:
            vocab[symptom] = len(vocab)

    return vocab


def build_disease_mapping(labels: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    unique_labels = sorted(set(labels))
    disease_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_disease = {idx: label for label, idx in disease_to_id.items()}
    return disease_to_id, id_to_disease