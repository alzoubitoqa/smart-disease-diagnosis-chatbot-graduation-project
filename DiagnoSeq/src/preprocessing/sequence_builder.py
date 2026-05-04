import numpy as np
from src.config import MAX_LEN


def encode_sequence(
    sequence: list[str],
    vocab: dict[str, int],
    max_len: int = MAX_LEN,
) -> np.ndarray:
    unk_id = vocab.get("<UNK>", 1)
    encoded = [vocab.get(token, unk_id) for token in sequence]
    encoded = encoded[:max_len]

    if len(encoded) < max_len:
        encoded += [0] * (max_len - len(encoded))

    return np.array(encoded, dtype=np.int32)


def encode_sequence_with_severity(
    sequence: list[str],
    vocab: dict[str, int],
    severity_map: dict[str, int],
    max_len: int = MAX_LEN,
) -> np.ndarray:
    encoded = []
    unk_id = vocab.get("<UNK>", 1)

    for token in sequence[:max_len]:
        token_id = vocab.get(token, unk_id)
        severity = severity_map.get(token, 1)
        encoded.append([token_id, severity])

    while len(encoded) < max_len:
        encoded.append([0, 0])

    return np.array(encoded, dtype=np.int32)