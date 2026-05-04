from sklearn.model_selection import train_test_split
from src.config import RANDOM_STATE, TEST_SIZE, VAL_SIZE


def build_group_key(sequence: list[str]) -> str:
    return "|".join(sorted(set(sequence)))


def split_indices_grouped(group_keys: list[str]) -> tuple[list[int], list[int], list[int]]:
    unique_groups = sorted(set(group_keys))

    train_groups, test_groups = train_test_split(
        unique_groups,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_groups, val_groups = train_test_split(
        train_groups,
        test_size=VAL_SIZE / (1 - TEST_SIZE),
        random_state=RANDOM_STATE,
    )

    train_set = set(train_groups)
    val_set = set(val_groups)
    test_set = set(test_groups)

    train_idx, val_idx, test_idx = [], [], []

    for idx, key in enumerate(group_keys):
        if key in train_set:
            train_idx.append(idx)
        elif key in val_set:
            val_idx.append(idx)
        else:
            test_idx.append(idx)

    return train_idx, val_idx, test_idx