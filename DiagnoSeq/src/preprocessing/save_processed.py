import numpy as np
from pathlib import Path


def save_split_arrays(base_dir: Path, X_train, X_val, X_test, y_train, y_val, y_test) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)

    np.save(base_dir / "X_train.npy", X_train)
    np.save(base_dir / "X_val.npy", X_val)
    np.save(base_dir / "X_test.npy", X_test)

    np.save(base_dir / "y_train.npy", y_train)
    np.save(base_dir / "y_val.npy", y_val)
    np.save(base_dir / "y_test.npy", y_test)