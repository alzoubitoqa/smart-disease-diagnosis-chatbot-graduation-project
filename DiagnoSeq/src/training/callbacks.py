import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, Callback


class BestKerasModelSaver(Callback):
    """
    Save the full model in native .keras format whenever the monitored metric improves.
    This avoids compatibility issues that sometimes appear with ModelCheckpoint + .keras.
    """

    def __init__(self, filepath: str, monitor: str = "val_loss", mode: str = "min", verbose: int = 1):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose

        if self.mode not in {"min", "max"}:
            raise ValueError("mode must be either 'min' or 'max'")

        self.best = np.inf if self.mode == "min" else -np.inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)

        if current is None:
            if self.verbose:
                print(f"\nWarning: Metric '{self.monitor}' not found in logs. Model was not saved.")
            return

        improved = (current < self.best) if self.mode == "min" else (current > self.best)

        if improved:
            old_best = self.best
            self.best = current

            if self.verbose:
                if np.isfinite(old_best):
                    print(
                        f"\nEpoch {epoch + 1}: {self.monitor} improved from "
                        f"{old_best:.6f} to {current:.6f}. Saving model to {self.filepath}"
                    )
                else:
                    print(
                        f"\nEpoch {epoch + 1}: {self.monitor} = {current:.6f}. "
                        f"Saving initial best model to {self.filepath}"
                    )

            self.model.save(self.filepath, overwrite=True)


def get_callbacks(model_path: str):
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        BestKerasModelSaver(
            filepath=model_path,
            monitor="val_loss",
            mode="min",
            verbose=1
        )
    ]