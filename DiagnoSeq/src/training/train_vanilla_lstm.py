import json
import numpy as np

from src.paths import PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, ENCODERS_DIR
from src.utils import load_pickle
from src.models.vanilla_lstm_model import build_vanilla_lstm_model
from src.training.callbacks import get_callbacks
from src.config import NUM_EPOCHS, BATCH_SIZE


def train_vanilla_lstm():
    X_train = np.load(PROCESSED_DIR / "vanilla_lstm" / "X_train.npy")
    X_val = np.load(PROCESSED_DIR / "vanilla_lstm" / "X_val.npy")
    y_train = np.load(PROCESSED_DIR / "vanilla_lstm" / "y_train.npy")
    y_val = np.load(PROCESSED_DIR / "vanilla_lstm" / "y_val.npy")

    metadata = load_pickle(ENCODERS_DIR / "metadata.pkl")
    vocab_size = metadata["vocab_size"]
    num_classes = metadata["num_classes"]

    model = build_vanilla_lstm_model(vocab_size=vocab_size, num_classes=num_classes)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks(str(MODELS_DIR / "vanilla_lstm_best.keras")),
        verbose=1,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "vanilla_lstm_history.json", "w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=2)

    return model