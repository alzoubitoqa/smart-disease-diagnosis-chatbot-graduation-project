import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.paths import PROCESSED_DIR, MODELS_DIR, REPORTS_DIR


def evaluate_single_model(model_path, X_test, y_test, model_name):
    model = load_model(model_path)
    probs = model.predict(X_test, verbose=0)
    preds = np.argmax(probs, axis=1)

    acc = accuracy_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="macro", zero_division=0
    )

    return {
        "model": model_name,
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }


def compare_all_models():
    rows = []

    X_test = np.load(PROCESSED_DIR / "rnn" / "X_test.npy")
    y_test = np.load(PROCESSED_DIR / "rnn" / "y_test.npy")
    rows.append(
        evaluate_single_model(
            MODELS_DIR / "rnn_best.keras",
            X_test,
            y_test,
            "Simple RNN"
        )
    )

    X_test = np.load(PROCESSED_DIR / "vanilla_lstm" / "X_test.npy")
    y_test = np.load(PROCESSED_DIR / "vanilla_lstm" / "y_test.npy")
    rows.append(
        evaluate_single_model(
            MODELS_DIR / "vanilla_lstm_best.keras",
            X_test,
            y_test,
            "Vanilla LSTM"
        )
    )

    X_test = np.load(PROCESSED_DIR / "deep_lstm" / "X_test.npy")
    y_test = np.load(PROCESSED_DIR / "deep_lstm" / "y_test.npy")
    rows.append(
        evaluate_single_model(
            MODELS_DIR / "deep_lstm_best.keras",
            X_test,
            y_test,
            "Deep LSTM"
        )
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS_DIR / "final_comparison.csv", index=False)
    print(df)
    return df