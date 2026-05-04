from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, BatchNormalization
from tensorflow.keras.optimizers import Adam
from src.config import LEARNING_RATE


def build_deep_lstm_model(num_classes: int, max_len: int = 17):
    model = Sequential([
        Masking(mask_value=0.0, input_shape=(max_len, 2)),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model