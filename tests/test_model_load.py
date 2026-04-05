from pathlib import Path
import tensorflow as tf

model_path = Path("artifacts/models/bilstm_final.keras")
print("Path:", model_path.resolve())
print("Exists:", model_path.exists())

if model_path.exists():
    print("Size:", model_path.stat().st_size, "bytes")

model = tf.keras.models.load_model(model_path, compile=False)
print("Model loaded successfully")
print(model.summary())











