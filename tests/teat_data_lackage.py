import numpy as np
X_train = np.load("artifacts/processed/X_symptoms_train.npy")
X_val = np.load("artifacts/processed/X_symptoms_val.npy")
X_test = np.load("artifacts/processed/X_symptoms_test.npy")

train_rows = set(tuple(row) for row in X_train)
val_rows = set(tuple(row) for row in X_val)
test_rows = set(tuple(row) for row in X_test)

print(f"تداخل Train/Val: {len(train_rows.intersection(val_rows))}")
print(f"تداخل Train/Test: {len(train_rows.intersection(test_rows))}")