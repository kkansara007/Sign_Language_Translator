import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
import tensorflowjs as tfjs

# ====== Config ======
SEQ_LEN = 20                       # frames per sequence
NUM_FEATURES = 63                  # 21 landmarks × (x,y,z)
EPOCHS = 30
BATCH_SIZE = 32

# ====== Load dataset ======
with open("model_basics.json") as f:
    raw_data = json.load(f)

X, y = [], []
for sample in raw_data:
    landmarks = np.array(sample["landmarks"], dtype=float)

    # --- Normalize landmarks (optional scaling, helps generalize) ---
    # Assumes raw values ~ 0–1000 (pixels). Adjust divisor if needed.
    landmarks = landmarks / 1000.0

    seq = np.zeros((SEQ_LEN, NUM_FEATURES))

    # Handle static vs motion samples
    if landmarks.ndim == 1:  
        # static (single frame → shape (63,))
        seq[0] = landmarks
    else:
        # sequence (shape (frames, 63))
        L = min(len(landmarks), SEQ_LEN)
        seq[:L] = landmarks[:L]

    X.append(seq)
    y.append(sample["label"])

X = np.array(X, dtype=float)
y = np.array(y)

# ====== Encode labels ======
enc = LabelEncoder()
y_enc = enc.fit_transform(y)
print("Labels:", enc.classes_)

# Save label list for frontend (important!)
with open("model_labels.json", "w") as f:
    json.dump(enc.classes_.tolist(), f)
print("Saved labels to model_labels.json")

# ====== Train/test split ======
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# ====== Build model ======
model = keras.Sequential([
    keras.layers.Input(shape=(SEQ_LEN, NUM_FEATURES)),

    # Stacked LSTMs for better temporal learning
    keras.layers.LSTM(128, return_sequences=True),
    keras.layers.LSTM(64),

    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(len(enc.classes_), activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# ====== Train ======
print("Training the model...")
checkpoint = keras.callbacks.ModelCheckpoint(
    "best_model.h5", save_best_only=True, monitor="val_accuracy", mode="max"
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint]
)

# ====== Evaluate ======
loss, acc = model.evaluate(X_test, y_test)
print(f"Final Test accuracy: {acc * 100:.2f}%")

# ====== Save and Convert ======
print("Saving final model to asl_model.h5...")
model.save("asl_model.h5")

print("Converting best model to TensorFlow.js format...")
best_model = keras.models.load_model("best_model.h5")
tfjs.converters.save_keras_model(best_model, "./model_tfjs")
print("✅ Conversion complete. TF.js model saved in 'model_tfjs' folder.")
