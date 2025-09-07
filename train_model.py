import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
import tensorflowjs as tfjs

# The sequence length (number of frames) for each sample
SEQ_LEN = 20
# The number of features per frame (21 landmarks * (x, y, z coordinates))
NUM_FEATURES = 63

# ===== Load dataset =====
with open("model_basics.json") as f:
    raw_data = json.load(f)

X, y = [], []
for sample in raw_data:
    landmarks = sample["landmarks"]

    # If it's static (1D) → pad into sequence
    if isinstance(landmarks[0], (int, float)):
        seq = np.zeros((SEQ_LEN, NUM_FEATURES))
        seq[0] = landmarks  # place static sign in first frame
    else:
        seq = np.zeros((SEQ_LEN, NUM_FEATURES))
        L = min(len(landmarks), SEQ_LEN)
        seq[:L] = landmarks[:L]  # truncate/pad sequence

    X.append(seq)
    y.append(sample["label"])

X = np.array(X)
y = np.array(y)

# Encode labels
enc = LabelEncoder()
y_enc = enc.fit_transform(y)
print("Labels:", enc.classes_)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)

# ===== Build LSTM Model =====
model = keras.Sequential([
    keras.layers.Input(shape=(SEQ_LEN, NUM_FEATURES)),
    keras.layers.LSTM(128, return_sequences=False),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(len(enc.classes_), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# ===== Train =====
print("Training the model...")
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=32)

# ===== Evaluate =====
loss, acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {acc * 100:.2f}%")

# ===== Save and Convert =====
print("Saving model to asl_model.h5...")
model.save("asl_model.h5")

# You will need to install tensorflowjs first:
# pip install tensorflowjs
print("Converting model to TensorFlow.js format...")
tfjs.converters.save_keras_model(model, "./model_tfjs")
print("Conversion complete. Your TF.js model is in the 'model_tfjs' directory.")
