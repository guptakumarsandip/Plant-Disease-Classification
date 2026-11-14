
from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight

from preprocess_dataset import create_datasets, NUM_CLASSES


# --------------------------------------------------
# 1. PROJECT PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models"
OUTPUT_PATH = ROOT / "outputs" / "training"
SPLIT_PATH = ROOT / "outputs" / "splits"

MODEL_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_PATH / "mobilenet_best.keras"
FINAL_MODEL_PATH = MODEL_PATH / "mobilenet_final.keras"

EPOCHS = 10
LEARNING_RATE = 0.0001
RANDOM_SEED = 42

tf.keras.utils.set_random_seed(RANDOM_SEED)


# --------------------------------------------------
# 2. LOAD CLASS MAPPING
# --------------------------------------------------

with open(
    SPLIT_PATH / "class_mapping.json",
    "r",
    encoding="utf-8"
) as file:
    class_mapping = json.load(file)

assert len(class_mapping) == NUM_CLASSES == 23

print("=" * 60)
print("MOBILENETV2 TRANSFER LEARNING")
print("=" * 60)

print(f"Number of classes: {NUM_CLASSES}")


# --------------------------------------------------
# 3. CALCULATE CLASS WEIGHTS
# --------------------------------------------------

print("\nCalculating class weights...")

train_df = pd.read_csv(SPLIT_PATH / "train.csv")

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=train_df["label"].astype(int).values
)

class_weights = {
    int(index): float(weight)
    for index, weight in enumerate(class_weights_array)
}

print("Class weights calculated successfully.")


# --------------------------------------------------
# 4. LOAD PREPROCESSED DATASETS
# --------------------------------------------------

print("\nLoading MobileNetV2 datasets...")

train_ds, val_ds, test_ds = create_datasets("mobilenet")

print("\nDatasets loaded successfully.")


# --------------------------------------------------
# 5. LOAD PRETRAINED MOBILENETV2
# --------------------------------------------------

print("\nLoading pretrained MobileNetV2...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers initially.
base_model.trainable = False


# --------------------------------------------------
# 6. BUILD CLASSIFICATION MODEL
# --------------------------------------------------

inputs = tf.keras.Input(
    shape=(224, 224, 3),
    name="input_image"
)

# Training=False keeps pretrained BatchNormalization
# layers in inference mode.
x = base_model(inputs, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dense(
    256,
    activation="relu"
)(x)

x = tf.keras.layers.Dropout(0.4)(x)

outputs = tf.keras.layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="disease_predictions"
)(x)

model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="Plant_Disease_MobileNetV2"
)


# --------------------------------------------------
# 7. COMPILE MODEL
# --------------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel architecture:")
model.summary()


# --------------------------------------------------
# 8. CALLBACKS
# --------------------------------------------------

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(BEST_MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]


# --------------------------------------------------
# 9. TRAIN MODEL
# --------------------------------------------------

print("\nStarting MobileNetV2 training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)


# --------------------------------------------------
# 10. SAVE TRAINED MODEL
# --------------------------------------------------

print("\nSaving final MobileNetV2 model...")

model.save(FINAL_MODEL_PATH)

print(f"Final model saved: {FINAL_MODEL_PATH}")
print(f"Best model saved: {BEST_MODEL_PATH}")


# --------------------------------------------------
# 11. SAVE TRAINING HISTORY
# --------------------------------------------------

history_df = pd.DataFrame(history.history)

history_df.to_csv(
    OUTPUT_PATH / "mobilenet_history.csv",
    index=False
)

print("\nTraining history saved.")


# --------------------------------------------------
# 12. PLOT ACCURACY
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("MobileNetV2 Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH / "mobilenet_accuracy.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 13. PLOT LOSS
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("MobileNetV2 Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH / "mobilenet_loss.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 14. TRAINING SUMMARY
# --------------------------------------------------

best_epoch = int(
    np.argmin(history.history["val_loss"]) + 1
)

best_val_accuracy = float(
    max(history.history["val_accuracy"])
)

print("\n" + "=" * 60)
print("MOBILENETV2 TRAINING COMPLETED!")
print("=" * 60)

print(f"Epochs completed: {len(history.history['loss'])}")
print(f"Best epoch by validation loss: {best_epoch}")
print(f"Best validation accuracy: {best_val_accuracy * 100:.2f}%")
print(f"Final training accuracy: {history.history['accuracy'][-1] * 100:.2f}%")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1] * 100:.2f}%")

print("\nTraining files saved in:")
print(OUTPUT_PATH)

print("=" * 60)
