
from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight

from preprocess_dataset import create_datasets, NUM_CLASSES


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models"
OUTPUT_PATH = ROOT / "outputs" / "training"
SPLIT_PATH = ROOT / "outputs" / "splits"

MODEL_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001
RANDOM_SEED = 42

tf.keras.utils.set_random_seed(RANDOM_SEED)


# ==================================================
# 2. LOAD CLASS MAPPING
# ==================================================

with open(
    SPLIT_PATH / "class_mapping.json",
    "r",
    encoding="utf-8"
) as file:
    class_mapping = json.load(file)

assert len(class_mapping) == NUM_CLASSES

print("=" * 60)
print("CUSTOM CNN - PLANT DISEASE CLASSIFICATION")
print("=" * 60)

print("Number of classes:", NUM_CLASSES)
print("Image size:", IMAGE_SIZE)
print("Batch size:", BATCH_SIZE)
print("Maximum epochs:", EPOCHS)


# ==================================================
# 3. LOAD PREPROCESSED DATASETS
# ==================================================

print("\nLoading training and validation datasets...")

train_ds, val_ds, test_ds = create_datasets("custom_cnn")

print("Training dataset ready.")
print("Validation dataset ready.")


# ==================================================
# 4. HANDLE CLASS IMBALANCE
# ==================================================

train_df = pd.read_csv(SPLIT_PATH / "train.csv")

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=train_df["label"].to_numpy()
)

class_weights = {
    int(class_id): float(weight)
    for class_id, weight in enumerate(class_weights_array)
}

print("\nClass weights calculated.")
print("Minimum class weight:", round(min(class_weights.values()), 3))
print("Maximum class weight:", round(max(class_weights.values()), 3))


# ==================================================
# 5. BUILD CUSTOM CNN MODEL
# ==================================================

def build_custom_cnn():

    inputs = tf.keras.Input(
        shape=(224, 224, 3),
        name="input_image"
    )

    x = inputs

    filters_list = [32, 64, 128, 256]

    for block_index, filters in enumerate(filters_list, start=1):

        # First convolution
        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size=3,
            padding="same",
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            name=f"block{block_index}_conv1"
        )(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        # Second convolution
        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size=3,
            padding="same",
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            name=f"block{block_index}_conv2"
        )(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        # Reduce spatial dimensions
        x = tf.keras.layers.MaxPooling2D(
            pool_size=2,
            name=f"block{block_index}_pool"
        )(x)

        # Regularization
        x = tf.keras.layers.Dropout(0.15)(x)

    # Global feature extraction
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Classification head
    x = tf.keras.layers.Dense(
        256,
        activation="relu"
    )(x)

    x = tf.keras.layers.Dropout(0.4)(x)

    # 23-class prediction
    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="disease_predictions"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="Plant_Disease_Custom_CNN"
    )

    return model


# ==================================================
# 6. INITIALIZE MODEL
# ==================================================

print("\nBuilding Custom CNN...")

model = build_custom_cnn()

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ==================================================
# 7. TRAINING CALLBACKS
# ==================================================

best_model_path = MODEL_PATH / "custom_cnn_best.keras"

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(best_model_path),
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
        min_lr=0.00001,
        verbose=1
    )
]


# ==================================================
# 8. TRAIN MODEL
# ==================================================

print("\nStarting Custom CNN training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)


# ==================================================
# 9. SAVE TRAINED MODEL
# ==================================================

final_model_path = MODEL_PATH / "custom_cnn_final.keras"

model.save(final_model_path)

print("\nModel saved successfully!")
print("Best model:", best_model_path)
print("Final model:", final_model_path)


# ==================================================
# 10. SAVE TRAINING HISTORY
# ==================================================

history_df = pd.DataFrame(history.history)

history_df.to_csv(
    OUTPUT_PATH / "custom_cnn_history.csv",
    index=False
)

# Save the epoch with the best validation loss
best_epoch = int(
    np.argmin(history.history["val_loss"]) + 1
)

print("\nBest epoch:", best_epoch)
print(
    "Best validation accuracy:",
    round(max(history.history["val_accuracy"]), 4)
)


# ==================================================
# 11. PLOT TRAINING ACCURACY
# ==================================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Custom CNN - Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH / "custom_cnn_accuracy.png",
    dpi=150
)

plt.close()


# ==================================================
# 12. PLOT TRAINING LOSS
# ==================================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Custom CNN - Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH / "custom_cnn_loss.png",
    dpi=150
)

plt.close()


# ==================================================
# 13. FINAL SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("CUSTOM CNN TRAINING COMPLETED")
print("=" * 60)

print("Epochs completed:", len(history.history["loss"]))

print(
    "Final training accuracy:",
    round(history.history["accuracy"][-1], 4)
)

print(
    "Final validation accuracy:",
    round(history.history["val_accuracy"][-1], 4)
)

print(
    "Best validation accuracy:",
    round(max(history.history["val_accuracy"]), 4)
)

print("\nTraining reports saved in:", OUTPUT_PATH)

print("\nCUSTOM CNN TRAINING FINISHED!")
