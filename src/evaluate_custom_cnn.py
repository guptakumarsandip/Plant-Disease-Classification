
from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from preprocess_dataset import create_datasets


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "custom_cnn_best.keras"
MAPPING_PATH = ROOT / "outputs" / "splits" / "class_mapping.json"

OUTPUT_PATH = ROOT / "outputs" / "evaluation"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD CLASS MAPPING
# --------------------------------------------------

with open(MAPPING_PATH, "r", encoding="utf-8") as file:
    class_mapping = json.load(file)

class_names = [
    name for name, index in
    sorted(class_mapping.items(), key=lambda item: item[1])
]

NUM_CLASSES = len(class_names)

print("=" * 60)
print("CUSTOM CNN MODEL EVALUATION")
print("=" * 60)

print(f"Number of classes: {NUM_CLASSES}")


# --------------------------------------------------
# 3. LOAD SAVED BEST MODEL
# --------------------------------------------------

print("\nLoading best Custom CNN model...")

model = tf.keras.models.load_model(MODEL_PATH)

model.summary()


# --------------------------------------------------
# 4. LOAD TEST DATASET
# --------------------------------------------------

print("\nLoading test dataset...")

_, _, test_ds = create_datasets("custom_cnn")


# --------------------------------------------------
# 5. EVALUATE MODEL
# --------------------------------------------------

print("\nEvaluating model on test dataset...")

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# --------------------------------------------------
# 6. GENERATE PREDICTIONS
# --------------------------------------------------

print("\nGenerating predictions...")

y_true = np.concatenate([
    labels.numpy() for _, labels in test_ds
])

predictions = model.predict(test_ds, verbose=1)

y_pred = np.argmax(predictions, axis=1)

accuracy = accuracy_score(y_true, y_pred)

print(f"\nVerified Test Accuracy: {accuracy * 100:.2f}%")


# --------------------------------------------------
# 7. CLASSIFICATION REPORT
# --------------------------------------------------

print("\nClassification Report:")

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(NUM_CLASSES)),
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

print(report_df.round(4))

report_df.to_csv(
    OUTPUT_PATH / "custom_cnn_classification_report.csv"
)


# --------------------------------------------------
# 8. CONFUSION MATRIX
# --------------------------------------------------

print("\nGenerating confusion matrix...")

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(NUM_CLASSES))
)

fig, ax = plt.subplots(figsize=(20, 18))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=90,
    colorbar=False,
    values_format="d"
)

plt.title("Custom CNN - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    OUTPUT_PATH / "custom_cnn_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 9. SAVE SUMMARY
# --------------------------------------------------

summary = {
    "model": "Custom CNN",
    "number_of_classes": NUM_CLASSES,
    "test_accuracy": float(accuracy),
    "test_accuracy_percentage": float(accuracy * 100),
    "test_loss": float(test_loss),
    "macro_precision": float(report["macro avg"]["precision"]),
    "macro_recall": float(report["macro avg"]["recall"]),
    "macro_f1_score": float(report["macro avg"]["f1-score"]),
    "weighted_precision": float(report["weighted avg"]["precision"]),
    "weighted_recall": float(report["weighted avg"]["recall"]),
    "weighted_f1_score": float(report["weighted avg"]["f1-score"])
}

with open(
    OUTPUT_PATH / "custom_cnn_evaluation_summary.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(summary, file, indent=4)


# --------------------------------------------------
# 10. FINAL OUTPUT
# --------------------------------------------------

print("\n" + "=" * 60)
print("CUSTOM CNN EVALUATION COMPLETED!")
print("=" * 60)

print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")
print(f"Macro F1-score: {summary['macro_f1_score']:.4f}")
print(f"Weighted F1-score: {summary['weighted_f1_score']:.4f}")

print(f"\nResults saved in: {OUTPUT_PATH}")
print("=" * 60)
