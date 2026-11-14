
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ==================================================
# 1. PROJECT PATHS
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = ROOT / "data" / "raw" / "PlantVillage_color"
CSV_PATH = ROOT / "data" / "class_distribution.csv"

OUTPUT_PATH = ROOT / "outputs" / "splits"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# ==================================================
# 2. CONFIGURATION
# ==================================================

RANDOM_STATE = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# ==================================================
# 3. LOAD SELECTED CLASSES
# ==================================================

class_df = pd.read_csv(CSV_PATH)

selected_classes = class_df["Disease Class"].tolist()

print("=" * 60)
print("PLANT DISEASE DATASET SPLITTING")
print("=" * 60)

# ==================================================
# 4. COLLECT IMAGE PATHS AND LABELS
# ==================================================

records = []

for class_id, class_name in enumerate(selected_classes):

    class_folder = DATASET_PATH / class_name

    if not class_folder.exists():
        raise FileNotFoundError(
            f"Missing class folder: {class_folder}"
        )

    image_files = sorted([
        file for file in class_folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ])

    expected_count = int(
        class_df.loc[
            class_df["Disease Class"] == class_name,
            "Image Count"
        ].iloc[0]
    )

    if len(image_files) != expected_count:
        raise ValueError(
            f"Count mismatch in {class_name}: "
            f"expected {expected_count}, "
            f"found {len(image_files)}"
        )

    for image_path in image_files:

        records.append({
            "filepath": image_path.relative_to(ROOT).as_posix(),
            "class_name": class_name,
            "label": class_id
        })

    print(f"{class_name}: {len(image_files)} images")

# Create complete dataframe
data = pd.DataFrame(records)

print("\nTotal images collected:", len(data))
print("Total classes:", data["class_name"].nunique())

# ==================================================
# 5. FIRST SPLIT: 70% TRAIN, 30% TEMPORARY
# ==================================================

train_df, temp_df = train_test_split(
    data,
    test_size=0.30,
    stratify=data["label"],
    random_state=RANDOM_STATE
)

# ==================================================
# 6. SECOND SPLIT: 15% VALIDATION, 15% TEST
# ==================================================

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=RANDOM_STATE
)

# ==================================================
# 7. SAVE SPLIT FILES
# ==================================================

train_df.to_csv(
    OUTPUT_PATH / "train.csv",
    index=False
)

val_df.to_csv(
    OUTPUT_PATH / "validation.csv",
    index=False
)

test_df.to_csv(
    OUTPUT_PATH / "test.csv",
    index=False
)

# Save class-to-label mapping
class_mapping = {
    class_name: class_id
    for class_id, class_name in enumerate(selected_classes)
}

import json

with open(
    OUTPUT_PATH / "class_mapping.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(class_mapping, file, indent=4)

# ==================================================
# 8. VERIFY CLASS DISTRIBUTION
# ==================================================

train_counts = train_df["class_name"].value_counts()
val_counts = val_df["class_name"].value_counts()
test_counts = test_df["class_name"].value_counts()

distribution = pd.DataFrame({
    "Class": selected_classes,
    "Train": [
        train_counts.get(c, 0) for c in selected_classes
    ],
    "Validation": [
        val_counts.get(c, 0) for c in selected_classes
    ],
    "Test": [
        test_counts.get(c, 0) for c in selected_classes
    ]
})

distribution.to_csv(
    OUTPUT_PATH / "split_distribution.csv",
    index=False
)

# ==================================================
# 9. FINAL SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("DATASET SPLITTING SUMMARY")
print("=" * 60)

print(f"Total images:      {len(data)}")
print(
    f"Training images:   {len(train_df)} "
    f"({len(train_df)/len(data)*100:.2f}%)"
)
print(
    f"Validation images: {len(val_df)} "
    f"({len(val_df)/len(data)*100:.2f}%)"
)
print(
    f"Testing images:    {len(test_df)} "
    f"({len(test_df)/len(data)*100:.2f}%)"
)

print("\nClasses in each split:")
print("Training:", train_df["label"].nunique())
print("Validation:", val_df["label"].nunique())
print("Testing:", test_df["label"].nunique())

print("\nSaved files:")
print("train.csv")
print("validation.csv")
print("test.csv")
print("class_mapping.json")
print("split_distribution.csv")

print("\nDATASET SPLITTING COMPLETED!")
