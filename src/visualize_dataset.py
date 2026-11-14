
from pathlib import Path
import random
import math
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Project paths
ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = ROOT / "data" / "raw" / "PlantVillage_color"
CSV_PATH = ROOT / "data" / "class_distribution.csv"

OUTPUT_PATH = ROOT / "outputs" / "visualizations"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Configuration
SAMPLES_PER_CLASS = 2
COLUMNS = 6
RANDOM_SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

random.seed(RANDOM_SEED)

# Load selected classes from CSV
df = pd.read_csv(CSV_PATH)

selected_classes = df["Disease Class"].tolist()

print(f"Selected classes: {len(selected_classes)}")

# Validate dataset
if not DATASET_PATH.exists():
    raise FileNotFoundError(DATASET_PATH)

available_classes = {
    folder.name for folder in DATASET_PATH.iterdir()
    if folder.is_dir()
}

missing_classes = set(selected_classes) - available_classes

if missing_classes:
    raise ValueError(
        f"Missing dataset folders: {missing_classes}"
    )

# Select images
selected_images = []

for class_name in selected_classes:

    class_folder = DATASET_PATH / class_name

    image_files = [
        file for file in class_folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    samples = random.sample(
        image_files,
        min(SAMPLES_PER_CLASS, len(image_files))
    )

    for image_path in samples:
        selected_images.append((image_path, class_name))

    print(f"{class_name}: {len(image_files)} images")

# Create grid
total = len(selected_images)
rows = math.ceil(total / COLUMNS)

fig, axes = plt.subplots(
    rows, COLUMNS,
    figsize=(20, rows * 3.2)
)

axes = axes.flatten()

displayed = 0
failed = []

for index, (image_path, class_name) in enumerate(selected_images):

    ax = axes[index]

    try:
        with Image.open(image_path) as image:
            ax.imshow(image.convert("RGB"))

        ax.set_title(
            class_name.replace("___", "\n"),
            fontsize=8
        )

        ax.axis("off")
        displayed += 1

    except Exception as error:
        failed.append(str(image_path))
        ax.set_title("Failed to load")
        ax.axis("off")

for index in range(total, len(axes)):
    axes[index].axis("off")

plt.suptitle(
    "PlantVillage - Selected 23 Disease Categories",
    fontsize=16
)

plt.tight_layout(rect=[0, 0, 1, 0.98])

output_file = OUTPUT_PATH / "selected_23_classes.png"

plt.savefig(output_file, dpi=150, bbox_inches="tight")

print("\n========== SUMMARY ==========")
print(f"Selected classes: {len(selected_classes)}")
print(f"Images selected: {total}")
print(f"Images displayed: {displayed}")
print(f"Failed images: {len(failed)}")
print(f"Saved to: {output_file}")

plt.show()