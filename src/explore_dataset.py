from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. Define dataset path
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "data" / "raw" / "PlantVillage_color"

# ==========================================
# 2. Select our five plants
# ==========================================
SELECTED_PLANTS = [
    "Tomato",
    "Potato",
    "Pepper",
    "Apple",
    "Corn"
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# ==========================================
# 3. Check dataset path
# ==========================================

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}"
    )

# ==========================================
# 4. Find selected disease folders
# ==========================================

data = []

for folder in sorted(DATASET_PATH.iterdir()):

    if not folder.is_dir():
        continue

    plant_name = folder.name.split("___")[0]

    if not any(
        plant_name.startswith(plant)
        for plant in SELECTED_PLANTS
    ):
        continue

    images = [
        file for file in folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    data.append({
        "Disease Class": folder.name,
        "Image Count": len(images)
    })

# ==========================================
# 5. Create DataFrame
# ==========================================

df = pd.DataFrame(data)

if df.empty:
    raise ValueError("No matching plant classes found.")

print("\n========== DATASET SUMMARY ==========")

print(df.to_string(index=False))

print("\nTotal Classes:", len(df))

print("Total Images:", df["Image Count"].sum())

# ==========================================
# 6. Save results
# ==========================================

output_file = BASE_DIR / "data" / "class_distribution.csv"

df.to_csv(output_file, index=False)

print("\nCSV saved successfully!")

# ==========================================
# 7. Visualize class distribution
# ==========================================

plt.figure(figsize=(12, 10))

plt.barh(
    df["Disease Class"],
    df["Image Count"]
)

plt.xlabel("Number of Images")

plt.ylabel("Disease Category")

plt.title("PlantVillage - Selected Plant Disease Classes")

plt.tight_layout()

plt.show()