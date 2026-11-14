
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import pandas as pd
import time

# ==================================================
# 1. PROJECT PATHS
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = ROOT / "data" / "raw" / "PlantVillage_color"
CSV_PATH = ROOT / "data" / "class_distribution.csv"

OUTPUT_PATH = ROOT / "outputs" / "quality_check"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# ==================================================
# 2. CONFIGURATION
# ==================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# ==================================================
# 3. LOAD SELECTED CLASSES
# ==================================================

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_PATH}"
    )

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Class distribution CSV not found: {CSV_PATH}"
    )

df = pd.read_csv(CSV_PATH)

selected_classes = df["Disease Class"].tolist()

print("=" * 60)
print("PLANT DISEASE DATASET - IMAGE QUALITY CHECK")
print("=" * 60)

print(f"Selected classes: {len(selected_classes)}")

# ==================================================
# 4. INITIALIZE VARIABLES
# ==================================================

image_records = []
corrupted_images = []
dimension_errors = []
class_summary = []

total_images = 0
valid_images = 0
total_bytes = 0

start_time = time.time()

# ==================================================
# 5. CHECK EACH CLASS
# ==================================================

for class_index, class_name in enumerate(
    selected_classes, start=1
):

    class_folder = DATASET_PATH / class_name

    if not class_folder.exists():
        print(f"\nMISSING CLASS: {class_name}")
        class_summary.append({
            "Class": class_name,
            "Expected Count": int(
                df.loc[
                    df["Disease Class"] == class_name,
                    "Image Count"
                ].iloc[0]
            ),
            "Actual Count": 0,
            "Valid Images": 0,
            "Corrupted Images": 0,
            "Count Match": False
        })
        continue

    image_files = [
        file for file in class_folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    expected_count = int(
        df.loc[
            df["Disease Class"] == class_name,
            "Image Count"
        ].iloc[0]
    )

    class_valid = 0
    class_corrupted = 0

    print(
        f"\n[{class_index}/{len(selected_classes)}] "
        f"Checking: {class_name}"
    )

    # ==============================================
    # 6. VALIDATE EACH IMAGE
    # ==============================================

    for image_path in image_files:

        total_images += 1

        try:
            file_size = image_path.stat().st_size
            total_bytes += file_size

            # Verify image integrity
            with Image.open(image_path) as image:
                image.verify()

            # Reopen after verify to inspect dimensions
            with Image.open(image_path) as image:
                width, height = image.size
                image_format = image.format

            if width <= 0 or height <= 0:
                raise ValueError("Invalid image dimensions")

            valid_images += 1
            class_valid += 1

            image_records.append({
                "Class": class_name,
                "Filename": image_path.name,
                "Width": width,
                "Height": height,
                "Format": image_format,
                "File Size (KB)": round(
                    file_size / 1024, 2
                ),
                "Status": "Valid"
            })

        except Exception as error:

            class_corrupted += 1

            error_message = (
                f"{type(error).__name__}: {error}"
            )

            corrupted_images.append({
                "Class": class_name,
                "Filename": image_path.name,
                "Error": error_message
            })

            if "dimension" in str(error).lower():
                dimension_errors.append(
                    str(image_path)
                )

    # ==============================================
    # 7. CLASS SUMMARY
    # ==============================================

    actual_count = len(image_files)

    class_summary.append({
        "Class": class_name,
        "Expected Count": expected_count,
        "Actual Count": actual_count,
        "Valid Images": class_valid,
        "Corrupted Images": class_corrupted,
        "Count Match": actual_count == expected_count
    })

    print(f"Expected images: {expected_count}")
    print(f"Actual images: {actual_count}")
    print(f"Valid images: {class_valid}")
    print(f"Corrupted images: {class_corrupted}")

# ==================================================
# 8. GENERATE REPORTS
# ==================================================

image_df = pd.DataFrame(image_records)
corrupted_df = pd.DataFrame(corrupted_images)
class_df = pd.DataFrame(class_summary)

image_df.to_csv(
    OUTPUT_PATH / "image_quality_report.csv",
    index=False
)

corrupted_df.to_csv(
    OUTPUT_PATH / "corrupted_images.csv",
    index=False
)

class_df.to_csv(
    OUTPUT_PATH / "class_quality_summary.csv",
    index=False
)

# ==================================================
# 9. FINAL SUMMARY
# ==================================================

elapsed_time = time.time() - start_time

count_mismatches = (
    class_df["Count Match"] == False
).sum()

total_expected = int(df["Image Count"].sum())

summary = f"""
====================================================
PLANT DISEASE DATASET - QUALITY CHECK SUMMARY
====================================================

Selected classes: {len(selected_classes)}

Expected images: {total_expected}
Actual images checked: {total_images}

Valid images: {valid_images}
Corrupted images: {len(corrupted_images)}

Image count mismatches: {count_mismatches}
Invalid dimension errors: {len(dimension_errors)}

Total dataset size: {total_bytes / (1024 ** 2):.2f} MB

Processing time: {elapsed_time:.2f} seconds

====================================================
RESULT
====================================================
"""

if (
    valid_images == total_expected
    and len(corrupted_images) == 0
    and count_mismatches == 0
):
    summary += "\nALL QUALITY CHECKS PASSED!"
else:
    summary += "\nQUALITY ISSUES DETECTED. REVIEW REPORTS."

print(summary)

with open(
    OUTPUT_PATH / "quality_summary.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(summary)

print("\nReports saved in:", OUTPUT_PATH)
