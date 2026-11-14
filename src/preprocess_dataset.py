
from pathlib import Path
import json

import pandas as pd
import tensorflow as tf

# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

SPLIT_PATH = ROOT / "outputs" / "splits"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 23
RANDOM_SEED = 42

AUTOTUNE = tf.data.AUTOTUNE

tf.keras.utils.set_random_seed(RANDOM_SEED)

# ==================================================
# 2. LOAD CLASS MAPPING
# ==================================================

mapping_path = SPLIT_PATH / "class_mapping.json"

with open(mapping_path, "r", encoding="utf-8") as file:
    class_mapping = json.load(file)

if len(class_mapping) != NUM_CLASSES:
    raise ValueError(
        f"Expected {NUM_CLASSES} classes, "
        f"found {len(class_mapping)}"
    )

print("Class mapping loaded successfully.")
print("Number of classes:", len(class_mapping))

# ==================================================
# 3. LOAD DATA SPLITS
# ==================================================

def load_split(filename):

    csv_path = SPLIT_PATH / filename

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    required_columns = {
        "filepath",
        "class_name",
        "label"
    }

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Missing columns in {filename}"
        )

    if df.empty:
        raise ValueError(f"{filename} is empty")

    if set(df["class_name"].unique()) != set(class_mapping):
        raise ValueError(
            f"Unexpected classes in {filename}"
        )

    expected_labels = df["class_name"].map(class_mapping)

    if not expected_labels.equals(
        df["label"].astype(int).rename(expected_labels.name)
    ):
        raise ValueError(
            f"Class labels do not match mapping in {filename}"
        )

    paths = [
        str(ROOT / filepath)
        for filepath in df["filepath"]
    ]

    for path in paths:
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    labels = df["label"].astype("int32").tolist()

    return paths, labels


train_paths, train_labels = load_split("train.csv")
val_paths, val_labels = load_split("validation.csv")
test_paths, test_labels = load_split("test.csv")

print("\nDataset sizes:")
print("Training:", len(train_paths))
print("Validation:", len(val_paths))
print("Testing:", len(test_paths))

# ==================================================
# 4. IMAGE LOADING AND RESIZING
# ==================================================

def load_and_resize_image(image_path, label):

    image = tf.io.read_file(image_path)

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        IMAGE_SIZE,
        antialias=True
    )

    return image, label


# ==================================================
# 5. DATA AUGMENTATION
# ==================================================

augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(
            "horizontal"
        ),

        tf.keras.layers.RandomRotation(
            0.08
        ),

        tf.keras.layers.RandomZoom(
            0.10
        ),
    ],
    name="training_augmentation"
)


# ==================================================
# 6. PREPROCESSING FOR EACH MODEL
# ==================================================

def preprocess_custom_cnn(image, label):

    image = image / 255.0

    return image, label


def preprocess_mobilenet(image, label):

    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )

    return image, label


# ==================================================
# 7. BUILD TF.DATA PIPELINES
# ==================================================

def create_datasets(model_type):

    if model_type == "custom_cnn":
        preprocess_fn = preprocess_custom_cnn

    elif model_type == "mobilenet":
        preprocess_fn = preprocess_mobilenet

    else:
        raise ValueError(
            "model_type must be custom_cnn or mobilenet"
        )

    def create_map_function(is_training):

        def process_image(image_path, label):

            image, label = load_and_resize_image(
                image_path,
                label
            )

            # Augmentation is applied only to training.
            if is_training:
                image = augmentation(
                    image,
                    training=True
                )

            image, label = preprocess_fn(
                image,
                label
            )

            image = tf.ensure_shape(
                image,
                [224, 224, 3]
            )

            return image, label

        return process_image

    # Training dataset
    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_paths, train_labels)
    )

    train_ds = train_ds.shuffle(
        buffer_size=len(train_paths),
        seed=RANDOM_SEED,
        reshuffle_each_iteration=True
    )

    train_ds = train_ds.map(
        create_map_function(True),
        num_parallel_calls=AUTOTUNE
    )

    train_ds = train_ds.batch(
        BATCH_SIZE
    )

    train_ds = train_ds.prefetch(
        AUTOTUNE
    )

    # Validation dataset
    val_ds = tf.data.Dataset.from_tensor_slices(
        (val_paths, val_labels)
    )

    val_ds = val_ds.map(
        create_map_function(False),
        num_parallel_calls=AUTOTUNE
    )

    val_ds = val_ds.batch(
        BATCH_SIZE
    ).prefetch(AUTOTUNE)

    # Testing dataset
    test_ds = tf.data.Dataset.from_tensor_slices(
        (test_paths, test_labels)
    )

    test_ds = test_ds.map(
        create_map_function(False),
        num_parallel_calls=AUTOTUNE
    )

    test_ds = test_ds.batch(
        BATCH_SIZE
    ).prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds


# ==================================================
# 8. VERIFY BOTH PREPROCESSING PIPELINES
# ==================================================

if __name__ == "__main__":

    for model_type in ["custom_cnn", "mobilenet"]:

        print("\n" + "=" * 55)
        print(f"VERIFYING: {model_type.upper()}")
        print("=" * 55)

        train_ds, val_ds, test_ds = create_datasets(
            model_type
        )

        # Inspect one training batch
        images, labels = next(iter(train_ds))

        print("Training image shape:", images.shape)
        print("Training label shape:", labels.shape)

        print(
            "Minimum pixel value:",
            float(tf.reduce_min(images))
        )

        print(
            "Maximum pixel value:",
            float(tf.reduce_max(images))
        )

        print(
            "Unique labels in batch:",
            len(tf.unique(labels)[0])
        )

        # Verify validation and test shapes
        val_images, _ = next(iter(val_ds))
        test_images, _ = next(iter(test_ds))

        print("Validation batch:", val_images.shape)
        print("Testing batch:", test_images.shape)

    print("\n" + "=" * 55)
    print("ALL PREPROCESSING PIPELINES VERIFIED!")
    print("=" * 55)
