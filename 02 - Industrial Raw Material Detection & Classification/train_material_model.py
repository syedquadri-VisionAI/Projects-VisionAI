
from pathlib import Path
from collections import Counter
import shutil
import yaml
import torch

from ultralytics import YOLO


# =====================================================
# 1. PROJECT PATHS
# =====================================================

BASE_DIR = Path(
    r"C:\path/to/dataset/folder"
)

DATASET_DIR = BASE_DIR / "my_dataset"
OUTPUT_DIR = DATASET_DIR / "cleaned_dataset"

# Original dataset folders
TRAIN_IMAGES = DATASET_DIR / "images" / "train"
VAL_IMAGES = DATASET_DIR / "images" / "val"

TRAIN_LABELS = DATASET_DIR / "labels" / "train"
VAL_LABELS = DATASET_DIR / "labels" / "val"


# =====================================================
# 2. CLASS MAPPING
# =====================================================

# Original ID : New ID

CLASS_MAPPING = {
    15: 0,  # dolomite
    16: 1,  # limestone
    17: 2   # iron_ore
}

CLASS_NAMES = {
    0: "dolomite",
    1: "limestone",
    2: "iron_ore"
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


# =====================================================
# 3. PREPARE CLEANED DATASET
# =====================================================

def prepare_split(split):

    image_dir = DATASET_DIR / "images" / split
    label_dir = DATASET_DIR / "labels" / split

    output_images = OUTPUT_DIR / "images" / split
    output_labels = OUTPUT_DIR / "labels" / split

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    class_counts = Counter()

    copied = 0
    skipped = 0

    for image_path in image_dir.iterdir():

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label_path = label_dir / (image_path.stem + ".txt")

        if not label_path.exists():
            print(f"Skipping missing label: {image_path.name}")
            skipped += 1
            continue

        new_annotations = []

        with open(label_path, "r", encoding="utf-8-sig") as file:

            for line_number, line in enumerate(file, start=1):

                values = line.strip().split()

                if not values:
                    continue

                if len(values) != 5:
                    raise ValueError(
                        f"Invalid annotation: {label_path}, line {line_number}"
                    )

                old_id = int(float(values[0]))

                # Ignore classes other than 15, 16, 17
                if old_id not in CLASS_MAPPING:
                    continue

                x, y, w, h = map(float, values[1:])

                # Validate YOLO normalized coordinates
                if not all(0 <= v <= 1 for v in (x, y, w, h)):
                    raise ValueError(
                        f"Invalid coordinates: {label_path}, line {line_number}"
                    )

                if w <= 0 or h <= 0:
                    raise ValueError(
                        f"Invalid bounding box: {label_path}, line {line_number}"
                    )

                new_id = CLASS_MAPPING[old_id]

                new_annotations.append(
                    f"{new_id} {x} {y} {w} {h}"
                )

                class_counts[new_id] += 1

        # Keep only images containing at least one target class
        if not new_annotations:
            print(f"Skipping image without target classes: {image_path.name}")
            skipped += 1
            continue

        # Copy original image
        shutil.copy2(
            image_path,
            output_images / image_path.name
        )

        # Save remapped annotations
        with open(
            output_labels / (image_path.stem + ".txt"),
            "w",
            encoding="utf-8"
        ) as file:

            file.write("\n".join(new_annotations))

        copied += 1

    print(f"\n{split.upper()} DATASET")
    print("Images copied:", copied)
    print("Images skipped:", skipped)

    return class_counts


# =====================================================
# 4. CREATE CLEANED DATASET
# =====================================================

def prepare_dataset():

    # Only the generated output folder is cleared.
    # Original images and labels remain untouched.

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    print("\nPreparing training dataset...")

    train_counts = prepare_split("train")
    val_counts = prepare_split("val")

    total_counts = train_counts + val_counts

    print("\nCLASS DISTRIBUTION")
    for class_id, class_name in CLASS_NAMES.items():

        print(
            f"{class_name}: "
            f"{total_counts[class_id]} objects"
        )

    if not all(total_counts[i] > 0 for i in CLASS_NAMES):
        raise ValueError(
            "One or more classes have zero annotations. "
            "Check the original class IDs before training."
        )

    # Create data.yaml
    data_yaml = {
        "path": str(OUTPUT_DIR).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "nc": 3,
        "names": CLASS_NAMES
    }

    yaml_path = OUTPUT_DIR / "data.yaml"

    with open(yaml_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(data_yaml, file, sort_keys=False)

    print("\nDataset YAML created:", yaml_path)

    return yaml_path


# =====================================================
# 5. TRAIN YOLO MODEL
# =====================================================

def train_model(yaml_path):

    print("\nLoading YOLO model...")

    model = YOLO("yolo11n.pt")

    # Automatically select GPU if available
    device = 0 if torch.cuda.is_available() else "cpu"

    print("Training device:", device)

    results = model.train(

        data=str(yaml_path),

        epochs=100,
        imgsz=640,
        batch=8,

        device=device,
        workers=0,

        patience=25,

        project=str(BASE_DIR / "training_results"),
        name="material_inspection",

        exist_ok=True,

        plots=True,
        save=True,
        verbose=True
    )

    print("\nTRAINING COMPLETED")

    print("Best model location:")
    print(
        BASE_DIR
        / "training_results"
        / "material_inspection"
        / "weights"
        / "best.pt"
    )

    return results


# =====================================================
# 6. MAIN
# =====================================================

if __name__ == "__main__":

    yaml_path = prepare_dataset()

    print("\nStarting YOLO training...")

    train_model(yaml_path)
