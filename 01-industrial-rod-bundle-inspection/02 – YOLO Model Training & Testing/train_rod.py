
import shutil
import torch
from pathlib import Path
from multiprocessing import freeze_support
from ultralytics import YOLO


# ==================================================
# 1. CONFIGURATION
# ==================================================

ORIGINAL_DATASET = Path(
    r"C:\Users\syedq\OneDrive\Desktop\python_practice\spark_and_spectra\my_dataset\rod_dataset"
)

TRAINING_FOLDER = Path(
    r"C:\Users\syedq\OneDrive\Desktop\python_practice\ml_training"
)

FILTERED_DATASET = TRAINING_FOLDER / "filtered_rod_dataset"

MODEL_NAME = "yolov8n.pt"

EPOCHS = 100
IMAGE_SIZE = 640
BATCH_SIZE = 4
WORKERS = 0

ORIGINAL_ROD_CLASS_ID = 15
NEW_ROD_CLASS_ID = 0

IMAGE_EXTENSIONS = [
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
]


# ==================================================
# 2. GPU CHECK
# ==================================================

def check_gpu():

    print("\n========== GPU CHECK ==========")

    print("PyTorch:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA GPU not detected.")

    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)

    return 0


# ==================================================
# 3. PREPARE ROD-ONLY DATASET
# ==================================================

def prepare_dataset():

    print("\n========== PREPARING ROD-ONLY DATASET ==========")

    if not ORIGINAL_DATASET.exists():
        raise FileNotFoundError(ORIGINAL_DATASET)

    if FILTERED_DATASET.exists():
        shutil.rmtree(FILTERED_DATASET)

    total_images = 0
    total_skipped_images = 0
    total_rod_boxes = 0
    total_other_boxes = 0

    for split in ["train", "val"]:

        source_images = ORIGINAL_DATASET / "images" / split
        source_labels = ORIGINAL_DATASET / "labels" / split

        target_images = FILTERED_DATASET / "images" / split
        target_labels = FILTERED_DATASET / "labels" / split

        if not source_images.exists():
            raise FileNotFoundError(source_images)

        if not source_labels.exists():
            raise FileNotFoundError(source_labels)

        target_images.mkdir(parents=True, exist_ok=True)
        target_labels.mkdir(parents=True, exist_ok=True)

        copied = 0
        skipped = 0
        rod_boxes = 0
        other_boxes = 0

        for image_file in source_images.iterdir():

            if image_file.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            label_file = source_labels / (image_file.stem + ".txt")

            if not label_file.exists():
                print(f"SKIPPED - No label: {image_file.name}")
                skipped += 1
                continue

            lines = label_file.read_text(
                encoding="utf-8"
            ).splitlines()

            rod_lines = []

            for line_number, line in enumerate(lines, start=1):

                if not line.strip():
                    continue

                parts = line.split()

                # Skip malformed annotations
                if len(parts) != 5:
                    print(
                        f"SKIPPED - Invalid annotation: "
                        f"{label_file.name}, line {line_number}"
                    )
                    other_boxes += 1
                    continue

                try:
                    class_id = int(parts[0])
                    coordinates = [float(x) for x in parts[1:]]

                except ValueError:
                    print(
                        f"SKIPPED - Invalid values: "
                        f"{label_file.name}, line {line_number}"
                    )
                    other_boxes += 1
                    continue

                # Skip every class except rod (15)
                if class_id != ORIGINAL_ROD_CLASS_ID:
                    other_boxes += 1
                    continue

                # Skip invalid bounding boxes
                if not all(0 <= x <= 1 for x in coordinates):
                    print(
                        f"SKIPPED - Invalid rod coordinates: "
                        f"{label_file.name}, line {line_number}"
                    )
                    other_boxes += 1
                    continue

                # Convert rod class ID 15 to 0
                parts[0] = str(NEW_ROD_CLASS_ID)

                rod_lines.append(" ".join(parts))
                rod_boxes += 1

            # Skip image if no valid rod annotation exists
            if not rod_lines:
                print(f"SKIPPED - No valid rods: {image_file.name}")
                skipped += 1
                continue

            # Copy image and save only rod annotations
            shutil.copy2(
                image_file,
                target_images / image_file.name
            )

            target_label = target_labels / label_file.name

            target_label.write_text(
                "\n".join(rod_lines) + "\n",
                encoding="utf-8"
            )

            copied += 1

        total_images += copied
        total_skipped_images += skipped
        total_rod_boxes += rod_boxes
        total_other_boxes += other_boxes

        print(f"\n{split.upper()} SUMMARY")
        print("Rod images copied:", copied)
        print("Images skipped:", skipped)
        print("Rod boxes retained:", rod_boxes)
        print("Other/invalid boxes skipped:", other_boxes)

        if copied == 0:
            raise RuntimeError(
                f"No valid rod images found in {split}."
            )

    # Generate single-class YAML
    yaml_content = f"""# Rod Only Dataset

path: '{FILTERED_DATASET.as_posix()}'

train: images/train
val: images/val

nc: 1

names:
  0: rod
"""

    yaml_path = FILTERED_DATASET / "dataset.yaml"

    yaml_path.write_text(
        yaml_content,
        encoding="utf-8"
    )

    print("\n========== ROD DATASET READY ==========")
    print("Total rod images:", total_images)
    print("Total skipped images:", total_skipped_images)
    print("Total rod boxes:", total_rod_boxes)
    print("Total other/invalid boxes skipped:", total_other_boxes)
    print("Dataset YAML:", yaml_path)

    return yaml_path


# ==================================================
# 4. TRAIN YOLO MODEL
# ==================================================

def train_model():

    print("\n========== STARTING ROD GPU TRAINING ==========")

    device = check_gpu()

    yaml_path = prepare_dataset()

    print("\nLoading model:", MODEL_NAME)

    model = YOLO(MODEL_NAME)

    results = model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=device,
        workers=WORKERS,
        project=str(TRAINING_FOLDER / "runs" / "rod"),
        name="rod_detection",
        exist_ok=True,
        patience=20,
        save=True,
        plots=True,
        verbose=True,
        cache=False
    )

    # ==================================================
    # 5. TRAINING RESULTS
    # ==================================================

    print("\n========== ROD TRAINING COMPLETED ==========")

    save_directory = Path(results.save_dir)

    best_model = save_directory / "weights" / "best.pt"
    last_model = save_directory / "weights" / "last.pt"

    print("\nBEST MODEL:")
    print(best_model)

    print("\nLAST MODEL:")
    print(last_model)

    if best_model.exists():
        print("\nSUCCESS: Best rod model saved successfully!")
    else:
        print("\nWARNING: Best model file not found.")

    return results


# ==================================================
# 6. WINDOWS SAFE MAIN
# ==================================================

if __name__ == "__main__":

    freeze_support()

    try:
        train_model()

    except Exception as error:
        print("\n========== ROD TRAINING FAILED ==========")
        print("Error:", error)
        raise
