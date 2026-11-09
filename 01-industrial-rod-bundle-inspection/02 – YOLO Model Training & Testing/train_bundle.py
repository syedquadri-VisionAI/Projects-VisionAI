
import shutil
import torch
from pathlib import Path
from multiprocessing import freeze_support
from ultralytics import YOLO


# ==================================================
# 1. CONFIGURATION
# ==================================================

ORIGINAL_DATASET = Path(
    r"C:\path\to\my_dataset\rod_dataset"
)

TRAINING_FOLDER = Path(
    r"C:\path\to\training\folder"
)

FILTERED_DATASET = TRAINING_FOLDER / "filtered_bundle_dataset"

MODEL_NAME = "yolov8n.pt"

EPOCHS = 100
IMAGE_SIZE = 640
BATCH_SIZE = 4
WORKERS = 0

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


# ==================================================
# 2. GPU CHECK
# ==================================================

def check_gpu():

    print("\n========== GPU CHECK ==========")

    print("PyTorch:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA GPU not detected. Training stopped.")

    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)

    return 0


# ==================================================
# 3. PREPARE FILTERED DATASET
# ==================================================

def prepare_dataset():

    print("\n========== PREPARING DATASET ==========")

    if not ORIGINAL_DATASET.exists():
        raise FileNotFoundError(ORIGINAL_DATASET)

    # Recreate filtered dataset without touching originals
    if FILTERED_DATASET.exists():
        shutil.rmtree(FILTERED_DATASET)

    total_copied = 0
    total_skipped = 0

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

        for image_file in source_images.iterdir():

            if image_file.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            label_file = source_labels / (image_file.stem + ".txt")

            # Skip images without annotation files
            if not label_file.exists():
                print(f"SKIPPED - No label: {image_file.name}")
                skipped += 1
                continue

            lines = label_file.read_text(
                encoding="utf-8"
            ).splitlines()

            # Remove empty lines
            lines = [line.strip() for line in lines if line.strip()]

            # Skip empty annotation files
            if not lines:
                print(f"SKIPPED - Empty label: {image_file.name}")
                skipped += 1
                continue

            # Validate annotations
            for line_number, line in enumerate(lines, start=1):

                parts = line.split()

                if len(parts) != 5:
                    raise ValueError(
                        f"Invalid YOLO format: {label_file}, line {line_number}"
                    )

                try:
                    class_id = int(parts[0])
                    coordinates = [float(x) for x in parts[1:]]

                except ValueError:
                    raise ValueError(
                        f"Invalid annotation values: {label_file}, line {line_number}"
                    )

                if class_id != 0:
                    raise ValueError(
                        f"Unexpected class ID {class_id} in {label_file}"
                    )

                if not all(0 <= x <= 1 for x in coordinates):
                    raise ValueError(
                        f"Coordinates outside 0-1 range: {label_file}, line {line_number}"
                    )

            # Copy only correctly labeled images and labels
            shutil.copy2(image_file, target_images / image_file.name)
            shutil.copy2(label_file, target_labels / label_file.name)

            copied += 1

        total_copied += copied
        total_skipped += skipped

        print(f"\n{split.upper()} SUMMARY")
        print("Labeled images copied:", copied)
        print("Unlabeled/empty images skipped:", skipped)

        if copied == 0:
            raise RuntimeError(f"No valid labeled images found in {split}.")

    # Generate YAML automatically
    yaml_content = f"""# Filtered Bundle Dataset

path: '{FILTERED_DATASET.as_posix()}'

train: images/train
val: images/val

nc: 1

names:
  0: bundle
"""

    yaml_path = FILTERED_DATASET / "dataset.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")

    print("\n========== DATASET READY ==========")
    print("Total labeled images:", total_copied)
    print("Total skipped images:", total_skipped)
    print("Dataset YAML:", yaml_path)

    return yaml_path


# ==================================================
# 4. TRAIN YOLO MODEL
# ==================================================

def train_model():

    print("\n========== STARTING GPU TRAINING ==========")

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

        project=str(TRAINING_FOLDER / "runs" / "bundle"),

        name="bundle_detection",

        exist_ok=True,

        patience=20,

        save=True,

        plots=True,

        verbose=True,

        cache=False

    )

    print("\n========== TRAINING COMPLETED ==========")

    save_directory = Path(results.save_dir)

    best_model = save_directory / "weights" / "best.pt"
    last_model = save_directory / "weights" / "last.pt"

    print("\nBEST MODEL:")
    print(best_model)

    print("\nLAST MODEL:")
    print(last_model)

    if best_model.exists():
        print("\nSUCCESS: Best model saved successfully!")
    else:
        print("\nWARNING: Best model file not found.")

    return results


# ==================================================
# 5. WINDOWS SAFE MAIN
# ==================================================

if __name__ == "__main__":

    freeze_support()

    try:

        train_model()

    except Exception as error:

        print("\n========== GPU TRAINING FAILED ==========")
        print("Error:", error)

        raise
