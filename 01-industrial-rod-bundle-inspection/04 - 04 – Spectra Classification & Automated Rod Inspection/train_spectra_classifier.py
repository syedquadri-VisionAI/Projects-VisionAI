
from ultralytics import YOLO
from pathlib import Path
import torch

# ==========================================
# CONFIGURATION
# ==========================================

DATASET = "spectra_dataset"

MODEL = "yolo11n-cls.pt"

EPOCHS = 100
IMAGE_SIZE = 224
BATCH_SIZE = 16

DEVICE = 0 if torch.cuda.is_available() else "cpu"

# ==========================================
# CHECK DATASET
# ==========================================

required = [
    "train/spectra",
    "train/no_spectra",
    "train/uncertain",
    "val/spectra",
    "val/no_spectra",
    "val/uncertain",
    "test/spectra",
    "test/no_spectra",
    "test/uncertain"
]

for folder in required:
    path = Path(DATASET) / folder

    if not path.exists():
        raise FileNotFoundError(
            f"Missing dataset folder: {path}"
        )

    count = len(list(path.glob("*.jpg")))

    print(f"{folder}: {count} JPG images")

    if count == 0:
        raise ValueError(f"No JPG images in {path}")

# ==========================================
# LOAD PRETRAINED CLASSIFICATION MODEL
# ==========================================

print("\nLoading YOLO classification model...")

model = YOLO(MODEL)

# ==========================================
# TRAIN
# ==========================================

if __name__ == "__main__":

    print("\n========== TRAINING STARTED ==========")

    results = model.train(
        data=DATASET,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=4,
        patience=20,
        optimizer="AdamW",
        lr0=0.001,
        project="runs/spectra",
        name="three_class_classifier",
        exist_ok=True,
        seed=42,
        plots=True
    )

    print("\n========== TRAINING COMPLETED ==========")

    print("Best model:")
    print(
        "runs/spectra/three_class_classifier/"
        "weights/best.pt"
    )
