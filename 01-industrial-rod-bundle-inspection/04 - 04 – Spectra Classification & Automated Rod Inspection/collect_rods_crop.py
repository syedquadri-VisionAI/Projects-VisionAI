
import cv2
from pathlib import Path
from ultralytics import YOLO

# ---------------- CONFIGURATION ----------------

MODEL_PATH = "C:/Users/syedq/OneDrive/Desktop/python_practice/ml_training/runs/rod/rod_detection/weights/best.pt"
VIDEO_PATH = r"C:\Users\syedq\OneDrive\Desktop\python_practice\spark_and_spectra\Video\Bundle.mp4"

OUTPUT_DIR = Path("C:/Users/syedq/OneDrive/Desktop/python_practice/ml_training/spectra_dataset/unclassified")

CONFIDENCE = 0.40
FRAME_INTERVAL = 30       # Process every 30 frames
PADDING = 15
MIN_WIDTH = 20
MIN_HEIGHT = 20

# ------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading rod detection model...")
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Could not open input video.")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Video frames: {total_frames}")
print(f"Video FPS: {fps}")

frame_number = 0
saved_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    if frame_number % FRAME_INTERVAL != 0:
        continue

    results = model.predict(
        source=frame,
        conf=CONFIDENCE,
        verbose=False
    )

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            h, w = frame.shape[:2]

            # Add crop padding
            x1 = max(0, x1 - PADDING)
            y1 = max(0, y1 - PADDING)
            x2 = min(w, x2 + PADDING)
            y2 = min(h, y2 + PADDING)

            if (x2 - x1) < MIN_WIDTH:
                continue

            if (y2 - y1) < MIN_HEIGHT:
                continue

            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            saved_count += 1

            filename = (
                f"rod_{saved_count:06d}"
                f"_frame_{frame_number}.jpg"
            )

            cv2.imwrite(
                str(OUTPUT_DIR / filename),
                crop
            )

    if frame_number % 300 == 0:
        print(
            f"Processed {frame_number}/{total_frames} "
            f"| Crops saved: {saved_count}"
        )

cap.release()

print("\n========== COLLECTION COMPLETE ==========")
print(f"Total rod crops: {saved_count}")
print(f"Saved location: {OUTPUT_DIR.resolve()}")
