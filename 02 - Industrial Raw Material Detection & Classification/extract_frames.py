import cv2
from pathlib import Path

# Project folder
BASE_DIR = Path(__file__).resolve().parent

# Video location
VIDEO_PATH = r"C:\Users\syedq\OneDrive\Desktop\Projects\raw_dataset\material_inspection_sample_videos\stone_conveyor (3).mp4"

# Output folder
OUTPUT_DIR = BASE_DIR / "extracted_frames2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Extract one image every 3 seconds
INTERVAL_SECONDS = 0.5

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    print("Error: Could not open video!")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_interval = max(1, int(fps * INTERVAL_SECONDS))

saved_count = 0
frame_number = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if frame_number % frame_interval == 0:
        saved_count += 1

        filename = OUTPUT_DIR / f"img{saved_count}.jpg"

        cv2.imwrite(str(filename), frame)

        print("Saved:", filename.name)

    frame_number += 1

cap.release()

print(f"\nCompleted! Extracted {saved_count} images.")