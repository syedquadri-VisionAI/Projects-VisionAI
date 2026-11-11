
import cv2
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO


# =====================================================
# 1. CONFIGURATION
# =====================================================

BASE_DIR = Path(
    r"C:\Users\syedq\OneDrive\Desktop\Projects\raw_dataset\material_inspection"
)

# Trained YOLO model
MODEL_PATH = (
    BASE_DIR
    / "training_results"
    / "material_inspection"
    / "weights"
    / "best.pt"
)

# Manually enter your input video path
VIDEO_PATH = Path(
    r"C:path/to/source/or/rtsp"
)

# Output directory
OUTPUT_DIR = BASE_DIR / "inference_outputs"

# Maximum saved output videos
MAX_OUTPUTS = 3

# Detection settings
CONFIDENCE = 0.25
IMAGE_SIZE = 640

# Bounding box and text settings
BOX_THICKNESS = 1
TEXT_SIZE = 0.35

# Live processing window
SHOW_WINDOW = True


# =====================================================
# 2. VALIDATE PATHS
# =====================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )

if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Input video not found: {VIDEO_PATH}"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# 3. LOAD YOLO MODEL
# =====================================================

print("\nLoading material inspection model...")

model = YOLO(str(MODEL_PATH))

print("Model loaded successfully.")
print("Available classes:", model.names)


# =====================================================
# 4. OPEN INPUT VIDEO
# =====================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Unable to open input video.")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if fps <= 0:
    fps = 25.0

print("\nVIDEO INFORMATION")
print("-------------------------")
print("Resolution:", width, "x", height)
print("FPS:", fps)
print("Total frames:", total_frames)


# =====================================================
# 5. CREATE OUTPUT VIDEO
# =====================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

output_path = (
    OUTPUT_DIR
    / f"material_inspection_{timestamp}.mp4"
)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(output_path),
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("Unable to create output video.")


# =====================================================
# 6. VIDEO PROCESSING
# =====================================================

frame_number = 0
stopped_by_user = False

print("\nStarting material detection...")
print("Press Q in the video window to stop.\n")

try:

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        # ---------------------------------------------
        # YOLO DETECTION
        # ---------------------------------------------

        results = model.predict(
            source=frame,
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            verbose=False
        )

        # ---------------------------------------------
        # DRAW BOUNDING BOXES AND LABELS
        # ---------------------------------------------

        annotated_frame = results[0].plot(
            line_width=BOX_THICKNESS,
            font_size=TEXT_SIZE,
            labels=True,
            boxes=True,
            conf=True
        )

        # ---------------------------------------------
        # SAVE PROCESSED FRAME
        # ---------------------------------------------

        writer.write(annotated_frame)

        # ---------------------------------------------
        # LIVE VIDEO WINDOW
        # ---------------------------------------------

        if SHOW_WINDOW:

            cv2.imshow(
                "Material Inspection - Live Processing",
                annotated_frame
            )

            # Press Q to stop
            if cv2.waitKey(1) & 0xFF == ord("q"):

                stopped_by_user = True

                print("\nProcessing stopped by user.")
                break

        # ---------------------------------------------
        # PROGRESS DISPLAY
        # ---------------------------------------------

        if frame_number % 50 == 0:

            if total_frames > 0:

                progress = (
                    frame_number / total_frames
                ) * 100

                print(
                    f"Processed {frame_number}/{total_frames} "
                    f"frames ({progress:.1f}%)"
                )

            else:

                print(
                    f"Processed frames: {frame_number}"
                )


except Exception:

    # Delete incomplete output if processing fails
    if output_path.exists():
        output_path.unlink()

    raise


finally:

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


# =====================================================
# 7. KEEP MAXIMUM 3 OUTPUT VIDEOS
# =====================================================

video_extensions = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv"
}

saved_videos = sorted(
    [
        file for file in OUTPUT_DIR.iterdir()
        if file.is_file()
        and file.suffix.lower() in video_extensions
    ],
    key=lambda file: file.stat().st_mtime,
    reverse=True
)

for old_video in saved_videos[MAX_OUTPUTS:]:

    old_video.unlink()

    print(
        "Deleted older output:",
        old_video.name
    )


# =====================================================
# 8. COMPLETION DETAILS
# =====================================================

print("\n========================================")
print("MATERIAL INSPECTION COMPLETED")
print("========================================")

print("Input video:", VIDEO_PATH)
print("Processed frames:", frame_number)
print("Output video:", output_path)

if stopped_by_user:
    print("Status: Stopped manually")
else:
    print("Status: Video processing completed")

print("\nOutput folder:", OUTPUT_DIR)

print("\nSAVED VIDEOS")

saved_videos = sorted(
    OUTPUT_DIR.glob("*.mp4"),
    key=lambda file: file.stat().st_mtime,
    reverse=True
)

for video in saved_videos:
    print(video.name)

print("\nMaximum output videos:", MAX_OUTPUTS)
print("========================================")
