
import cv2
import os
import time
from pathlib import Path
from ultralytics import YOLO

# ============================================================
# 1. USER CONFIGURATION
# ============================================================

MODEL_PATH = r"C:\Users\syedq\OneDrive\Desktop\Projects\raw_dataset\conveyor_foreign_material_detection\Conveyor_FOD_Training\Foreign_Object_Detection\weights\best.pt"

VIDEO_PATH = r"C:\Users\syedq\OneDrive\Desktop\Projects\raw_dataset\conveyor_foreign_material_detection\sample_videos\boardcamera_iron_ore.mp4"
OUTPUT_DIR = Path("output")

MAX_OUTPUT_VIDEOS = 6

CONFIDENCE = 0.45
TRACKER = "bytetrack.yaml"

# Set True to use GPU automatically when CUDA is available
USE_GPU = True

WINDOW_NAME = "Foreign Object Detection Analytics"

# ============================================================
# 2. OUTPUT FOLDER MANAGEMENT
# ============================================================

def manage_output_folder():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    supported_formats = (".mp4", ".avi", ".mkv", ".mov")

    videos = [
        file for file in OUTPUT_DIR.iterdir()
        if file.is_file() and file.suffix.lower() in supported_formats
    ]

    # Sort oldest first
    videos.sort(key=lambda file: file.stat().st_mtime)

    # Reserve space for the next output video
    while len(videos) >= MAX_OUTPUT_VIDEOS:

        oldest_video = videos.pop(0)

        try:
            oldest_video.unlink()
            print(f"Deleted oldest video: {oldest_video.name}")
        except OSError as e:
            raise RuntimeError(
                f"Cannot delete oldest video {oldest_video}: {e}"
            )

# ============================================================
# 3. VALIDATE PATHS
# ============================================================

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not os.path.isfile(VIDEO_PATH):
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 4. LOAD YOLO MODEL
# ============================================================

print("\nLoading Foreign Object Detection Model...")

model = YOLO(MODEL_PATH)

device = 0 if USE_GPU else "cpu"

print("Model loaded successfully!")
print("Inference device:", device)

# ============================================================
# 5. OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Unable to open input video.")

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 25.0

video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

ret, first_frame = cap.read()

if not ret:
    cap.release()
    raise RuntimeError("Unable to read the first video frame.")

print(f"Video resolution: {video_width} x {video_height}")
print(f"Video FPS: {fps}")

# ============================================================
# 6. SELECT REGION OF INTEREST (ROI)
# ============================================================

print("\n==========================================")
print("DRAW YOUR REGION OF INTEREST")
print("1. Use mouse to draw ROI.")
print("2. Press ENTER or SPACE to confirm.")
print("3. Press ESC to cancel.")
print("==========================================")

roi = cv2.selectROI(
    "Select ROI",
    first_frame,
    showCrosshair=True,
    fromCenter=False
)

cv2.destroyWindow("Select ROI")

roi_x, roi_y, roi_w, roi_h = map(int, roi)

if roi_w <= 0 or roi_h <= 0:
    cap.release()
    cv2.destroyAllWindows()
    raise RuntimeError("ROI not selected. Restart the script.")

roi_x1 = roi_x
roi_y1 = roi_y
roi_x2 = roi_x + roi_w
roi_y2 = roi_y + roi_h

print("ROI selected successfully!")

# ============================================================
# 7. PREPARE OUTPUT VIDEO
# ============================================================

# Create a timestamped filename
timestamp = time.strftime("%Y%m%d_%H%M%S")

output_path = OUTPUT_DIR / f"FOD_Analytics_{timestamp}.mp4"

video_writer = None

# ============================================================
# 8. ANALYTICS VARIABLES
# ============================================================

unique_object_ids = set()

current_count = 0
total_unique_count = 0

frame_number = 0

# ============================================================
# 9. START VIDEO ANALYTICS
# ============================================================

print("\nStarting Foreign Object Detection Analytics...")
print("Press Q to stop processing.")

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

try:

    while True:

        ret, frame = cap.read()

        if not ret:
            print("\nVideo processing completed.")
            break

        frame_number += 1

        # ====================================================
        # YOLO OBJECT DETECTION AND TRACKING
        # ====================================================

        results = model.track(
            source=frame,
            persist=True,
            tracker=TRACKER,
            conf=CONFIDENCE,
            device=device,
            verbose=False
        )

        result = results[0]

        current_count = 0

        # ====================================================
        # DRAW ROI
        # ====================================================

        cv2.rectangle(
            frame,
            (roi_x1, roi_y1),
            (roi_x2, roi_y2),
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "ANALYSIS ROI",
            (roi_x1, max(roi_y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        # ====================================================
        # PROCESS DETECTIONS
        # ====================================================

        if result.boxes is not None and len(result.boxes) > 0:

            boxes = result.boxes.xyxy.cpu().numpy()

            if result.boxes.id is not None:
                track_ids = result.boxes.id.int().cpu().tolist()
            else:
                track_ids = [None] * len(boxes)

            for box, track_id in zip(boxes, track_ids):

                bx1, by1, bx2, by2 = map(int, box)

                # Detection center
                center_x = int((bx1 + bx2) / 2)
                center_y = int((by1 + by2) / 2)

                # Check whether detection center is inside ROI
                inside_roi = (
                    roi_x1 <= center_x <= roi_x2
                    and roi_y1 <= center_y <= roi_y2
                )

                if not inside_roi:
                    continue

                current_count += 1

                # Unique tracked object count
                if track_id is not None:
                    unique_object_ids.add(track_id)

                # Draw detection bounding box
                cv2.rectangle(
                    frame,
                    (bx1, by1),
                    (bx2, by2),
                    (0, 0, 255),
                    2
                )

                label = "FOREIGN OBJECT"

                if track_id is not None:
                    label += f" ID:{track_id}"

                cv2.putText(
                    frame,
                    label,
                    (bx1, max(by1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

        total_unique_count = len(unique_object_ids)

        # ====================================================
        # 10. CREATE LEFT ANALYTICS PANEL
        # ====================================================

        panel_width = 300

        panel = frame.copy()

        # Dark background for panel
        cv2.rectangle(
            panel,
            (0, 0),
            (panel_width, video_height),
            (25, 25, 25),
            -1
        )

        # Combine panel and original analytics frame
        display_frame = cv2.hconcat([
            panel[:, :panel_width],
            frame
        ])

        # Panel heading
        cv2.putText(
            display_frame,
            "FOREIGN OBJECT",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            "DETECTION ANALYTICS",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        cv2.line(
            display_frame,
            (20, 95),
            (panel_width - 20, 95),
            (150, 150, 150),
            1
        )

        # Current detection count
        cv2.putText(
            display_frame,
            "CURRENT DETECTIONS",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (200, 200, 200),
            1
        )

        cv2.putText(
            display_frame,
            str(current_count),
            (20, 205),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            (0, 255, 0),
            4
        )

        # Total unique objects
        cv2.putText(
            display_frame,
            "TOTAL UNIQUE OBJECTS",
            (20, 270),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (200, 200, 200),
            1
        )

        cv2.putText(
            display_frame,
            str(total_unique_count),
            (20, 330),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            (0, 165, 255),
            4
        )

        # Frame number
        cv2.putText(
            display_frame,
            f"FRAME: {frame_number}",
            (20, 400),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        cv2.putText(
            display_frame,
            "Q - EXIT",
            (20, video_height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        # ====================================================
        # 11. INITIALIZE VIDEO WRITER
        # ====================================================

        if video_writer is None:

            # Make room before creating the next output
            manage_output_folder()

            output_height, output_width = display_frame.shape[:2]

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            video_writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                fps,
                (output_width, output_height)
            )

            if not video_writer.isOpened():
                raise RuntimeError("Unable to create output video.")

            print(f"\nSaving analytics video to: {output_path}")

        # Save processed frame
        video_writer.write(display_frame)

        # ====================================================
        # 12. DISPLAY ANALYTICS WINDOW
        # ====================================================

        cv2.imshow(WINDOW_NAME, display_frame)

        delay = max(1, int(1000 / fps))

        key = cv2.waitKey(delay) & 0xFF

        if key == ord("q"):
            print("\nAnalytics stopped by user.")
            break

finally:

    # ========================================================
    # 13. RELEASE RESOURCES
    # ========================================================

    if video_writer is not None:
        video_writer.release()

    cap.release()
    cv2.destroyAllWindows()

# ============================================================
# 14. FINAL ANALYTICS SUMMARY
# ============================================================

print("\n==========================================")
print("FOREIGN OBJECT DETECTION SUMMARY")
print("==========================================")

print(f"Total Frames Processed : {frame_number}")
print(f"Unique Foreign Objects : {total_unique_count}")
print(f"Output Folder          : {OUTPUT_DIR.resolve()}")

if output_path.exists() and output_path.stat().st_size > 0:
    print(f"Saved Output Video     : {output_path.resolve()}")
else:
    print("No output video was saved.")

print("==========================================")
print("PROCESS COMPLETED")
