
import os
import cv2
import torch
from tkinter import Tk, filedialog
from ultralytics import YOLO

# ==========================================
# 1. MODEL PATH
# ==========================================

MODEL_PATH = (
    r"C:\path\to\model\best.pt"
)

# ==========================================
# 2. OUTPUT DIRECTORY
# ==========================================

OUTPUT_DIR = (
    r"C:\path\to\save\the\results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR, "bundle_detection_output.mp4"
)

# ==========================================
# 3. SELECT INPUT VIDEO
# ==========================================

root = Tk()
root.withdraw()
root.attributes("-topmost", True)

VIDEO_PATH = filedialog.askopenfilename(
    title="Select Bundle Inspection Video",
    filetypes=[
        ("Video files", "*.mp4 *.avi *.mov *.mkv"),
        ("All files", "*.*")
    ]
)

root.destroy()

if not VIDEO_PATH:
    print("No video selected. Exiting.")
    raise SystemExit

if not os.path.isfile(MODEL_PATH):
    print("ERROR: Model file not found:")
    print(MODEL_PATH)
    raise SystemExit

# ==========================================
# 4. CHECK GPU
# ==========================================

if torch.cuda.is_available():
    DEVICE = 0
    print("GPU:", torch.cuda.get_device_name(0))
else:
    DEVICE = "cpu"
    print("CUDA unavailable. Using CPU.")

# ==========================================
# 5. LOAD YOLO MODEL
# ==========================================

print("\nLoading bundle detection model...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully!")

# ==========================================
# 6. OPEN VIDEO
# ==========================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    raise SystemExit

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if fps <= 0:
    fps = 25

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)

if not out.isOpened():
    print("ERROR: Could not create output video.")
    cap.release()
    raise SystemExit

# ==========================================
# 7. PROCESS VIDEO FRAME BY FRAME
# ==========================================

frame_number = 0
total_detections = 0

print("\nStarting bundle detection...")
print("Press Q in the video window to stop.\n")

cv2.namedWindow("Bundle Detection", cv2.WINDOW_NORMAL)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # YOLO prediction
    results = model.predict(
        source=frame,
        conf=0.25,
        imgsz=640,
        device=DEVICE,
        verbose=False
    )

    result = results[0]

    # Count detections in this frame
    detections = len(result.boxes)
    total_detections += detections

    # Draw bounding boxes
    annotated_frame = result.plot()

    # Display frame counter
    cv2.putText(
        annotated_frame,
        f"Frame: {frame_number}/{total_frames}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # Display bundle count in current frame
    cv2.putText(
        annotated_frame,
        f"Bundles detected: {detections}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # Save annotated frame
    out.write(annotated_frame)

    # Show video window
    cv2.imshow("Bundle Detection", annotated_frame)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("\nDetection stopped by user.")
        break

# ==========================================
# 8. RELEASE RESOURCES
# ==========================================

cap.release()
out.release()
cv2.destroyAllWindows()

# ==========================================
# 9. FINAL REPORT
# ==========================================

print("\n========== DETECTION COMPLETED ==========")

print("Video:", VIDEO_PATH)
print("Frames processed:", frame_number)
print("Total detections across frames:", total_detections)

print("\nAnnotated video saved at:")
print(OUTPUT_VIDEO)

print("\n=========================================")
