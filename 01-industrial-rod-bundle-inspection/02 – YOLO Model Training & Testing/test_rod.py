
import cv2
import torch
from pathlib import Path
from tkinter import Tk, filedialog
from ultralytics import YOLO


# ==================================================
# 1. CONFIGURATION
# ==================================================

MODEL_PATH = Path(
    r"C:\Users\syedq\OneDrive\Desktop\python_practice\ml_training\runs\rod\rod_detection\weights\best.pt"
)

OUTPUT_DIR = Path(
    r"C:\Users\syedq\OneDrive\Desktop\python_practice\ml_training\rod_test_results"
)

CONFIDENCE = 0.25
IMAGE_SIZE = 640

# Bounding box settings
BOX_COLOR = (0, 255, 0)  # Green
BOX_THICKNESS = 1        # Thin bounding box

IMAGE_EXTENSIONS = [
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
]

VIDEO_EXTENSIONS = [
    ".mp4", ".avi", ".mov", ".mkv"
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. GPU CHECK
# ==================================================

print("\n========== GPU CHECK ==========")

if torch.cuda.is_available():

    DEVICE = 0

    print("GPU:", torch.cuda.get_device_name(0))

else:

    DEVICE = "cpu"

    print("CUDA unavailable. Using CPU.")


# ==================================================
# 3. LOAD TRAINED MODEL
# ==================================================

print("\n========== LOADING ROD MODEL ==========")

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = YOLO(str(MODEL_PATH))

print("Model loaded successfully!")
print("Model classes:", model.names)


# ==================================================
# 4. SELECT IMAGE OR VIDEO
# ==================================================

root = Tk()

root.withdraw()

root.attributes("-topmost", True)

FILE_PATH = filedialog.askopenfilename(

    title="Select Image or Video for Rod Detection",

    filetypes=[

        (
            "Images and Videos",
            "*.jpg *.jpeg *.png *.bmp *.webp *.mp4 *.avi *.mov *.mkv"
        ),

        ("All files", "*.*")

    ]

)

root.destroy()


if not FILE_PATH:

    print("\nNo file selected. Exiting.")

    raise SystemExit


FILE_PATH = Path(FILE_PATH)

print("\nSelected file:", FILE_PATH)


# ==================================================
# 5. DRAW THIN BOUNDING BOXES
# ==================================================

def draw_thin_boxes(frame, result):

    annotated_frame = frame.copy()

    if result.boxes is None:

        return annotated_frame

    for box in result.boxes:

        coordinates = box.xyxy[0].cpu().numpy()

        x1, y1, x2, y2 = coordinates

        cv2.rectangle(

            annotated_frame,

            (int(x1), int(y1)),

            (int(x2), int(y2)),

            BOX_COLOR,

            BOX_THICKNESS

        )

    return annotated_frame


# ==================================================
# 6. IMAGE DETECTION
# ==================================================

def test_image():

    print("\n========== IMAGE DETECTION ==========")

    image = cv2.imread(str(FILE_PATH))

    if image is None:

        raise RuntimeError("Could not read image.")

    results = model.predict(

        source=image,

        conf=CONFIDENCE,

        imgsz=IMAGE_SIZE,

        device=DEVICE,

        verbose=False

    )

    result = results[0]

    rod_count = len(result.boxes)

    # Draw boxes without labels
    annotated_image = draw_thin_boxes(
        image,
        result
    )

    print("\nRods detected:", rod_count)

    output_path = OUTPUT_DIR / (
        FILE_PATH.stem + "_rod_detection.jpg"
    )

    cv2.imwrite(
        str(output_path),
        annotated_image
    )

    # Display image
    cv2.namedWindow(
        "Rod Detection - Image",
        cv2.WINDOW_NORMAL
    )

    cv2.imshow(
        "Rod Detection - Image",
        annotated_image
    )

    print("\nOutput saved:", output_path)

    print("Press any key to close the image window.")

    cv2.waitKey(0)

    cv2.destroyAllWindows()


# ==================================================
# 7. VIDEO DETECTION
# ==================================================

def test_video():

    print("\n========== VIDEO DETECTION ==========")

    cap = cv2.VideoCapture(str(FILE_PATH))

    if not cap.isOpened():

        raise RuntimeError("Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:

        fps = 25

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    output_path = OUTPUT_DIR / (
        FILE_PATH.stem + "_rod_detection.mp4"
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(

        str(output_path),

        fourcc,

        fps,

        (width, height)

    )

    if not out.isOpened():

        cap.release()

        raise RuntimeError(
            "Could not create output video."
        )

    frame_number = 0

    total_detections = 0

    # Create video window
    cv2.namedWindow(
        "Rod Detection - Video",
        cv2.WINDOW_NORMAL
    )

    print("\nProcessing video...")

    print("Green thin boxes = detected rods")

    print("No labels or confidence scores displayed.")

    print("Press Q to stop video detection.\n")

    try:

        while True:

            ret, frame = cap.read()

            if not ret:

                break

            frame_number += 1

            # YOLO detection
            results = model.predict(

                source=frame,

                conf=CONFIDENCE,

                imgsz=IMAGE_SIZE,

                device=DEVICE,

                verbose=False

            )

            result = results[0]

            # Count rods in current frame
            rod_count = len(result.boxes)

            total_detections += rod_count

            # Draw only thin green boxes
            annotated_frame = draw_thin_boxes(

                frame,

                result

            )

            # Save annotated frame
            out.write(annotated_frame)

            # Display video
            cv2.imshow(

                "Rod Detection - Video",

                annotated_frame

            )

            # Press Q to stop
            if cv2.waitKey(1) & 0xFF == ord("q"):

                print("\nDetection stopped by user.")

                break

    finally:

        cap.release()

        out.release()

        cv2.destroyAllWindows()

    # Final report
    print("\n========== VIDEO RESULTS ==========")

    print("Frames processed:", frame_number)

    print(
        "Total detections across frames:",
        total_detections
    )

    print("\nAnnotated video saved at:")

    print(output_path)


# ==================================================
# 8. RUN IMAGE OR VIDEO TEST
# ==================================================

if FILE_PATH.suffix.lower() in IMAGE_EXTENSIONS:

    test_image()

elif FILE_PATH.suffix.lower() in VIDEO_EXTENSIONS:

    test_video()

else:

    print("\nUnsupported file format.")

