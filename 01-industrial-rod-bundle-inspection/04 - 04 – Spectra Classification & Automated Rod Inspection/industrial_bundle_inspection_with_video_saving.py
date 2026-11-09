
import cv2
import torch
import numpy as np
import tkinter as tk
from datetime import datetime

from pathlib import Path
from tkinter import filedialog, messagebox
from ultralytics import YOLO


# ============================================================
# 1. MODEL CONFIGURATION
# ============================================================

BUNDLE_MODEL_PATH = r"C:/Users\syedq\OneDrive\Desktop\python_practice\ml_training\bundle_runs\bundle\bundle_detection\weights\best.pt"
ROD_MODEL_PATH = r"C:\Users\syedq\OneDrive\Desktop\python_practice\ml_training\runs\rod\rod_detection\weights\best.pt"
SPECTRA_MODEL_PATH = r"C:\Users\syedq\OneDrive\Desktop\python_practice\ml_training\runs\classify\runs\spectra\three_class_classifier\weights\best.pt"

ROD_CONF = 0.40
BUNDLE_CONF = 0.40
CLASS_CONF = 0.65

CLASS_IMAGE_SIZE = 224

# Display window maximum dimensions
DISPLAY_WIDTH = 1400
DISPLAY_HEIGHT = 850

# Processed videos will be saved here (relative to the folder you run the script from)
OUTPUT_VIDEO_DIR = Path("output_videos")

# ============================================================
# 2. DEVICE
# ============================================================

DEVICE = 0 if torch.cuda.is_available() else "cpu"

print("======================================")
print("INDUSTRIAL BUNDLE INSPECTION")
print("======================================")
print("Device:", DEVICE)

# ============================================================
# 3. LOAD MODELS
# ============================================================

def load_models():

    paths = [
        BUNDLE_MODEL_PATH,
        ROD_MODEL_PATH,
        SPECTRA_MODEL_PATH
    ]

    for path in paths:
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"Model not found: {path}"
            )

    print("\nLoading Bundle Model...")
    bundle_model = YOLO(BUNDLE_MODEL_PATH)

    print("Loading Rod Model...")
    rod_model = YOLO(ROD_MODEL_PATH)

    print("Loading Spectra Classification Model...")
    spectra_model = YOLO(SPECTRA_MODEL_PATH)

    print("\nAll three models loaded successfully!")

    return bundle_model, rod_model, spectra_model


# ============================================================
# 4. MANUAL IMAGE / VIDEO SELECTION
# ============================================================

def select_file():

    root = tk.Tk()
    root.withdraw()

    root.attributes("-topmost", True)

    file_path = filedialog.askopenfilename(
        title="Select Bundle Inspection Image or Video",

        filetypes=[
            (
                "Supported Files",
                "*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov *.mkv"
            ),
            ("Images", "*.jpg *.jpeg *.png *.bmp"),
            ("Videos", "*.mp4 *.avi *.mov *.mkv"),
            ("All Files", "*.*")
        ]
    )

    root.destroy()

    if not file_path:
        return None

    return file_path


# ============================================================
# 5. DETECT AND TRACK BUNDLES
# ============================================================

def detect_bundles(frame, bundle_model, is_video):

    if is_video:

        results = bundle_model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=BUNDLE_CONF,
            verbose=False
        )

    else:

        results = bundle_model.predict(
            source=frame,
            conf=BUNDLE_CONF,
            verbose=False
        )

    bundles = []

    if not results or results[0].boxes is None:
        return bundles

    boxes = results[0].boxes

    for index, box in enumerate(boxes):

        x1, y1, x2, y2 = map(
            int, box.xyxy[0].tolist()
        )

        track_id = None

        if box.id is not None:
            track_id = int(box.id.item())

        # Static image: assign local bundle ID
        if track_id is None:
            track_id = index + 1

        bundles.append({
            "id": track_id,
            "bbox": (x1, y1, x2, y2)
        })

    return bundles


# ============================================================
# 6. DETECT RODS
# ============================================================

def detect_rods(frame, rod_model):

    results = rod_model.predict(
        source=frame,
        conf=ROD_CONF,
        verbose=False
    )

    rods = []

    if not results or results[0].boxes is None:
        return rods

    for box in results[0].boxes:

        x1, y1, x2, y2 = map(
            int, box.xyxy[0].tolist()
        )

        rods.append({
            "bbox": (x1, y1, x2, y2)
        })

    return rods


# ============================================================
# 7. ASSIGN EACH ROD TO ITS BUNDLE
# ============================================================

def assign_rods_to_bundles(rods, bundles):

    for bundle in bundles:

        bundle["rods"] = []

    for rod in rods:

        rx1, ry1, rx2, ry2 = rod["bbox"]

        center_x = (rx1 + rx2) // 2
        center_y = (ry1 + ry2) // 2

        candidates = []

        for bundle in bundles:

            bx1, by1, bx2, by2 = bundle["bbox"]

            inside = (
                bx1 <= center_x <= bx2
                and by1 <= center_y <= by2
            )

            if inside:

                area = max(
                    1, (bx2 - bx1) * (by2 - by1)
                )

                candidates.append((area, bundle))

        # If multiple bundle boxes overlap,
        # assign to the smallest containing box.
        if candidates:

            selected_bundle = min(
                candidates,
                key=lambda item: item[0]
            )[1]

            selected_bundle["rods"].append(rod)


# ============================================================
# 8. THREE-CLASS SPECTRA CLASSIFICATION
# ============================================================

def classify_rods(
    frame,
    bundles,
    spectra_model
):

    crops = []
    rod_references = []

    height, width = frame.shape[:2]

    # Collect all valid rod crops
    for bundle in bundles:

        for rod in bundle["rods"]:

            x1, y1, x2, y2 = rod["bbox"]

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                rod["class"] = "uncertain"
                rod["confidence"] = 0.0
                continue

            crops.append(crop)
            rod_references.append(rod)

    if not crops:
        return

    # Batch classification
    predictions = spectra_model.predict(
        source=crops,
        imgsz=CLASS_IMAGE_SIZE,
        device=DEVICE,
        verbose=False
    )

    for rod, result in zip(
        rod_references, predictions
    ):

        if result.probs is None:

            rod["class"] = "uncertain"
            rod["confidence"] = 0.0
            continue

        class_id = int(result.probs.top1)

        confidence = float(
            result.probs.top1conf
        )

        predicted_class = str(
            result.names[class_id]
        ).lower().replace("-", "_").replace(" ", "_")

        allowed_classes = [
            "spectra",
            "no_spectra",
            "uncertain"
        ]

        if predicted_class not in allowed_classes:
            predicted_class = "uncertain"

        # Low-confidence predictions become uncertain
        if confidence < CLASS_CONF:
            predicted_class = "uncertain"

        rod["class"] = predicted_class
        rod["confidence"] = confidence


# ============================================================
# 9. CALCULATE BUNDLE-WISE COUNTS
# ============================================================

def calculate_counts(bundles):

    for bundle in bundles:

        rods = bundle.get("rods", [])

        total = len(rods)

        spectra = sum(
            r.get("class") == "spectra"
            for r in rods
        )

        no_spectra = sum(
            r.get("class") == "no_spectra"
            for r in rods
        )

        uncertain = sum(
            r.get("class") == "uncertain"
            for r in rods
        )

        bundle["total_rods"] = total
        bundle["spectra"] = spectra
        bundle["no_spectra"] = no_spectra
        bundle["uncertain"] = uncertain


# ============================================================
# 10. DRAW ROD BOXES
# ============================================================

def draw_rod_boxes(frame, bundles):

    colors = {
        "spectra": (0, 200, 0),
        "no_spectra": (0, 0, 255),
        "uncertain": (0, 200, 255)
    }

    for bundle in bundles:

        for rod in bundle.get("rods", []):

            x1, y1, x2, y2 = rod["bbox"]

            label = rod.get(
                "class", "uncertain"
            )

            confidence = rod.get(
                "confidence", 0.0
            )

            color = colors[label]

            # Thin bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                1
            )

            text = (
                f"{label}: {confidence:.2f}"
            )

            cv2.putText(
                frame,
                text,
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                color,
                1,
                cv2.LINE_AA
            )


# ============================================================
# 11. DRAW BUNDLE BOXES AND COUNTS
# ============================================================

def draw_bundle_info(frame, bundles):

    for bundle in bundles:

        x1, y1, x2, y2 = bundle["bbox"]

        bundle_id = bundle["id"]

        # Bundle bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 180, 0),
            2
        )

        title = f"Bundle ID: {bundle_id}"

        cv2.putText(
            frame,
            title,
            (x1, max(25, y1 - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 180, 0),
            2,
            cv2.LINE_AA
        )

    # Dashboard panel
    panel_x = 10
    panel_y = 10

    panel_width = 330
    panel_height = 42 + len(bundles) * 115

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (panel_x + panel_width, panel_y + panel_height),
        (20, 20, 20),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.75,
        frame,
        0.25,
        0,
        frame
    )

    y = panel_y + 30

    cv2.putText(
        frame,
        "BUNDLE INSPECTION",
        (panel_x + 10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    for bundle in bundles:

        y += 27

        lines = [
            (
                f"Bundle {bundle['id']} | "
                f"Total: {bundle['total_rods']}"
            ),
            f"Spectra: {bundle['spectra']}",
            f"No_Spectra: {bundle['no_spectra']}",
            f"Uncertain: {bundle['uncertain']}"
        ]

        colors = [
            (255, 255, 255),
            (0, 255, 0),
            (0, 0, 255),
            (0, 200, 255)
        ]

        for line, color in zip(lines, colors):

            cv2.putText(
                frame,
                line,
                (panel_x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA
            )

            y += 19


# ============================================================
# 12. PROCESS EACH FRAME
# ============================================================

def process_frame(
    frame,
    bundle_model,
    rod_model,
    spectra_model,
    is_video
):

    bundles = detect_bundles(
        frame,
        bundle_model,
        is_video
    )

    rods = detect_rods(
        frame,
        rod_model
    )

    assign_rods_to_bundles(
        rods,
        bundles
    )

    classify_rods(
        frame,
        bundles,
        spectra_model
    )

    calculate_counts(bundles)

    output = frame.copy()

    draw_rod_boxes(output, bundles)

    draw_bundle_info(output, bundles)

    return output, bundles


# ============================================================
# 13. DISPLAY IMAGE
# ============================================================

def process_image(
    path,
    bundle_model,
    rod_model,
    spectra_model
):

    frame = cv2.imread(path)

    if frame is None:
        raise RuntimeError("Unable to read image.")

    output, bundles = process_frame(
        frame,
        bundle_model,
        rod_model,
        spectra_model,
        False
    )

    print_results(bundles)

    show_output(output)


# ============================================================
# 14. DISPLAY VIDEO
# ============================================================

def process_video(
    path,
    bundle_model,
    rod_model,
    spectra_model
):
    """
    Process an input video, display the annotated frames, and save
    the complete processed video in OUTPUT_VIDEO_DIR.
    """

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {path}")

    video_writer = None
    output_path = None
    frame_number = 0

    # Read source video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Some video files report invalid FPS. Use a safe fallback.
    if not fps or not np.isfinite(fps) or fps <= 0:
        fps = 30.0
        print("Warning: Source FPS could not be read. Using 30 FPS.")

    if frame_width <= 0 or frame_height <= 0:
        cap.release()
        raise RuntimeError("Could not read the input video's dimensions.")

    # Create output folder and a unique output filename
    OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    input_path = Path(path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_VIDEO_DIR / (
        f"{input_path.stem}_processed_{timestamp}.mp4"
    )

    # MP4 writer. The output uses the original video's FPS and resolution.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (frame_width, frame_height)
    )

    if not video_writer.isOpened():
        cap.release()
        raise RuntimeError(
            "Could not create the output video. "
            "Check that OpenCV supports the MP4V codec and that the "
            "output folder is writable."
        )

    print("\\n======================================")
    print("VIDEO INSPECTION STARTED")
    print("======================================")
    print(f"Input video : {input_path}")
    print(f"Output video: {output_path.resolve()}")
    print(f"Resolution  : {frame_width} x {frame_height}")
    print(f"FPS         : {fps:.2f}")
    print(f"Total frames: {total_frames if total_frames > 0 else 'Unknown'}")
    print("\\nPress Q or ESC to stop processing.")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1

            output, bundles = process_frame(
                frame,
                bundle_model,
                rod_model,
                spectra_model,
                True
            )

            # Add frame number to the annotated output.
            cv2.putText(
                output,
                f"Frame: {frame_number}",
                (20, output.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            # Save the full-resolution processed frame.
            # Do not save the resized display frame.
            if output.shape[1] != frame_width or output.shape[0] != frame_height:
                output = cv2.resize(output, (frame_width, frame_height))

            video_writer.write(output)

            # Display a resized copy only; the saved video stays full resolution.
            show_output(output, wait=1)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("\\nProcessing stopped by user.")
                break

            if frame_number % 100 == 0:
                if total_frames > 0:
                    percent = (frame_number / total_frames) * 100
                    print(
                        f"Processed {frame_number}/{total_frames} "
                        f"frames ({percent:.1f}%)"
                    )
                else:
                    print(f"Processed {frame_number} frames...")

    finally:
        # Always release both resources, even if processing raises an error.
        cap.release()

        if video_writer is not None:
            video_writer.release()

        cv2.destroyAllWindows()

    if frame_number == 0:
        # Remove an empty/invalid output file if no frames were processed.
        if output_path is not None and output_path.exists():
            output_path.unlink()
        raise RuntimeError("No frames were read from the selected video.")

    print("\\n======================================")
    print("VIDEO INSPECTION COMPLETED")
    print("======================================")
    print(f"Frames processed: {frame_number}")
    print(f"Saved output    : {output_path.resolve()}")

    try:
        messagebox.showinfo(
            "Inspection Completed",
            f"Processed video saved successfully.\\n\\n"
            f"Location:\\n{output_path.resolve()}\\n\\n"
            f"Frames processed: {frame_number}"
        )
    except Exception:
        pass

    return str(output_path)


# ============================================================
# 15. DISPLAY WINDOW
# ============================================================

def show_output(frame, wait=0):

    h, w = frame.shape[:2]

    scale = min(
        DISPLAY_WIDTH / w,
        DISPLAY_HEIGHT / h,
        1.0
    )

    if scale < 1.0:

        frame = cv2.resize(
            frame,
            (
                int(w * scale),
                int(h * scale)
            )
        )

    cv2.namedWindow(
        "Industrial Bundle Inspection",
        cv2.WINDOW_NORMAL
    )

    cv2.imshow(
        "Industrial Bundle Inspection",
        frame
    )

    if wait == 0:
        cv2.waitKey(0)


# ============================================================
# 16. PRINT RESULTS
# ============================================================

def print_results(bundles):

    print("\n========== INSPECTION RESULTS ==========")

    for bundle in bundles:

        print(
            f"\nBundle ID: {bundle['id']}"
        )

        print(
            "Total Rods:",
            bundle["total_rods"]
        )

        print(
            "Spectra:",
            bundle["spectra"]
        )

        print(
            "No_Spectra:",
            bundle["no_spectra"]
        )

        print(
            "Uncertain:",
            bundle["uncertain"]
        )

        print(
            "Count Validation:",
            bundle["total_rods"] == (
                bundle["spectra"]
                + bundle["no_spectra"]
                + bundle["uncertain"]
            )
        )


# ============================================================
# 17. MAIN APPLICATION
# ============================================================

def main():

    try:

        # Load all models first
        bundle_model, rod_model, spectra_model = (
            load_models()
        )

        while True:

            print("\nSelect an image or video.")

            file_path = select_file()

            if not file_path:
                print("No file selected.")
                break

            extension = Path(file_path).suffix.lower()

            image_extensions = [
                ".jpg", ".jpeg", ".png", ".bmp"
            ]

            video_extensions = [
                ".mp4", ".avi", ".mov", ".mkv"
            ]

            if extension in image_extensions:

                process_image(
                    file_path,
                    bundle_model,
                    rod_model,
                    spectra_model
                )

            elif extension in video_extensions:

                process_video(
                    file_path,
                    bundle_model,
                    rod_model,
                    spectra_model
                )

            else:

                messagebox.showerror(
                    "Unsupported File",
                    "Please select a supported image or video."
                )

                continue

            cv2.destroyAllWindows()

            # Select another file or exit
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)

            again = messagebox.askyesno(
                "Inspection Completed",
                "Do you want to inspect another image/video?"
            )

            root.destroy()

            if not again:
                break

    except Exception as error:

        print("\nAPPLICATION ERROR:")
        print(error)

        try:
            messagebox.showerror(
                "Inspection Error",
                str(error)
            )
        except Exception:
            pass

    finally:

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
