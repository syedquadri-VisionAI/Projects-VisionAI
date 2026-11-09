
from pathlib import Path
import sys

import cv2
from ultralytics import YOLO

import config


def validate_file(path: Path, description: str):
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(
            f"{description} not found:\n{path}\n"
            "Check config.py"
        )


def draw_label(image, text, x, y, color=(30, 150, 30)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1

    (tw, th), baseline = cv2.getTextSize(
        text, font, scale, thickness
    )

    x = max(0, int(x))
    y = max(th + baseline + 5, int(y))

    cv2.rectangle(
        image,
        (x, y - th - baseline - 5),
        (x + tw + 8, y + 3),
        color,
        -1
    )

    cv2.putText(
        image,
        text,
        (x + 4, y - baseline - 1),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


def select_roi(frame):
    print("\nDraw inspection ROI.")
    print("Press ENTER/SPACE to confirm.")

    roi = cv2.selectROI(
        "Select Inspection ROI",
        frame,
        showCrosshair=True,
        fromCenter=False
    )

    cv2.destroyWindow("Select Inspection ROI")

    x, y, w, h = map(int, roi)

    if w <= 0 or h <= 0:
        raise RuntimeError("Invalid ROI selected.")

    return x, y, x + w, y + h


def is_bundle_inside_roi(box, roi):
    x1, y1, x2, y2 = box
    rx1, ry1, rx2, ry2 = roi

    if config.REQUIRE_FULL_BUNDLE_INSIDE_ROI:
        return (
            x1 >= rx1 and
            y1 >= ry1 and
            x2 <= rx2 and
            y2 <= ry2
        )

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2


def crop_bundle(frame, box):
    """Return bundle crop and its exact frame coordinates."""

    frame_h, frame_w = frame.shape[:2]

    x1, y1, x2, y2 = map(int, box)

    pad_x = int((x2 - x1) * config.CROP_PADDING)
    pad_y = int((y2 - y1) * config.CROP_PADDING)

    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)

    cx2 = min(frame_w, x2 + pad_x)
    cy2 = min(frame_h, y2 + pad_y)

    if cx2 <= cx1 or cy2 <= cy1:
        return None, 0, 0

    crop = frame[cy1:cy2, cx1:cx2].copy()

    return crop, cx1, cy1


def detect_rods(rod_model, bundle_crop):

    if bundle_crop is None or bundle_crop.size == 0:
        return []

    results = rod_model.predict(
        source=bundle_crop,
        conf=config.ROD_CONFIDENCE,
        iou=config.ROD_IOU,
        imgsz=config.ROD_IMAGE_SIZE,

        # IMPORTANT:
        # Your current rod model has only one class.
        # Do not filter its class ID using old configuration.
        classes=None,

        verbose=False
    )

    if not results or results[0].boxes is None:
        return []

    detections = []

    for box in results[0].boxes:

        coords = box.xyxy[0].cpu().tolist()
        confidence = float(box.conf[0].cpu())

        detections.append((coords, confidence))

    return detections


def draw_rod_boxes(display, detections, offset_x, offset_y):

    for rod_box, confidence in detections:

        rx1, ry1, rx2, ry2 = map(int, rod_box)

        # Convert crop coordinates into original frame coordinates.
        rx1 += offset_x
        ry1 += offset_y
        rx2 += offset_x
        ry2 += offset_y

        cv2.rectangle(
            display,
            (rx1, ry1),
            (rx2, ry2),
            (255, 180, 0),
            1
        )


def draw_bundle_summary(display, bundle_counts):

    row_height = 32
    panel_height = 48 + len(bundle_counts) * row_height

    panel_width = 440

    cv2.rectangle(
        display,
        (10, 10),
        (panel_width, panel_height),
        (25, 25, 25),
        -1
    )

    cv2.putText(
        display,
        "BUNDLE-WISE ROD COUNT",
        (22, 37),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    total_rods = 0

    for index, (track_id, data) in enumerate(bundle_counts.items()):

        bundle_number = data["bundle_number"]
        rods = data["best_count"]

        total_rods += rods

        text = (
            f"BundleID = {bundle_number}  |  "
            f"Total rods in Bundle{bundle_number} = {rods}"
        )

        cv2.putText(
            display,
            text,
            (22, 65 + index * row_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return total_rods


def main():

    validate_file(config.VIDEO_PATH, "Input video")
    validate_file(config.BUNDLE_MODEL_PATH, "Bundle model")
    validate_file(config.ROD_MODEL_PATH, "Rod model")

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading bundle model...")
    bundle_model = YOLO(str(config.BUNDLE_MODEL_PATH))

    print("Loading rod model...")
    rod_model = YOLO(str(config.ROD_MODEL_PATH))

    cap = cv2.VideoCapture(str(config.VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError("Could not open input video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ok, first_frame = cap.read()

    if not ok:
        cap.release()
        raise RuntimeError("Could not read video.")

    roi = select_roi(first_frame)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(config.OUTPUT_VIDEO_PATH),
        fourcc,
        fps,
        (frame_w, frame_h)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output video.")

    # Track ID -> bundle information.
    bundle_counts = {}

    next_bundle_number = 1

    frame_number = 0

    print("\n========== PROCESSING STARTED ==========\n")

    try:

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            frame_number += 1

            clean_frame = frame.copy()
            display = frame.copy()

            tracking_results = bundle_model.track(
                source=frame,
                persist=config.TRACKER_PERSIST,
                tracker=config.TRACKER_CONFIG,
                conf=config.BUNDLE_CONFIDENCE,
                iou=config.BUNDLE_IOU,
                imgsz=config.BUNDLE_IMAGE_SIZE,
                classes=config.BUNDLE_CLASS_IDS,
                verbose=False
            )

            if (
                tracking_results
                and tracking_results[0].boxes is not None
            ):

                for detected_box in tracking_results[0].boxes:

                    if detected_box.id is None:
                        continue

                    coords = detected_box.xyxy[0].cpu().tolist()

                    x1, y1, x2, y2 = map(int, coords)

                    track_id = int(
                        detected_box.id[0].cpu()
                    )

                    # Only inspect bundles inside ROI.
                    if not is_bundle_inside_roi(
                        (x1, y1, x2, y2), roi
                    ):
                        continue

                    # Assign sequential bundle number once.
                    if track_id not in bundle_counts:

                        bundle_counts[track_id] = {
                            "bundle_number": next_bundle_number,
                            "best_count": 0
                        }

                        print(
                            f"\nBundleID = {next_bundle_number}"
                        )

                        next_bundle_number += 1

                    bundle_number = bundle_counts[
                        track_id
                    ]["bundle_number"]

                    # Draw bundle rectangle.
                    if config.SHOW_BUNDLE_BOX:

                        cv2.rectangle(
                            display,
                            (x1, y1),
                            (x2, y2),
                            (0, 220, 0),
                            2
                        )

                    # Run rod detection EVERY FRAME.
                    bundle_crop, crop_x, crop_y = crop_bundle(
                        clean_frame,
                        (x1, y1, x2, y2)
                    )

                    rod_detections = detect_rods(
                        rod_model,
                        bundle_crop
                    )

                    current_rod_count = len(rod_detections)

                    # Retain the highest count observed for this bundle.
                    previous_count = bundle_counts[
                        track_id
                    ]["best_count"]

                    if current_rod_count > previous_count:

                        bundle_counts[
                            track_id
                        ]["best_count"] = current_rod_count

                        print(
                            f"BundleID = {bundle_number} | "
                            f"Total rods in Bundle{bundle_number} = "
                            f"{current_rod_count}"
                        )

                    # Draw rod bounding boxes on every frame.
                    if config.SHOW_ROD_BOXES:

                        draw_rod_boxes(
                            display,
                            rod_detections,
                            crop_x,
                            crop_y
                        )

                    # Show bundle number.
                    if config.SHOW_BUNDLE_ID:

                        draw_label(
                            display,
                            f"BundleID: {bundle_number}",
                            x1,
                            y1,
                            (0, 130, 0)
                        )

            # Draw inspection ROI.
            if config.SHOW_ROI:

                rx1, ry1, rx2, ry2 = roi

                cv2.rectangle(
                    display,
                    (rx1, ry1),
                    (rx2, ry2),
                    (0, 255, 255),
                    2
                )

            # Draw individual bundle counts.
            total_rods = draw_bundle_summary(
                display,
                bundle_counts
            )

            total_bundles = len(bundle_counts)

            # Total counter.
            cv2.putText(
                display,
                f"Total Bundles: {total_bundles}",
                (20, frame_h - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                display,
                f"Total Rods: {total_rods}",
                (20, frame_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            writer.write(display)

            if config.DISPLAY_SCALE != 1.0:

                shown = cv2.resize(
                    display,
                    None,
                    fx=config.DISPLAY_SCALE,
                    fy=config.DISPLAY_SCALE
                )

            else:
                shown = display

            cv2.imshow(config.WINDOW_NAME, shown)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(config.QUIT_KEY):
                print("\nStopped by user.")
                break

    finally:

        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    # Final report.
    print("\n========== FINAL BUNDLE REPORT ==========\n")

    grand_total = 0

    for track_id, data in bundle_counts.items():

        bundle_number = data["bundle_number"]
        rods = data["best_count"]

        grand_total += rods

        print(
            f"BundleID = {bundle_number} "
            f"&& Total rods in Bundle{bundle_number} = {rods}"
        )

    print("\n========================================")
    print(f"Total Bundles = {len(bundle_counts)}")
    print(f"Total Rods = {grand_total}")
    print(f"Output Video = {config.OUTPUT_VIDEO_PATH}")
    print("========================================")


if __name__ == "__main__":

    try:
        main()

    except Exception as exc:

        print(f"\nERROR: {exc}")
        sys.exit(1)
