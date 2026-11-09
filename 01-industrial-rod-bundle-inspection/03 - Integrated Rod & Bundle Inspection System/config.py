from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

# Project root: the folder containing this config.py file
BASE_DIR = Path(__file__).resolve().parent

# Change these paths to match your repository folder structure.
VIDEO_PATH = BASE_DIR / "sample_video" / "demo.mp4"

BUNDLE_MODEL_PATH = (
    BASE_DIR / "models" / "bundle_detection" / "weights" / "best.pt"
)

ROD_MODEL_PATH = (
    BASE_DIR / "models" / "rod_detection" / "weights" / "best.pt"
)

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_VIDEO_PATH = OUTPUT_DIR / "counted_video.mp4"


# ============================================================
# BUNDLE MODEL SETTINGS
# ============================================================

BUNDLE_CONFIDENCE = 0.25
BUNDLE_IOU = 0.50
BUNDLE_IMAGE_SIZE = 640

# None means use all classes. For a single "bundle" class, None is fine.
BUNDLE_CLASS_IDS = None


# ============================================================
# ROD MODEL SETTINGS
# ============================================================

ROD_CONFIDENCE = 0.25
ROD_IOU = 0.0
ROD_IMAGE_SIZE = 640

# None means use all classes.
# If you only want particular class IDs, use a list such as [0].
ROD_CLASS_IDS = None


# ============================================================
# TRACKING SETTINGS
# ============================================================

TRACKER_CONFIG = "bytetrack.yaml"
TRACKER_PERSIST = True


# ============================================================
# ROI AND CROP SETTINGS
# ============================================================

# Extra crop margin around the detected bundle:
# 0.05 = 5% of bundle width/height on each side.
CROP_PADDING = 0.05

# Count a bundle only when its complete bounding box is inside the ROI.
REQUIRE_FULL_BUNDLE_INSIDE_ROI = True

# Draw the ROI rectangle and its label.
SHOW_ROI = True


# ============================================================
# DISPLAY SETTINGS
# ============================================================

WINDOW_NAME = "Steel Rod Bundle Counter"
DISPLAY_SCALE = 1.0

SHOW_BUNDLE_BOX = True
SHOW_BUNDLE_ID = True
SHOW_ROD_BOXES = True

# Thin, readable detection boxes.
BUNDLE_BOX_THICKNESS = 2
ROD_BOX_THICKNESS = 1

# Keep display uncluttered: only show the total counters in the overlay.
SHOW_ROD_CLASS_LABELS = False

QUIT_KEY = "q"


# ============================================================
# COUNTING SETTINGS
# ============================================================

# Count each tracked bundle ID once.
COUNT_EACH_BUNDLE_ONCE = True

# When True, rod detections are counted once when a tracked bundle
# first qualifies inside the ROI. This is a single-frame count.
COUNT_RODS_ONCE_PER_BUNDLE = True
