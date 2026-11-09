# Module 03 – Integrated Rod & Bundle Inspection System

## 1. Project Overview

This module demonstrates an integrated industrial computer vision application that combines two independently trained YOLO object detection models to inspect steel rod bundles from video input.

The system uses:

- **Bundle Detection Model:** Identifies individual rod bundles in the video.
- **Rod Detection Model:** Detects individual rods within the detected bundle region.

The application processes each detected bundle, runs rod detection on its region, and estimates the number of rods associated with each bundle.

The objective is to explore automated rod inspection and per-bundle rod counting using deep learning and computer vision.

## 2. Project Objectives

- Detect individual steel rod bundles from video.
- Assign tracking IDs to detected bundles.
- Extract bundle regions for further inspection.
- Detect individual rods within each bundle.
- Estimate the rod count separately for each bundle.
- Display detection results and bundle-wise counts on the video.
- Generate annotated inspection output.

## 3. System Architecture

```text
                 INPUT VIDEO
                      |
                      v
             BUNDLE DETECTION
                (YOLO Model)
                      |
                      v
             BUNDLE IDENTIFICATION
               & TRACKING IDs
                      |
                      v
              BUNDLE REGION
                 EXTRACTION
                      |
                      v
                ROD DETECTION
                 (YOLO Model)
                      |
                      v
               ROD DETECTION
                  COUNTING
                      |
                      v
             BUNDLE-WISE RESULTS
                      |
                      v
              ANNOTATED VIDEO
```

## 4. Detection Models

The application integrates two separate single-class YOLO models.

| Model | Class ID | Class Name | Purpose |
|---|---:|---|---|
| Rod Model | 0 | rod | Detect individual rods |
| Bundle Model | 0 | bundle | Detect rod bundles |

Both models are trained independently using their respective datasets and integrated into a common video-processing pipeline.

## 5. Processing Workflow

### Step 1 – Input Video

The application reads an input video using OpenCV.

### Step 2 – Bundle Detection

The bundle detection model identifies rod bundles within the video frame.

### Step 3 – Bundle Tracking

Detected bundles are assigned tracking IDs to distinguish individual bundles during video processing.

### Step 4 – Bundle Region Extraction

The detected bundle region is extracted and prepared for rod detection.

### Step 5 – Rod Detection

The rod detection model processes the extracted bundle region and identifies individual rods.

### Step 6 – Bundle-wise Rod Count

The application estimates the number of rods detected within each bundle region and associates the result with its bundle ID.

### Step 7 – Visual Output

The application displays bundle detection boxes, rod detection boxes, bundle IDs, and estimated rod counts on the processed video.

## 6. Example Inspection Output

The intended display format is:

```text
Bundle ID: 1
Total Rods in Bundle 1: XX

Bundle ID: 2
Total Rods in Bundle 2: XX

Bundle ID: 3
Total Rods in Bundle 3: XX
```

`XX` represents the number of rod detections returned by the model for that bundle.

The count is an estimated computer vision result and is not necessarily the verified physical number of rods.

## 7. Technologies Used

- Python
- Ultralytics YOLO
- PyTorch
- OpenCV
- NumPy
- Deep Learning
- Object Detection
- Object Tracking
- Region of Interest (ROI) Processing
- Video Analytics

## 8. Repository Structure

```text
03-integrated-rod-bundle-inspection/
│
├── README.md
├── main.py
├── config.py
├── requirements.txt
│
├── models/
│   └── README.md
│
├── screenshots/
│   ├── 01-input-video.png
│   ├── 02-bundle-detection.png
│   ├── 03-rod-detection.png
│   ├── 04-bundle-rod-count.png
│   └── 05-final-output.png
│
└── sample-output/
   
