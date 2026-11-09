
# 04 – Spectra Classification & Automated Rod Inspection

## Overview

This module focuses on automated Spectra classification of individual steel rod faces using Computer Vision and Deep Learning.

The system extracts individual rod-face images using a trained YOLO rod detection model, prepares a three-class classification dataset, trains a YOLO classification model, and integrates the trained classifier into an industrial rod bundle inspection application.

The objective is to automatically identify rod-face conditions based on the presence or absence of a brown Spectra dot.

## 1. Rod Face Crop Extraction

**Script:** `collect_rods_crop.py`

A trained rod detection model is used to:

- Detect individual rods from input images.
- Extract individual rod-face regions.
- Generate cropped rod images.
- Collect rod-face images for classification dataset preparation.

## 2. Spectra Dataset Preparation

The extracted rod-face images are categorized into three classes.

| Class | Description |
|---|---|
| Spectra | Brown dot present on the rod face |
| No Spectra | Clear rod face without a brown dot |
| Uncertain | Coloured, unclear or unrecognized rod-face appearance |

### Dataset Structure

```text
spectra_dataset/
│
├── train/
│   ├── spectra/
│   ├── no_spectra/
│   └── uncertain/
│
├── val/
│   ├── spectra/
│   ├── no_spectra/
│   └── uncertain/
│
└── test/
    ├── spectra/
    ├── no_spectra/
    └── uncertain/
```

## 3. Spectra Classifier Training & Testing

**Script:** `train_spectra_classifier.py`

A YOLO classification model is trained using the prepared three-class dataset.

Training workflow:

- Load the prepared classification dataset.
- Configure the YOLO classification model.
- Train the model.
- Validate model performance.
- Evaluate classification results.
- Save the trained model weights.

**Expected trained model:**

`spectra_classifier_best.pt`

## 4. Integrated Industrial Inspection

**Script:** `industrial_bundle_inspection.py`

The final application integrates three independently trained models:

1. Bundle Detection Model
2. Rod Detection Model
3. Spectra Classification Model

### Inspection Capabilities

- Bundle detection and tracking.
- Individual rod detection.
- Rod-to-bundle assignment.
- Spectra classification.
- No-Spectra identification.
- Uncertain classification.
- Bundle-wise rod counting.
- Classification confidence visualization.
- Annotated image inspection.
- Video inspection and automatic output video saving.

### Bundle-wise Inspection Results

For each detected bundle, the system calculates:

- Total Rods
- Spectra Count
- No-Spectra Count
- Uncertain Count

## 5. Integrated Processing Workflow

```text
Input Image / Video
        |
        v
Bundle Detection Model
        |
        v
Rod Detection Model
        |
        v
Individual Rod Detection
        |
        v
Rod Face Classification
        |
        v
Spectra Classification Model
        |
        v
Spectra / No-Spectra / Uncertain
        |
        v
Bundle-wise Count Calculation
        |
        v
Annotated Inspection Output
        |
        v
Processed Image / Video
```

## 6. Model Files

The application requires three trained model weights:

```text
models/
├── bundle_best.pt
├── rod_best.pt
└── spectra_classifier_best.pt
```

Model weights are maintained separately and are not included in this repository.

## 7. Technologies Used

- Python
- PyTorch
- Ultralytics YOLO
- OpenCV
- NumPy
- Computer Vision
- Deep Learning
- ByteTrack

## 8. Project Outcome

Developed an integrated AI-based industrial rod inspection workflow combining object detection, rod-face image extraction, three-class image classification, bundle-wise counting, and automated video inspection.

The system demonstrates an end-to-end Computer Vision application for industrial quality inspection.

## Author

**Syed Quadri**

GitHub: https://github.com/syedquadri-VisionAI

