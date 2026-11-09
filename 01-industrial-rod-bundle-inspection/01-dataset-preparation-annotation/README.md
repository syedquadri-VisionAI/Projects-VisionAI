# 01 – Dataset Preparation & Annotation

## Overview

This module demonstrates the dataset preparation workflow for an industrial computer vision application focused on steel rod and rod bundle detection.

The dataset was prepared using video frame extraction, image selection, dataset organization, and manual object annotation using LabelImg.

The prepared datasets are used to train two independent YOLO object detection models.

## Workflow

### 1. Video Frame Extraction

- Used Python and OpenCV to extract frames from industrial video footage.
- Selected relevant frames containing steel rods and rod bundles.
- Saved extracted images for dataset preparation.

**Script:** `extract_frames.py`

### 2. Dataset Collection and Organization

- Organized selected images into separate rod and bundle categories.
- Maintained representative sample images for documentation.
- Prepared separate datasets for individual rod detection and complete bundle detection.

**Folder:** `dataset-examples/`

### 3. Image Annotation Using LabelImg

Manually annotated images using LabelImg and exported bounding boxes in YOLO format.

**Rod Dataset**
- Class ID: `0`
- Class Name: `rod`
- Objective: Identify individual steel rods.

**Bundle Dataset**
- Class ID: `0`
- Class Name: `bundle`
- Objective: Identify complete rod bundles.

These are separate single-class datasets intended for two independent models.

**Folder:** `annotated-examples/`

### 4. Annotation Format

YOLO annotation format:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Bounding box coordinates are normalized relative to image dimensions.

### 5. Dataset Examples

The repository includes selected sample images and annotated examples to demonstrate the dataset preparation process.

The complete training dataset is maintained separately.

## Project Structure

```text
01-dataset-preparation-annotation/
│
├── README.md
├── extract_frames.py
├── dataset-examples/
├── annotated-examples/
└── screenshots/
```

## Tools & Technologies

- Python
- OpenCV
- LabelImg
- YOLO Annotation Format
- Image Processing

## Output

Two independently annotated datasets:

1. Steel Rod Detection Dataset
2. Steel Bundle Detection Dataset

These datasets form the input for the YOLO model training and testing module.

## Notes

The repository contains selected sample images and annotation examples, not necessarily the complete training dataset.

Industrial images and video footage are included only where sharing permission is available.
