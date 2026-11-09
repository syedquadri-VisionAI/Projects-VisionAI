# Industrial Rod & Bundle Inspection Using Computer Vision

## Project Overview

An industrial computer vision project designed to detect steel rods, identify rod bundles, and estimate the number of rods present in each bundle using YOLO-based object detection and OpenCV.

The project demonstrates the complete computer vision development workflow, from dataset preparation and image annotation to model training, testing, and integrated video inspection.

## Project Workflow

1. Dataset collection and preparation
2. Image filtering and annotation using LabelImg
3. Independent YOLO model training for rod and bundle detection
4. Model testing on images and videos
5. Integrated rod and bundle detection
6. Bundle-wise rod counting and visual inspection

## Project Modules

### 01 – Dataset Preparation & Annotation

- Collect industrial rod and bundle images.
- Filter and organize image datasets.
- Annotate rod images using LabelImg.
- Annotate bundle images using LabelImg.
- Prepare YOLO-compatible datasets.

### 02 – YOLO Model Training & Testing

Two independent single-class object detection models:

- Rod Detection Model – detects individual rods.
- Bundle Detection Model – detects complete rod bundles.

Includes training, validation, image testing, and video testing scripts.

### 03 – Integrated Rod & Bundle Inspection

The integrated application combines both models to:

- Detect rod bundles in video footage.
- Track bundles and assign bundle IDs.
- Select and process a defined inspection region.
- Detect individual rods inside each bundle.
- Estimate rod counts for individual bundles.
- Display bundle-wise and overall counting information.
- Generate annotated video output.

## Technology Stack

- Python
- OpenCV
- Ultralytics YOLO
- PyTorch
- NumPy
- LabelImg
- Computer Vision
- Object Detection
- Video Processing
- Object Tracking

## Current Development Status

This project is an experimental industrial computer vision prototype. Detection and counting accuracy depend on image quality, annotation quality, model performance, camera position, occlusion, and tracking consistency.

Model performance and physical counting accuracy will be documented after validation.

## Repository Structure

- `01-dataset-preparation-annotation`
- `02-yolo-model-training-testing`
- `03-integrated-rod-bundle-inspection`
- `docs`

## Future Improvements

- Improve detection accuracy with additional annotated data.
- Evaluate counting accuracy against manually verified rod counts.
- Improve tracking stability and bundle identification.
- Optimize inference speed for edge deployment.
- Develop a production-ready inspection dashboard.
