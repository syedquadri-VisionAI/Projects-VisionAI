
# Industrial Raw Material Detection & Classification

### AI-Based Material Inspection Using YOLO11, Python and OpenCV

## Project Overview

This project focuses on detecting and classifying industrial raw materials from images and video footage using a custom-trained YOLO11 object detection model.

The system identifies three types of raw materials:

- **Dolomite**
- **Limestone**
- **Iron Ore**

The objective is to explore how computer vision can support raw material identification, visual inspection and monitoring in industrial environments.

## Project Workflow

The following diagram provides a high-level overview of the complete project workflow.

```text
       Raw Material Images / Video
                  |
                  v
          Frame Extraction
                  |
                  v
       Dataset Preparation
                  |
                  v
       Image Annotation
                  |
                  v
        Model Configuration
                  |
                  v
          YOLO11 Training
                  |
                  v
         Trained Detection Model
                  |
                  v
          Video Input
                  |
                  v
       Frame-by-Frame Detection
                  |
                  v
       Material Classification
                  |
                  v
       Annotated Output Video
```

## Project Structure

| Folder / File | Description |
|---|---|
| `data-annotation-preparation/` | Resources and files related to dataset preparation and annotation. |
| `sample-images/` | Sample raw material images used for reference and testing. |
| `screenshot-results/` | Screenshots showing model detection and inference results. |
| `data.yaml` | YOLO dataset configuration containing class names and dataset details. |
| `extract_frames.py` | Extracts image frames from video footage for dataset preparation. |
| `train_material_model.py` | Script used to train the custom YOLO11 model. |
| `materials_detect.py` | Performs material detection and classification using the trained model. |

## Detection Classes

| Class ID | Material |
|---:|---|
| 0 | Dolomite |
| 1 | Limestone |
| 2 | Iron Ore |

## Technologies Used

- Python
- YOLO11 (Ultralytics)
- OpenCV
- PyTorch
- LabelImg
- Computer Vision

## Potential Industrial Applications

- Raw material identification in steel manufacturing.
- Conveyor belt material monitoring.
- Automated visual inspection.
- Material classification and segregation.
- AI-based industrial monitoring systems.

## Project Scope

This project demonstrates the application of deep learning and computer vision to industrial raw material detection. It provides a foundation for further development towards real-time camera integration, material monitoring and industrial automation.

---

**Domain:** Industrial AI | Computer Vision | Steel Manufacturing

**Technologies:** Python | YOLO11 | OpenCV | PyTorch
