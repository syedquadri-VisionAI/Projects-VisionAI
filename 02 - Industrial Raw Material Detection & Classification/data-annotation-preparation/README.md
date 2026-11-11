
## Dataset Preparation & Training Flow

The dataset preparation process follows this structure:

```text
data-annotation-preparation/
│
├── raw-images/
│   └── Original Images
│
├── annotated-images/
│   ├── Images/
│   │   ├── train/
│   │   └── val/
│   │
│   └── Labels/
│       ├── train/
│       └── val/
│
└── data.yaml
        |
        v
   Dataset Configuration
        |
        v
     YOLO11 Training
        |
        v
   Trained Model (best.pt)
        |
        v
   Material Detection
```

### Dataset Annotation & Training Steps

1. **Image Collection:** Collect raw material images from available image sources and video footage.
2. **Frame Extraction:** Extract frames from videos using `extract_frames.py`.
3. **Image Annotation:** Annotate Dolomite, Limestone and Iron Ore using a bounding-box annotation tool.
4. **Dataset Organization:** Organize annotated images and corresponding YOLO label files into training and validation folders.
5. **Dataset Configuration:** Configure dataset paths and class names in `data.yaml`.
6. **Model Training:** Train the custom YOLO11 model using `train_material_model.py`.
7. **Model Output:** Use the trained `best.pt` weights for material detection and video inference.

### YOLO Dataset Structure

```text
dataset/
│
├── images/
│   ├── train/
│   └── val/
│
├── labels/
│   ├── train/
│   └── val/
│
└── data.yaml
```

**Note:** The directory names above illustrate the dataset organization expected by YOLO. Adapt them to the actual paths configured in `data.yaml`.
