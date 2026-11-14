# AI-Based Industrial Conveyor Foreign Material & Anomaly Detection System

**Real-Time Foreign Material Detection, Object Tracking & ROI-Based Video Analytics Using Computer Vision**

## 1. Project Overview

Industrial conveyor systems are widely used to transport raw materials in steel plants, mining operations, cement industries and other bulk material handling facilities.

The presence of unwanted foreign materials in raw material streams can affect product quality, damage downstream equipment and interrupt production operations.

This project implements an AI-powered computer vision solution to detect foreign materials in industrial conveyor videos using a custom-trained YOLO object detection model.

The system supports Region of Interest (ROI) selection, real-time detection, object tracking and video analytics with a live detection counter.

## 2. Project Objectives

- Detect unwanted foreign materials in industrial conveyor raw material streams.
- Perform real-time object detection using a custom-trained YOLO model.
- Monitor a user-defined conveyor region using ROI selection.
- Track detected foreign objects across video frames.
- Display live detection counts and cumulative tracked-object counts.
- Generate processed videos with detection bounding boxes and analytics overlays.

## 3. Key Features

| Feature | Description |
|---|---|
| AI-Based Detection | Custom-trained YOLO model for foreign material detection |
| ROI Selection | Interactive selection of the conveyor monitoring area |
| Object Tracking | ByteTrack-based tracking of detected objects |
| Live Analytics | Current detections and cumulative unique tracked objects |
| Video Processing | Supports manually specified input video paths |
| Output Generation | Saves processed videos with detection overlays |
| Output Management | Automatically maintains a maximum of 6 output videos |
| GPU Acceleration | Supports NVIDIA CUDA-enabled GPUs |

## 4. Technologies Used

- **Programming Language:** Python
- **Computer Vision:** OpenCV
- **Object Detection:** Ultralytics YOLO
- **Object Tracking:** ByteTrack
- **Deep Learning Framework:** PyTorch
- **Video Analytics:** OpenCV Video Processing
- **Hardware Acceleration:** NVIDIA CUDA (when supported)

## 5. System Architecture & Workflow

```text
Industrial Conveyor Video
          |
          v
   Video Input Loading
          |
          v
     ROI Selection
          |
          v
  Custom-Trained YOLO Model
          |
          v
  Foreign Material Detection
          |
          v
     Object Tracking
          |
          v
    ROI-Based Filtering
          |
          v
    Live Video Analytics
          |
          v
  Detection Count & Bounding Boxes
          |
          v
   Processed Video Output
```

## 6. Project Structure

```text
03 - AI-Based Industrial Conveyor Foreign Material & Anomaly Detection System/
│
├── models/
│   └── best.pt
│
├── results-screenshot/
│   ├── detection-result.png
│   └── analytics-result.png
│
├── sample-annotation-example/
│   ├── images/
│   └── labels/
│
├── sample-images/
│   ├── input/
│   └── detection-results/
│
├── foreign_object_detection.py
├── requirements.txt
└── README.md
```

*The folder structure above is the recommended organization. Adjust filenames and subfolders to match the actual files in your repository.*

## 7. Model Information

The system uses a custom-trained YOLO object detection model.

| Parameter | Configuration |
|---|---|
| Model Architecture | YOLO |
| Model Weights | `best.pt` |
| Detection Class | `foreign_object` |
| Number of Classes | 1 |
| Detection Input | Industrial conveyor video |
| Output | Foreign material bounding boxes |

The trained model identifies unwanted foreign materials based on the visual patterns learned from the annotated training dataset.

## 8. Sample Results

### Detection Results

Add your actual detection screenshots below.

![Foreign Object Detection Result](results-screenshot/detection-result.png)

### ROI-Based Video Analytics

The application allows users to select a monitoring region and displays detection analytics alongside the processed video.

![ROI Analytics Result](results-screenshot/analytics-result.png)

## 9. Installation

### Prerequisites

- Python 3.10 or later
- NVIDIA GPU with compatible CUDA-enabled PyTorch installation (optional)
- Custom-trained YOLO model weights
- Input conveyor video

### Clone the Repository

```bash
git clone https://github.com/syedquadri-VisionAI/Projects-VisionAI.git
```

Navigate to the project directory:

```bash
cd "Projects-VisionAI/03 - AI-Based Industrial Conveyor Foreign Material & Anomaly Detection System"
```

### Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

If you do not have a requirements file yet, install the core packages:

```bash
pip install ultralytics opencv-python
```

Install the PyTorch version compatible with your NVIDIA GPU and CUDA environment if GPU acceleration is required.

## 10. How to Run

1. Place the trained model inside the `models/` directory.
2. Open `foreign_object_detection.py`.
3. Configure the model and input video paths.
4. Run the Python script.

```python
MODEL_PATH = r"models/best.pt"
VIDEO_PATH = r"path/to/your/conveyor_video.mp4"
```

Execute:

```bash
python foreign_object_detection.py
```

### ROI Selection

- The first video frame opens for ROI selection.
- Draw a rectangular region around the conveyor monitoring area.
- Press ENTER or SPACE to confirm.
- Video analytics starts automatically.

### Analytics Window

The application displays:

- Foreign material detection bounding boxes.
- Object tracking IDs when available.
- Current foreign object detections within the ROI.
- Cumulative unique tracked-object count.
- Frame processing information.

Press **Q** to stop video processing.

## 11. Output

Processed videos are automatically saved in the `output/` directory.

Each output video contains:

- Original conveyor video.
- Selected ROI boundary.
- Foreign material detection bounding boxes.
- Object tracking IDs.
- Live analytics panel.

The application maintains a maximum of six output videos and removes the oldest video when a new output needs to be saved.

## 12. Applications

Potential industrial applications include:

- Steel manufacturing and raw material handling.
- Mining and mineral processing.
- Conveyor belt monitoring.
- Bulk material inspection.
- Industrial material quality monitoring.
- Automated visual inspection systems.

## 13. Future Enhancements

- Integration with industrial IP cameras and RTSP video streams.
- PLC and relay-based alarm integration.
- Real-time event logging and historical analytics.
- Foreign material detection alerts.
- Integration with industrial dashboards and SCADA systems.
- Edge AI deployment for continuous conveyor monitoring.

## 14. Author

**Syed Quadri**

Software Engineer & Deployment Lead | AI & Computer Vision | Industrial Digitalization

GitHub: [syedquadri-VisionAI](https://github.com/syedquadri-VisionAI)

LinkedIn: [Connect on LinkedIn](https://www.linkedin.com/)

---

*This project demonstrates the application of computer vision and deep learning for automated foreign material inspection and monitoring in industrial conveyor systems.*
