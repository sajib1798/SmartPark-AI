# SmartPark AI

SmartPark AI is a computer-vision-based parking monitoring system that detects parking-space occupancy and common vehicle classes from parking-lot images and videos.

The project combines a custom fine-tuned YOLOv8n parking occupancy model with a COCO-pretrained YOLOv8n vehicle detector and presents the results through an interactive Streamlit dashboard.

## Features

* Empty parking-space detection
* Occupied parking-space detection
* Parking occupancy percentage
* Vehicle detection
* Car, motorcycle, bus, and truck counting
* Image inference
* Video inference
* Bounding-box visualization
* Detection confidence scores
* Parking analytics
* Streamlit dashboard
* Model evaluation using Precision, Recall, mAP@50, and mAP@50-95
* Automated dataset validation
* Pytest-based unit testing

## System Architecture

```text
                       Input
                 Image / Video
                       |
             +---------+---------+
             |                   |
             v                   v
     Custom YOLOv8n       Pretrained YOLOv8n
     Parking Detector      Vehicle Detector
             |                   |
       +-----+-----+       +-----+---------+
       |           |       |     |    |    |
       v           v       v     v    v    v
     Empty      Occupied   Car  Bike  Bus Truck
       |           |
       +-----+-----+
             |
             v
      Parking Analytics
             |
             +-------------------+
                                 |
                                 v
                       Combined Analytics
                                 |
                                 v
                       Streamlit Dashboard
```

## Technology Stack

* Python 3.11
* PyTorch
* Ultralytics YOLOv8
* OpenCV
* Streamlit
* Pandas
* Pillow
* Pytest
* uv

## Dataset

The parking occupancy model is trained on a YOLO-format parking dataset containing two classes:

```text
0 — space-empty
1 — space-occupied
```

Expected dataset structure:

```text
data/dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

The dataset is intentionally excluded from GitHub because of its size.

## Model Training

The project fine-tunes a COCO-pretrained YOLOv8n model for parking occupancy detection.

```bash
uv run python -m scripts.train_model
```

The best model checkpoint is generated at:

```text
outputs/training/smartpark_yolov8n/weights/best.pt
```

## Model Evaluation

Run:

```bash
uv run python -m scripts.evaluate_model
```

Evaluation metrics include:

| Metric    |       Result |
| --------- | -----------: |
| Precision | ACTUAL_VALUE |
| Recall    | ACTUAL_VALUE |
| mAP@50    | ACTUAL_VALUE |
| mAP@50-95 | ACTUAL_VALUE |

Replace these values with the actual results obtained from the trained model.

## Image Prediction

Run:

```bash
uv run python -m scripts.predict_image "path/to/image.jpg"
```

## Combined Parking and Vehicle Analysis

Run:

```bash
uv run python -m scripts.combined_prediction "path/to/image.jpg"
```

The combined pipeline produces:

* total parking spaces
* occupied spaces
* available spaces
* occupancy percentage
* total detected vehicles
* car count
* motorcycle count
* bus count
* truck count

## Video Prediction

Run:

```bash
uv run python -m scripts.predict_video "path/to/video.mp4"
```

## Streamlit Dashboard

Start the application using:

```bash
uv run streamlit run app/ui/streamlit_app.py
```

The dashboard provides:

* image upload
* video upload
* annotated detections
* parking statistics
* vehicle statistics
* detection tables
* model evaluation metrics

## Dataset Validation

Before training:

```bash
uv run python -m scripts.validate_dataset
```

The validator checks:

* image integrity
* YOLO label format
* image-label matching
* class IDs
* normalized bounding-box coordinates
* train/validation/test splits
* class distribution

## Tests

Run:

```bash
uv run pytest
```

## Project Health Check

Run:

```bash
uv run python -m scripts.project_check
```

This checks the local dataset, trained model, evaluation metrics, CUDA status, and core project directories.

## Project Workflow

```text
Dataset
   |
   v
Validation
   |
   v
YOLOv8n Fine-Tuning
   |
   v
Model Evaluation
   |
   v
Parking Occupancy Detection
   |
   +----------------------+
   |                      |
   v                      v
Image Inference       Video Inference
   |
   v
Vehicle Detection
   |
   v
Parking Analytics
   |
   v
Streamlit Dashboard
```

## Limitations

* Parking-space counts may fluctuate when a parking space is temporarily missed by the detector.
* Performance depends on camera angle, lighting, occlusion, and similarity to the training dataset.
* Vehicle and parking-space detectors operate independently, so detected vehicle count does not always equal occupied parking-space count.
* Video processing performance depends on available GPU hardware.
* The current video pipeline prioritizes parking occupancy detection to reduce computational load.

## Future Improvements

* Temporal smoothing for stable video occupancy counts
* Parking-slot tracking
* Fixed parking-region configuration
* CCTV/RTSP integration
* Cloud deployment
* Database-backed occupancy history
* Parking availability forecasting
* Notification system for parking capacity
* License-plate recognition as a separate optional module

## Project Purpose

SmartPark AI demonstrates an end-to-end computer vision workflow including dataset preparation, transfer learning, object detection, evaluation, inference, analytics, testing, and interactive application development.

The project was designed as a practical portfolio project for computer vision and AI engineering roles.
