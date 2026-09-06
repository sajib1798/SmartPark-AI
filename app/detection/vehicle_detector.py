from typing import Any

import numpy as np
from ultralytics import YOLO

from app.config import settings
from app.detection.model import get_device


# ============================================================
# LOAD VEHICLE MODEL
# ============================================================

def load_vehicle_model() -> YOLO:
    """
    Load locally stored pretrained YOLOv8n
    for vehicle detection.

    Expected location:

    E:/SmartPark_AI/models/yolov8n.pt
    """

    model_path = settings.VEHICLE_MODEL

    if not model_path.exists():

        raise FileNotFoundError(
            "\nPretrained vehicle model not found.\n"
            f"Expected location:\n{model_path}\n\n"
            "Place yolov8n.pt inside the models folder."
        )

    return YOLO(
        str(model_path)
    )


# ============================================================
# EXTRACT VEHICLE DETECTIONS
# ============================================================

def extract_vehicle_detections(
    result,
) -> list[dict[str, Any]]:
    """
    Keep only supported vehicle classes
    from YOLO COCO detections.
    """

    detections = []

    if result.boxes is None:
        return detections

    class_names = result.names

    for box in result.boxes:

        class_id = int(
            box.cls[0].item()
        )

        class_name = (
            class_names[class_id]
        )

        # Ignore non-vehicle classes
        if (
            class_name
            not in settings.VEHICLE_CLASS_NAMES
        ):
            continue

        confidence = float(
            box.conf[0].item()
        )

        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .tolist()
        )

        detections.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(
                    confidence,
                    4,
                ),
                "bbox": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                },
            }
        )

    return detections


# ============================================================
# COUNT VEHICLES
# ============================================================

def count_vehicles(
    detections: list[dict[str, Any]],
) -> dict[str, int]:
    """
    Count detected vehicle classes.
    """

    counts = {
        "car": 0,
        "motorcycle": 0,
        "bus": 0,
        "truck": 0,
    }

    for detection in detections:

        class_name = (
            detection["class_name"]
        )

        if class_name in counts:
            counts[class_name] += 1

    return counts


# ============================================================
# VEHICLE PREDICTION
# ============================================================

def predict_vehicles(
    frame: np.ndarray,
    model: YOLO | None = None,
) -> dict[str, Any]:
    """
    Detect vehicles in one OpenCV frame.
    """

    if model is None:

        model = load_vehicle_model()

    results = model.predict(
        source=frame,
        conf=(
            settings
            .VEHICLE_CONFIDENCE_THRESHOLD
        ),
        iou=(
            settings
            .VEHICLE_IOU_THRESHOLD
        ),
        imgsz=settings.IMAGE_SIZE,
        device=get_device(),
        verbose=False,
    )

    result = results[0]

    detections = (
        extract_vehicle_detections(
            result
        )
    )

    counts = count_vehicles(
        detections
    )

    total_vehicles = sum(
        counts.values()
    )

    return {
        "detections": detections,
        "counts": counts,
        "total_vehicles": total_vehicles,
    }