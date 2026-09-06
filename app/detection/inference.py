from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from app.config import settings
from app.detection.analytics import (
    calculate_occupancy_statistics,
)
from app.detection.model import (
    get_device,
    load_trained_model,
)


# ============================================================
# COUNT DETECTIONS
# ============================================================

def count_parking_detections(
    result,
) -> tuple[int, int]:
    """
    Count empty and occupied parking-space detections
    from one Ultralytics result object.
    """

    empty_count = 0
    occupied_count = 0

    if result.boxes is None:
        return empty_count, occupied_count

    class_names = result.names

    for box in result.boxes:

        class_id = int(
            box.cls[0].item()
        )

        class_name = class_names[
            class_id
        ]

        if (
            class_name
            == settings.EMPTY_CLASS_NAME
        ):

            empty_count += 1

        elif (
            class_name
            == settings.OCCUPIED_CLASS_NAME
        ):

            occupied_count += 1

    return (
        empty_count,
        occupied_count,
    )


# ============================================================
# EXTRACT DETECTIONS
# ============================================================

def extract_detections(
    result,
) -> list[dict[str, Any]]:
    """
    Convert YOLO detections into simple dictionaries.
    """

    detections = []

    if result.boxes is None:
        return detections

    class_names = result.names

    for box in result.boxes:

        class_id = int(
            box.cls[0].item()
        )

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
                "class_name": (
                    class_names[class_id]
                ),
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
# PREDICT NUMPY FRAME
# ============================================================

def predict_frame(
    frame: np.ndarray,
    model: YOLO | None = None,
) -> dict[str, Any]:
    """
    Run parking occupancy detection
    on one OpenCV image/frame.
    """

    if model is None:
        model = load_trained_model()

    results = model.predict(
        source=frame,
        conf=settings.CONFIDENCE_THRESHOLD,
        iou=settings.IOU_THRESHOLD,
        imgsz=settings.IMAGE_SIZE,
        device=get_device(),
        verbose=False,
    )

    result = results[0]

    empty_count, occupied_count = (
        count_parking_detections(
            result
        )
    )

    statistics = (
        calculate_occupancy_statistics(
            empty_count=empty_count,
            occupied_count=occupied_count,
        )
    )

    detections = extract_detections(
        result
    )

    annotated_frame = result.plot()

    return {
        "annotated_frame": annotated_frame,
        "detections": detections,
        "statistics": statistics,
    }


# ============================================================
# PREDICT IMAGE FILE
# ============================================================

def predict_image(
    image_path: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """
    Run inference on one image file.
    """

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: "
            f"{image_path}"
        )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise ValueError(
            f"Could not read image: "
            f"{image_path}"
        )

    model = load_trained_model()

    result = predict_frame(
        frame=image,
        model=model,
    )

    if output_path is not None:

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        success = cv2.imwrite(
            str(output_path),
            result["annotated_frame"],
        )

        if not success:

            raise RuntimeError(
                f"Could not save image: "
                f"{output_path}"
            )

    return result