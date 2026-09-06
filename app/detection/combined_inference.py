from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config import settings

from app.detection.analytics import (
    calculate_vehicle_statistics,
    create_parking_summary,
)

from app.detection.inference import (
    predict_frame,
)

from app.detection.model import (
    load_trained_model,
)

from app.detection.vehicle_detector import (
    load_vehicle_model,
    predict_vehicles,
)


# ============================================================
# DRAW VEHICLE BOXES
# ============================================================

def draw_vehicle_boxes(
    frame: np.ndarray,
    detections: list[dict[str, Any]],
) -> np.ndarray:
    """
    Draw vehicle bounding boxes on image.
    """

    output = frame.copy()

    for detection in detections:

        bbox = detection["bbox"]

        x1 = int(bbox["x1"])
        y1 = int(bbox["y1"])
        x2 = int(bbox["x2"])
        y2 = int(bbox["y2"])

        class_name = detection[
            "class_name"
        ]

        confidence = detection[
            "confidence"
        ]

        label = (
            f"{class_name} "
            f"{confidence:.2f}"
        )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2,
        )

        cv2.putText(
            output,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return output


# ============================================================
# DRAW ANALYTICS PANEL
# ============================================================

def draw_statistics_panel(
    frame: np.ndarray,
    summary: dict[str, Any],
) -> np.ndarray:
    """
    Draw SmartPark statistics
    directly onto a frame.
    """

    output = frame.copy()

    parking = summary["parking"]
    vehicles = summary["vehicles"]

    lines = [
        (
            f"Total Spaces: "
            f"{parking['total_spaces']}"
        ),
        (
            f"Occupied: "
            f"{parking['occupied_spaces']}"
        ),
        (
            f"Available: "
            f"{parking['empty_spaces']}"
        ),
        (
            f"Occupancy: "
            f"{parking['occupancy_percentage']}%"
        ),
        (
            f"Vehicles: "
            f"{vehicles['total_vehicles']}"
        ),
    ]

    y = 30

    for line in lines:

        cv2.putText(
            output,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        y += 30

    return output


# ============================================================
# COMBINED FRAME INFERENCE
# ============================================================

def predict_combined_frame(
    frame: np.ndarray,
    parking_model=None,
    vehicle_model=None,
) -> dict[str, Any]:
    """
    Run both parking occupancy detection
    and vehicle detection on one frame.
    """

    if parking_model is None:

        parking_model = (
            load_trained_model()
        )

    if vehicle_model is None:

        vehicle_model = (
            load_vehicle_model()
        )

    # --------------------------------------------------------
    # Parking occupancy model
    # --------------------------------------------------------

    parking_result = predict_frame(
        frame=frame,
        model=parking_model,
    )

    # --------------------------------------------------------
    # Vehicle model
    # --------------------------------------------------------

    vehicle_result = predict_vehicles(
        frame=frame,
        model=vehicle_model,
    )

    # --------------------------------------------------------
    # Vehicle analytics
    # --------------------------------------------------------

    vehicle_statistics = (
        calculate_vehicle_statistics(
            vehicle_result["counts"]
        )
    )

    # --------------------------------------------------------
    # Combined summary
    # --------------------------------------------------------

    summary = create_parking_summary(
        parking_statistics=(
            parking_result["statistics"]
        ),
        vehicle_statistics=(
            vehicle_statistics
        ),
    )

    # --------------------------------------------------------
    # Start with parking annotations
    # --------------------------------------------------------

    annotated_frame = (
        parking_result[
            "annotated_frame"
        ]
    )

    # --------------------------------------------------------
    # Add vehicle boxes
    # --------------------------------------------------------

    annotated_frame = (
        draw_vehicle_boxes(
            frame=annotated_frame,
            detections=(
                vehicle_result[
                    "detections"
                ]
            ),
        )
    )

    # --------------------------------------------------------
    # Add statistics
    # --------------------------------------------------------

    annotated_frame = (
        draw_statistics_panel(
            frame=annotated_frame,
            summary=summary,
        )
    )

    return {
        "annotated_frame": (
            annotated_frame
        ),
        "summary": summary,
        "parking_detections": (
            parking_result[
                "detections"
            ]
        ),
        "vehicle_detections": (
            vehicle_result[
                "detections"
            ]
        ),
    }


# ============================================================
# COMBINED IMAGE INFERENCE
# ============================================================

def predict_combined_image(
    image_path: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """
    Run complete SmartPark analysis
    on one image.
    """

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: "
            f"{image_path}"
        )

    frame = cv2.imread(
        str(image_path)
    )

    if frame is None:

        raise ValueError(
            f"Could not read image: "
            f"{image_path}"
        )

    result = predict_combined_frame(
        frame
    )

    if output_path is not None:

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        success = cv2.imwrite(
            str(output_path),
            result[
                "annotated_frame"
            ],
        )

        if not success:

            raise RuntimeError(
                f"Could not save "
                f"{output_path}"
            )

    return result