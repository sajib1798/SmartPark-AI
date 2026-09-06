# ============================================================
# IMPORT PATH FIX
# ============================================================

import sys
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# IMPORTS
# ============================================================

import json
import tempfile

import cv2
import pandas as pd
import streamlit as st

from app.config import settings

from app.detection.combined_inference import (
    predict_combined_image,
)

from app.detection.inference import (
    predict_frame,
)

from app.detection.model import (
    load_trained_model,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title=settings.UI_TITLE,
    page_icon="🚗",
    layout="wide",
)


# ============================================================
# LOAD EVALUATION METRICS
# ============================================================

def load_evaluation_metrics() -> dict:

    file_path = (
        settings.EVALUATION_METRICS_FILE
    )

    if not file_path.exists():
        return {}

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    except Exception:
        return {}


# ============================================================
# MODEL STATUS
# ============================================================

def parking_model_available() -> bool:

    return (
        settings.BEST_MODEL_FILE.exists()
    )


def vehicle_model_available() -> bool:

    return (
        settings.VEHICLE_MODEL.exists()
    )


# ============================================================
# TEMP FILE
# ============================================================

def save_uploaded_file(
    uploaded_file,
) -> Path:

    settings.TEMP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = Path(
        uploaded_file.name
    ).suffix

    temp_file = (
        tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
            dir=settings.TEMP_DIR,
        )
    )

    temp_file.write(
        uploaded_file.getvalue()
    )

    temp_file.close()

    return Path(
        temp_file.name
    )


# ============================================================
# PARKING METRICS
# ============================================================

def show_parking_metrics(
    summary: dict,
) -> None:

    parking = summary[
        "parking"
    ]

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    with col1:

        st.metric(
            "Total Spaces",
            parking[
                "total_spaces"
            ],
        )

    with col2:

        st.metric(
            "Occupied",
            parking[
                "occupied_spaces"
            ],
        )

    with col3:

        st.metric(
            "Available",
            parking[
                "empty_spaces"
            ],
        )

    with col4:

        st.metric(
            "Occupancy",
            (
                f"{parking['occupancy_percentage']}%"
            ),
        )


# ============================================================
# VEHICLE METRICS
# ============================================================

def show_vehicle_metrics(
    summary: dict,
) -> None:

    vehicles = summary[
        "vehicles"
    ]

    counts = vehicles[
        "vehicle_counts"
    ]

    st.subheader(
        "Vehicle Statistics"
    )

    col1, col2, col3, col4, col5 = (
        st.columns(5)
    )

    with col1:

        st.metric(
            "Total Vehicles",
            vehicles[
                "total_vehicles"
            ],
        )

    with col2:

        st.metric(
            "Cars",
            counts.get(
                "car",
                0,
            ),
        )

    with col3:

        st.metric(
            "Motorcycles",
            counts.get(
                "motorcycle",
                0,
            ),
        )

    with col4:

        st.metric(
            "Buses",
            counts.get(
                "bus",
                0,
            ),
        )

    with col5:

        st.metric(
            "Trucks",
            counts.get(
                "truck",
                0,
            ),
        )


# ============================================================
# DETECTION TABLE
# ============================================================

def create_detection_dataframe(
    detections: list[dict],
) -> pd.DataFrame:

    rows = []

    for detection in detections:

        bbox = detection[
            "bbox"
        ]

        rows.append(
            {
                "Class":
                    detection[
                        "class_name"
                    ],

                "Confidence":
                    detection[
                        "confidence"
                    ],

                "X1": bbox["x1"],
                "Y1": bbox["y1"],
                "X2": bbox["x2"],
                "Y2": bbox["y2"],
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def image_mode() -> None:

    st.header(
        "Image Analysis"
    )

    st.write(
        "Upload a parking-lot image to "
        "detect parking occupancy and vehicles."
    )

    uploaded_file = (
        st.file_uploader(
            "Upload Parking Image",
            type=[
                "jpg",
                "jpeg",
                "png",
                "bmp",
            ],
            key="parking_image",
        )
    )

    if uploaded_file is None:
        return

    # --------------------------------------------------------
    # ORIGINAL IMAGE
    # --------------------------------------------------------

    st.subheader(
        "Original Image"
    )

    st.image(
        uploaded_file,
        width="stretch",
    )

    # --------------------------------------------------------
    # CHECK MODELS
    # --------------------------------------------------------

    if not parking_model_available():

        st.error(
            "Custom parking model "
            "best.pt is missing."
        )

        st.write(
            "Expected:"
        )

        st.code(
            str(
                settings.BEST_MODEL_FILE
            )
        )

        return

    if not vehicle_model_available():

        st.error(
            "Pretrained vehicle model "
            "yolov8n.pt is missing."
        )

        st.write(
            "Place the file here:"
        )

        st.code(
            str(
                settings.VEHICLE_MODEL
            )
        )

        return

    # --------------------------------------------------------
    # RUN ANALYSIS
    # --------------------------------------------------------

    if st.button(
        "Run SmartPark Analysis",
        type="primary",
    ):

        image_path = (
            save_uploaded_file(
                uploaded_file
            )
        )

        with st.spinner(
            "Running parking and "
            "vehicle detection..."
        ):

            try:

                result = (
                    predict_combined_image(
                        image_path=image_path
                    )
                )

            except Exception as error:

                st.error(
                    f"Prediction failed: "
                    f"{error}"
                )

                return

        st.success(
            "Analysis completed successfully."
        )

        summary = result[
            "summary"
        ]

        # ----------------------------------------------------
        # PARKING STATS
        # ----------------------------------------------------

        st.subheader(
            "Parking Statistics"
        )

        show_parking_metrics(
            summary
        )

        occupancy = (
            summary[
                "parking"
            ][
                "occupancy_percentage"
            ]
            / 100
        )

        occupancy = min(
            max(
                occupancy,
                0.0,
            ),
            1.0,
        )

        st.progress(
            occupancy
        )

        # ----------------------------------------------------
        # VEHICLE STATS
        # ----------------------------------------------------

        show_vehicle_metrics(
            summary
        )

        # ----------------------------------------------------
        # ANNOTATED IMAGE
        # ----------------------------------------------------

        st.subheader(
            "Detection Result"
        )

        annotated_frame = (
            result[
                "annotated_frame"
            ]
        )

        annotated_rgb = cv2.cvtColor(
            annotated_frame,
            cv2.COLOR_BGR2RGB,
        )

        st.image(
            annotated_rgb,
            width="stretch",
        )

        # ----------------------------------------------------
        # DETECTION TABLES
        # ----------------------------------------------------

        st.subheader(
            "Detection Details"
        )

        parking_tab, vehicle_tab = (
            st.tabs(
                [
                    "Parking Spaces",
                    "Vehicles",
                ]
            )
        )

        with parking_tab:

            parking_df = (
                create_detection_dataframe(
                    result[
                        "parking_detections"
                    ]
                )
            )

            if parking_df.empty:

                st.info(
                    "No parking detections."
                )

            else:

                st.dataframe(
                    parking_df,
                    width="stretch",
                    hide_index=True,
                )

        with vehicle_tab:

            vehicle_df = (
                create_detection_dataframe(
                    result[
                        "vehicle_detections"
                    ]
                )
            )

            if vehicle_df.empty:

                st.info(
                    "No vehicle detections."
                )

            else:

                st.dataframe(
                    vehicle_df,
                    width="stretch",
                    hide_index=True,
                )


# ============================================================
# VIDEO PROCESSING
# ============================================================

def process_video(
    video_path: Path,
) -> Path:

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():

        raise RuntimeError(
            "Could not open video."
        )

    width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 25.0

    frame_count = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    settings.VIDEO_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        settings.VIDEO_OUTPUT_DIR
        / f"processed_{video_path.stem}.mp4"
    )

    codec = (
        cv2.VideoWriter_fourcc(
            *settings.VIDEO_CODEC
        )
    )

    writer = cv2.VideoWriter(
        str(output_path),
        codec,
        fps,
        (
            width,
            height,
        ),
    )

    if not writer.isOpened():

        capture.release()

        raise RuntimeError(
            "Could not create output video."
        )

    # Load custom model once
    model = (
        load_trained_model()
    )

    progress_bar = (
        st.progress(0.0)
    )

    status_text = (
        st.empty()
    )

    frame_number = 0

    while True:

        success, frame = (
            capture.read()
        )

        if not success:
            break

        frame_number += 1

        result = predict_frame(
            frame=frame,
            model=model,
        )

        annotated_frame = (
            result[
                "annotated_frame"
            ]
        )

        stats = result[
            "statistics"
        ]

        display_text = (
            f"Total: "
            f"{stats['total_spaces']} | "
            f"Occupied: "
            f"{stats['occupied_spaces']} | "
            f"Available: "
            f"{stats['empty_spaces']} | "
            f"Occupancy: "
            f"{stats['occupancy_percentage']}%"
        )

        cv2.putText(
            annotated_frame,
            display_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(
            annotated_frame
        )

        if frame_count > 0:

            progress = (
                frame_number
                / frame_count
            )

            progress_bar.progress(
                min(
                    progress,
                    1.0,
                )
            )

        status_text.write(
            f"Processing frame "
            f"{frame_number}"
        )

    capture.release()

    writer.release()

    progress_bar.progress(
        1.0
    )

    status_text.write(
        "Video processing complete."
    )

    return output_path


# ============================================================
# VIDEO MODE
# ============================================================

def video_mode() -> None:

    st.header(
        "Video Analysis"
    )

    st.write(
        "Upload a parking video for "
        "parking occupancy detection."
    )

    uploaded_video = (
        st.file_uploader(
            "Upload Parking Video",
            type=[
                "mp4",
                "avi",
                "mov",
                "mkv",
            ],
            key="parking_video",
        )
    )

    if uploaded_video is None:
        return

    st.subheader(
        "Original Video"
    )

    st.video(
        uploaded_video
    )

    if not parking_model_available():

        st.error(
            "Custom best.pt model "
            "is missing."
        )

        return

    if st.button(
        "Process Video",
        type="primary",
    ):

        video_path = (
            save_uploaded_file(
                uploaded_video
            )
        )

        try:

            output_path = (
                process_video(
                    video_path
                )
            )

        except Exception as error:

            st.error(
                f"Video processing "
                f"failed: {error}"
            )

            return

        st.success(
            "Video processing completed."
        )

        st.subheader(
            "Processed Video"
        )

        with open(
            output_path,
            "rb",
        ) as video_file:

            st.video(
                video_file.read()
            )


# ============================================================
# MODEL INFORMATION
# ============================================================

def model_information() -> None:

    st.header(
        "Model Information"
    )

    col1, col2, col3 = (
        st.columns(3)
    )

    with col1:

        st.metric(
            "Parking Model",
            (
                "Ready"
                if parking_model_available()
                else "Missing"
            ),
        )

    with col2:

        st.metric(
            "Vehicle Model",
            (
                "Ready"
                if vehicle_model_available()
                else "Missing"
            ),
        )

    with col3:

        st.metric(
            "Image Size",
            settings.IMAGE_SIZE,
        )

    # --------------------------------------------------------
    # MODEL PATHS
    # --------------------------------------------------------

    st.subheader(
        "Parking Model"
    )

    st.code(
        str(
            settings.BEST_MODEL_FILE
        )
    )

    st.subheader(
        "Vehicle Model"
    )

    st.code(
        str(
            settings.VEHICLE_MODEL
        )
    )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    st.subheader(
        "Evaluation Metrics"
    )

    metrics = (
        load_evaluation_metrics()
    )

    if not metrics:

        st.info(
            "Evaluation metrics "
            "are not available yet."
        )

        return

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    with col1:

        st.metric(
            "Precision",
            f"{metrics.get('precision', 0):.3f}",
        )

    with col2:

        st.metric(
            "Recall",
            f"{metrics.get('recall', 0):.3f}",
        )

    with col3:

        st.metric(
            "mAP@50",
            f"{metrics.get('map50', 0):.3f}",
        )

    with col4:

        st.metric(
            "mAP@50-95",
            f"{metrics.get('map50_95', 0):.3f}",
        )


# ============================================================
# ABOUT
# ============================================================

def about_page() -> None:

    st.header(
        "About SmartPark AI"
    )

    st.write(
        """
        SmartPark AI is an end-to-end
        computer vision system for parking
        occupancy and vehicle detection.
        """
    )

    st.subheader(
        "Features"
    )

    st.write(
        """
        - Empty parking-space detection
        - Occupied parking-space detection
        - Parking occupancy percentage
        - Car detection
        - Motorcycle detection
        - Bus detection
        - Truck detection
        - Image inference
        - Video inference
        - Model evaluation dashboard
        """
    )

    st.subheader(
        "Technology Stack"
    )

    st.write(
        """
        Python, PyTorch, YOLOv8,
        OpenCV, Streamlit, Pandas,
        Pillow and Pytest.
        """
    )


# ============================================================
# SIDEBAR
# ============================================================

def sidebar() -> str:

    with st.sidebar:

        st.title(
            "🚗 SmartPark AI"
        )

        st.caption(
            "Parking Occupancy "
            "& Vehicle Detection"
        )

        st.divider()

        page = st.radio(
            "Navigation",
            [
                "Image Analysis",
                "Video Analysis",
                "Model Information",
                "About",
            ],
        )

        st.divider()

        st.write(
            "### Model Status"
        )

        if parking_model_available():

            st.success(
                "Parking model ready"
            )

        else:

            st.error(
                "Parking model missing"
            )

        if vehicle_model_available():

            st.success(
                "Vehicle model ready"
            )

        else:

            st.warning(
                "Vehicle model missing"
            )

        st.caption(
            f"Version "
            f"{settings.VERSION}"
        )

    return page


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    st.title(
        "🚗 SmartPark AI"
    )

    st.subheader(
        settings.UI_SUBTITLE
    )

    st.write(
        "AI-powered parking occupancy "
        "and vehicle monitoring system."
    )

    st.divider()

    selected_page = (
        sidebar()
    )

    if (
        selected_page
        == "Image Analysis"
    ):

        image_mode()

    elif (
        selected_page
        == "Video Analysis"
    ):

        video_mode()

    elif (
        selected_page
        == "Model Information"
    ):

        model_information()

    elif (
        selected_page
        == "About"
    ):

        about_page()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()