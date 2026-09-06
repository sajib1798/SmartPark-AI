import argparse
from pathlib import Path

import cv2

from app.config import settings
from app.detection.inference import (
    predict_frame,
)
from app.detection.model import (
    load_trained_model,
)


# ============================================================
# PROCESS VIDEO
# ============================================================

def process_video(
    video_path: Path,
) -> Path:
    """
    Run SmartPark detection on a video.
    """

    if not video_path.exists():

        raise FileNotFoundError(
            f"Video not found: "
            f"{video_path}"
        )

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():

        raise RuntimeError(
            f"Could not open video: "
            f"{video_path}"
        )

    # --------------------------------------------------------
    # Get video information
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    settings.VIDEO_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        settings.VIDEO_OUTPUT_DIR
        / f"predicted_{video_path.stem}.mp4"
    )

    codec = cv2.VideoWriter_fourcc(
        *settings.VIDEO_CODEC
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

    # --------------------------------------------------------
    # Load model once
    # --------------------------------------------------------

    model = load_trained_model()

    frame_number = 0

    # --------------------------------------------------------
    # Process frames
    # --------------------------------------------------------

    while True:

        success, frame = capture.read()

        if not success:
            break

        frame_number += 1

        result = predict_frame(
            frame=frame,
            model=model,
        )

        annotated_frame = result[
            "annotated_frame"
        ]

        statistics = result[
            "statistics"
        ]

        # ----------------------------------------------------
        # Add statistics to frame
        # ----------------------------------------------------

        text = (
            f"Total: "
            f"{statistics['total_spaces']} | "
            f"Occupied: "
            f"{statistics['occupied_spaces']} | "
            f"Available: "
            f"{statistics['empty_spaces']} | "
            f"Occupancy: "
            f"{statistics['occupancy_percentage']}%"
        )

        cv2.putText(
            annotated_frame,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(
            annotated_frame
        )

        if frame_number % 30 == 0:

            print(
                f"Processed "
                f"{frame_number} frames"
            )

    capture.release()

    writer.release()

    return output_path


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Run SmartPark AI inference "
            "on parking video."
        )
    )

    parser.add_argument(
        "video",
        type=str,
        help="Path to parking video",
    )

    args = parser.parse_args()

    video_path = Path(
        args.video
    ).resolve()

    print("=" * 70)

    print(
        "SmartPark AI - Video Prediction"
    )

    print("=" * 70)

    output_path = process_video(
        video_path
    )

    print()

    print(
        "Video processing completed."
    )

    print(
        "Output:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()