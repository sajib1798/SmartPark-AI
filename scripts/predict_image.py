import argparse
import json
from pathlib import Path

from app.config import settings
from app.detection.inference import (
    predict_image,
)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Run SmartPark AI inference "
            "on one parking image."
        )
    )

    parser.add_argument(
        "image",
        type=str,
        help="Path to parking image",
    )

    args = parser.parse_args()

    image_path = Path(
        args.image
    ).resolve()

    # --------------------------------------------------------
    # Output location
    # --------------------------------------------------------

    output_dir = (
        settings.PREDICTION_OUTPUT_DIR
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_image = (
        output_dir
        / f"predicted_{image_path.name}"
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    result = predict_image(
        image_path=image_path,
        output_path=output_image,
    )

    statistics = result[
        "statistics"
    ]

    # --------------------------------------------------------
    # Print statistics
    # --------------------------------------------------------

    print()

    print("=" * 70)

    print(
        "SmartPark AI - Image Prediction"
    )

    print("=" * 70)

    print(
        f"Image: {image_path.name}"
    )

    print(
        f"Total spaces: "
        f"{statistics['total_spaces']}"
    )

    print(
        f"Occupied: "
        f"{statistics['occupied_spaces']}"
    )

    print(
        f"Available: "
        f"{statistics['empty_spaces']}"
    )

    print(
        f"Occupancy: "
        f"{statistics['occupancy_percentage']}%"
    )

    print(
        f"Availability: "
        f"{statistics['availability_percentage']}%"
    )

    print()

    print(
        f"Annotated image saved to:"
    )

    print(
        output_image
    )

    # --------------------------------------------------------
    # Save JSON result
    # --------------------------------------------------------

    report_file = (
        output_dir
        / f"{image_path.stem}_result.json"
    )

    json_result = {
        "image": image_path.name,
        "statistics": statistics,
        "detections": result[
            "detections"
        ],
    }

    report_file.write_text(
        json.dumps(
            json_result,
            indent=4,
        ),
        encoding="utf-8",
    )

    print()

    print(
        f"Detection report saved to:"
    )

    print(
        report_file
    )


if __name__ == "__main__":
    main()