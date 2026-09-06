import argparse
import json
from pathlib import Path

from app.config import settings

from app.detection.combined_inference import (
    predict_combined_image,
)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Run SmartPark AI combined "
            "parking and vehicle detection."
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

    settings.COMBINED_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_image = (
        settings.COMBINED_OUTPUT_DIR
        / f"combined_{image_path.name}"
    )

    result = predict_combined_image(
        image_path=image_path,
        output_path=output_image,
    )

    summary = result["summary"]

    parking = summary["parking"]

    vehicles = summary["vehicles"]

    print()

    print("=" * 70)

    print(
        "SmartPark AI - Combined Analysis"
    )

    print("=" * 70)

    print()

    print("PARKING")

    print(
        f"Total spaces : "
        f"{parking['total_spaces']}"
    )

    print(
        f"Occupied     : "
        f"{parking['occupied_spaces']}"
    )

    print(
        f"Available    : "
        f"{parking['empty_spaces']}"
    )

    print(
        f"Occupancy    : "
        f"{parking['occupancy_percentage']}%"
    )

    print()

    print("VEHICLES")

    print(
        f"Total vehicles : "
        f"{vehicles['total_vehicles']}"
    )

    for class_name, count in (
        vehicles[
            "vehicle_counts"
        ].items()
    ):

        print(
            f"{class_name:<12}: {count}"
        )

    # --------------------------------------------------------
    # JSON report
    # --------------------------------------------------------

    report_path = (
        settings.COMBINED_OUTPUT_DIR
        / f"{image_path.stem}_combined.json"
    )

    report_data = {
        "image": image_path.name,
        "summary": summary,
        "parking_detections": (
            result[
                "parking_detections"
            ]
        ),
        "vehicle_detections": (
            result[
                "vehicle_detections"
            ]
        ),
    }

    report_path.write_text(
        json.dumps(
            report_data,
            indent=4,
        ),
        encoding="utf-8",
    )

    print()

    print(
        "Annotated image:"
    )

    print(
        output_image
    )

    print()

    print(
        "JSON report:"
    )

    print(
        report_path
    )


if __name__ == "__main__":
    main()