import argparse
from pathlib import Path

from app.config import settings

from app.detection.model import (
    get_device,
    load_trained_model,
)


# ============================================================
# PREDICT
# ============================================================

def predict_image(
    image_path: Path,
) -> None:
    """
    Run parking detection on one image.
    """

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    model = load_trained_model()

    model.predict(
        source=str(image_path),

        conf=settings.CONFIDENCE_THRESHOLD,

        imgsz=settings.IMAGE_SIZE,

        device=get_device(),

        save=True,

        project=str(
            settings.PREDICTION_OUTPUT_DIR
        ),

        name="image_prediction",

        exist_ok=True,
    )


# ============================================================
# COMMAND LINE
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Run SmartPark AI prediction "
            "on a parking image."
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

    predict_image(
        image_path
    )

    print()

    print(
        "Prediction completed."
    )

    print(
        "Results saved to:"
    )

    print(
        settings.PREDICTION_OUTPUT_DIR
        / "image_prediction"
    )


if __name__ == "__main__":
    main()