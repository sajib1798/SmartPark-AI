import json

from app.config import settings

from app.detection.model import (
    get_device,
    get_device_name,
    load_trained_model,
)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Evaluate the trained SmartPark model.
    """

    print("=" * 70)

    print(
        "SmartPark AI - Model Evaluation"
    )

    print("=" * 70)

    if not settings.BEST_MODEL_FILE.exists():

        raise FileNotFoundError(
            "best.pt not found. "
            "Train the model first.\n"
            f"Expected location: "
            f"{settings.BEST_MODEL_FILE}"
        )

    device = get_device()

    print(
        f"Evaluation device: "
        f"{get_device_name()}"
    )

    print(
        f"Model: "
        f"{settings.BEST_MODEL_FILE}"
    )

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    model = load_trained_model()

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    metrics = model.val(
        data=str(
            settings.DATA_YAML_FILE
        ),

        split="test",

        imgsz=settings.IMAGE_SIZE,

        batch=settings.BATCH_SIZE,

        device=device,

        project=str(
            settings.OUTPUT_DIR / "evaluation"
        ),

        name="smartpark_test",

        plots=True,

        save_json=False,
    )

    # --------------------------------------------------------
    # Extract metrics
    # --------------------------------------------------------

    precision = float(
        metrics.box.mp
    )

    recall = float(
        metrics.box.mr
    )

    map50 = float(
        metrics.box.map50
    )

    map50_95 = float(
        metrics.box.map
    )

    results = {
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map50_95": map50_95,
    }

    # --------------------------------------------------------
    # Save evaluation report
    # --------------------------------------------------------

    evaluation_file = (
        settings.OUTPUT_DIR
        / "evaluation_metrics.json"
    )

    evaluation_file.write_text(
        json.dumps(
            results,
            indent=4,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Print metrics
    # --------------------------------------------------------

    print()

    print("Evaluation Results")

    print("-" * 70)

    print(
        f"Precision  : {precision:.4f}"
    )

    print(
        f"Recall     : {recall:.4f}"
    )

    print(
        f"mAP@50     : {map50:.4f}"
    )

    print(
        f"mAP@50-95  : {map50_95:.4f}"
    )

    print()

    print(
        "Evaluation report saved to:"
    )

    print(
        evaluation_file
    )


if __name__ == "__main__":
    main()