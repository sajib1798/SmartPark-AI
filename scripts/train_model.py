from app.config import settings

from app.detection.model import (
    get_device,
    get_device_name,
    load_pretrained_model,
)


# ============================================================
# VALIDATE REQUIRED FILES
# ============================================================

def validate_training_setup() -> None:
    """
    Check that the required dataset files
    exist before starting training.
    """

    if not settings.DATA_YAML_FILE.exists():

        raise FileNotFoundError(
            "Dataset YAML file not found: "
            f"{settings.DATA_YAML_FILE}"
        )

    if not settings.TRAIN_IMAGES_DIR.exists():

        raise FileNotFoundError(
            "Training images directory not found: "
            f"{settings.TRAIN_IMAGES_DIR}"
        )

    if not settings.TRAIN_LABELS_DIR.exists():

        raise FileNotFoundError(
            "Training labels directory not found: "
            f"{settings.TRAIN_LABELS_DIR}"
        )

    if not settings.VALID_IMAGES_DIR.exists():

        raise FileNotFoundError(
            "Validation images directory not found: "
            f"{settings.VALID_IMAGES_DIR}"
        )

    if not settings.VALID_LABELS_DIR.exists():

        raise FileNotFoundError(
            "Validation labels directory not found: "
            f"{settings.VALID_LABELS_DIR}"
        )


# ============================================================
# TRAIN
# ============================================================

def main() -> None:
    """
    Fine-tune YOLOv8n on the
    SmartPark parking dataset.
    """

    print("=" * 70)

    print(
        "SmartPark AI - Phase 2 YOLO Training"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    validate_training_setup()

    # --------------------------------------------------------
    # Select device
    # --------------------------------------------------------

    device = get_device()

    device_name = get_device_name()

    print(
        f"Training device: {device_name}"
    )

    print(
        f"Base model: "
        f"{settings.PRETRAINED_MODEL}"
    )

    print(
        f"Dataset: "
        f"{settings.DATA_YAML_FILE}"
    )

    print(
        f"Epochs: {settings.EPOCHS}"
    )

    print(
        f"Image size: {settings.IMAGE_SIZE}"
    )

    print(
        f"Batch size: {settings.BATCH_SIZE}"
    )

    print()

    # --------------------------------------------------------
    # Load pretrained model
    # --------------------------------------------------------

    model = load_pretrained_model()

    # --------------------------------------------------------
    # Fine-tune model
    # --------------------------------------------------------

    model.train(
        data=str(
            settings.DATA_YAML_FILE
        ),

        epochs=settings.EPOCHS,

        imgsz=settings.IMAGE_SIZE,

        batch=settings.BATCH_SIZE,

        device=device,

        workers=settings.WORKERS,

        project=str(
            settings.TRAINING_OUTPUT_DIR
        ),

        name=settings.TRAINING_RUN_NAME,

        exist_ok=True,

        pretrained=True,

        plots=True,

        save=True,

        verbose=True,
    )

    # --------------------------------------------------------
    # Training completed
    # --------------------------------------------------------

    print()

    print("=" * 70)

    print(
        "Training completed."
    )

    print(
        "Expected best model:"
    )

    print(
        settings.BEST_MODEL_FILE
    )

    print("=" * 70)


if __name__ == "__main__":
    main()