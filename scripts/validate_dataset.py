import json

from app.config import settings

from app.data.dataset_stats import (
    create_split_statistics,
)

from app.data.dataset_validator import (
    validate_split,
)


# ============================================================
# DATASET CLASSES
# ============================================================

CLASS_NAMES = [
    "space-empty",
    "space-occupied",
]


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Validate the complete parking dataset.
    """

    print("=" * 70)

    print(
        "SmartPark AI - Phase 1 Dataset Validation"
    )

    print("=" * 70)

    num_classes = len(CLASS_NAMES)

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_result = validate_split(
        split_name="train",
        image_dir=settings.TRAIN_IMAGES_DIR,
        label_dir=settings.TRAIN_LABELS_DIR,
        num_classes=num_classes,
    )

    # --------------------------------------------------------
    # VALID
    # --------------------------------------------------------

    valid_result = validate_split(
        split_name="valid",
        image_dir=settings.VALID_IMAGES_DIR,
        label_dir=settings.VALID_LABELS_DIR,
        num_classes=num_classes,
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_result = validate_split(
        split_name="test",
        image_dir=settings.TEST_IMAGES_DIR,
        label_dir=settings.TEST_LABELS_DIR,
        num_classes=num_classes,
    )

    validation_results = {
        "train": train_result,
        "valid": valid_result,
        "test": test_result,
    }

    # --------------------------------------------------------
    # DATASET STATISTICS
    # --------------------------------------------------------

    statistics = {}

    split_directories = {
        "train": settings.TRAIN_LABELS_DIR,
        "valid": settings.VALID_LABELS_DIR,
        "test": settings.TEST_LABELS_DIR,
    }

    for split_name, label_dir in (
        split_directories.items()
    ):

        if label_dir.exists():

            statistics[split_name] = (
                create_split_statistics(
                    label_dir=label_dir,
                    class_names=CLASS_NAMES,
                )
            )

    # --------------------------------------------------------
    # CREATE REPORT
    # --------------------------------------------------------

    report = {
        "project": settings.APP_NAME,
        "phase": 1,
        "classes": CLASS_NAMES,
        "validation": validation_results,
        "statistics": statistics,
    }

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    settings.OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    settings.DATASET_REPORT_FILE.write_text(
        json.dumps(
            report,
            indent=4,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------------

    print()

    for split_name, result in (
        validation_results.items()
    ):

        print(
            f"{split_name.upper():<10}"
            f"Images: {result['image_count']:<6}"
            f"Labels: {result['label_count']:<6}"
            f"Valid: {result['valid']}"
        )

    # --------------------------------------------------------
    # PRINT CLASS COUNTS
    # --------------------------------------------------------

    print()

    print("Class Statistics")

    print("-" * 70)

    for split_name, split_stats in (
        statistics.items()
    ):

        print(
            f"\n{split_name.upper()}"
        )

        for class_name, count in (
            split_stats["class_counts"].items()
        ):

            print(
                f"  {class_name:<20}: {count}"
            )

    # --------------------------------------------------------
    # CHECK COMPLETE DATASET
    # --------------------------------------------------------

    dataset_valid = all(
        result["valid"]
        for result in validation_results.values()
    )

    print()

    if dataset_valid:

        print(
            "Dataset validation completed successfully."
        )

        print(
            "Dataset is ready for Phase 2."
        )

    else:

        print(
            "Dataset contains validation errors."
        )

        print()

        for split_name, result in (
            validation_results.items()
        ):

            for error in result["errors"][:20]:

                print(
                    f"[{split_name}] {error}"
                )

    print()

    print(
        "Dataset report saved to:"
    )

    print(
        settings.DATASET_REPORT_FILE
    )


if __name__ == "__main__":
    main()