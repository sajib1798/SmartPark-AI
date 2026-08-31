from pathlib import Path
from typing import Any

from PIL import Image

from app.config import settings


YOLO_VALUES_PER_LINE = 5


# ============================================================
# GET IMAGE FILES
# ============================================================

def get_image_files(
    image_dir: Path,
) -> list[Path]:
    """
    Return all supported image files
    from a directory.
    """

    if not image_dir.exists():
        return []

    image_files = []

    for file_path in image_dir.iterdir():

        if (
            file_path.is_file()
            and file_path.suffix.lower()
            in settings.IMAGE_EXTENSIONS
        ):
            image_files.append(file_path)

    return sorted(image_files)


# ============================================================
# GET LABEL FILES
# ============================================================

def get_label_files(
    label_dir: Path,
) -> list[Path]:
    """
    Return all YOLO label files.
    """

    if not label_dir.exists():
        return []

    label_files = []

    for file_path in label_dir.iterdir():

        if (
            file_path.is_file()
            and file_path.suffix.lower() == ".txt"
        ):
            label_files.append(file_path)

    return sorted(label_files)


# ============================================================
# VALIDATE IMAGE
# ============================================================

def validate_image(
    image_path: Path,
) -> tuple[bool, str]:
    """
    Check whether an image can be opened.
    """

    try:

        with Image.open(image_path) as image:
            image.verify()

        return True, ""

    except Exception as error:

        return (
            False,
            f"Invalid image {image_path.name}: {error}",
        )


# ============================================================
# CHECK NORMALIZED COORDINATE
# ============================================================

def is_normalized_coordinate(
    value: float,
) -> bool:
    """
    YOLO bounding-box values should
    normally be between 0 and 1.
    """

    return 0.0 <= value <= 1.0


# ============================================================
# VALIDATE ONE LABEL FILE
# ============================================================

def validate_label_file(
    label_path: Path,
    num_classes: int,
) -> tuple[bool, list[str]]:
    """
    Validate a YOLO label file.

    Expected format:

    class_id x_center y_center width height
    """

    errors = []

    try:

        lines = label_path.read_text(
            encoding="utf-8"
        ).splitlines()

    except Exception as error:

        return (
            False,
            [
                f"Could not read "
                f"{label_path.name}: {error}"
            ],
        )

    for line_number, line in enumerate(
        lines,
        start=1,
    ):

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        # ----------------------------------------------------
        # YOLO detection label must contain 5 values
        # ----------------------------------------------------

        if len(parts) != YOLO_VALUES_PER_LINE:

            errors.append(
                f"{label_path.name}, "
                f"line {line_number}: "
                f"expected 5 values, "
                f"got {len(parts)}."
            )

            continue

        # ----------------------------------------------------
        # CLASS ID
        # ----------------------------------------------------

        try:

            class_id = int(parts[0])

        except ValueError:

            errors.append(
                f"{label_path.name}, "
                f"line {line_number}: "
                "class ID must be integer."
            )

            continue

        if not 0 <= class_id < num_classes:

            errors.append(
                f"{label_path.name}, "
                f"line {line_number}: "
                f"invalid class ID {class_id}."
            )

        # ----------------------------------------------------
        # BOUNDING BOX VALUES
        # ----------------------------------------------------

        try:

            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

        except ValueError:

            errors.append(
                f"{label_path.name}, "
                f"line {line_number}: "
                "bounding-box values must be numbers."
            )

            continue

        coordinates = {
            "x_center": x_center,
            "y_center": y_center,
            "width": width,
            "height": height,
        }

        # ----------------------------------------------------
        # CHECK NORMALIZED RANGE
        # ----------------------------------------------------

        for name, value in coordinates.items():

            if not is_normalized_coordinate(value):

                errors.append(
                    f"{label_path.name}, "
                    f"line {line_number}: "
                    f"{name}={value} "
                    "is outside [0, 1]."
                )

        # ----------------------------------------------------
        # WIDTH AND HEIGHT CANNOT BE ZERO
        # ----------------------------------------------------

        if width <= 0:

            errors.append(
                f"{label_path.name}, "
                f"line {line_number}: "
                "width must be greater than 0."
            )

        if height <= 0:

            errors.append(
                f"{label_path.name}, "
                f"line {line_number}: "
                "height must be greater than 0."
            )

    return len(errors) == 0, errors


# ============================================================
# CHECK IMAGE-LABEL PAIRS
# ============================================================

def validate_image_label_pairs(
    image_dir: Path,
    label_dir: Path,
) -> list[str]:
    """
    Check that every image has a
    corresponding YOLO label.
    """

    errors = []

    image_files = get_image_files(
        image_dir
    )

    label_files = get_label_files(
        label_dir
    )

    image_names = {
        file_path.stem
        for file_path in image_files
    }

    label_names = {
        file_path.stem
        for file_path in label_files
    }

    # --------------------------------------------------------
    # IMAGES WITHOUT LABEL FILES
    # --------------------------------------------------------

    missing_labels = (
        image_names - label_names
    )

    for name in sorted(missing_labels):

        errors.append(
            f"Image '{name}' "
            "does not have a label file."
        )

    # --------------------------------------------------------
    # LABELS WITHOUT IMAGES
    # --------------------------------------------------------

    missing_images = (
        label_names - image_names
    )

    for name in sorted(missing_images):

        errors.append(
            f"Label '{name}' "
            "does not have an image file."
        )

    return errors


# ============================================================
# VALIDATE ONE DATASET SPLIT
# ============================================================

def validate_split(
    split_name: str,
    image_dir: Path,
    label_dir: Path,
    num_classes: int,
) -> dict[str, Any]:
    """
    Validate one dataset split:
    train, valid or test.
    """

    errors = []

    # --------------------------------------------------------
    # CHECK DIRECTORIES
    # --------------------------------------------------------

    if not image_dir.exists():

        errors.append(
            f"Missing image directory: "
            f"{image_dir}"
        )

    if not label_dir.exists():

        errors.append(
            f"Missing label directory: "
            f"{label_dir}"
        )

    if errors:

        return {
            "split": split_name,
            "valid": False,
            "image_count": 0,
            "label_count": 0,
            "errors": errors,
        }

    # --------------------------------------------------------
    # GET FILES
    # --------------------------------------------------------

    image_files = get_image_files(
        image_dir
    )

    label_files = get_label_files(
        label_dir
    )

    # --------------------------------------------------------
    # IMAGE-LABEL PAIR CHECK
    # --------------------------------------------------------

    pair_errors = validate_image_label_pairs(
        image_dir=image_dir,
        label_dir=label_dir,
    )

    errors.extend(pair_errors)

    # --------------------------------------------------------
    # IMAGE VALIDATION
    # --------------------------------------------------------

    for image_path in image_files:

        valid, message = validate_image(
            image_path
        )

        if not valid:
            errors.append(message)

    # --------------------------------------------------------
    # LABEL VALIDATION
    # --------------------------------------------------------

    for label_path in label_files:

        valid, label_errors = (
            validate_label_file(
                label_path=label_path,
                num_classes=num_classes,
            )
        )

        if not valid:
            errors.extend(label_errors)

    # --------------------------------------------------------
    # FINAL SPLIT RESULT
    # --------------------------------------------------------

    return {
        "split": split_name,
        "valid": len(errors) == 0,
        "image_count": len(image_files),
        "label_count": len(label_files),
        "errors": errors,
    }