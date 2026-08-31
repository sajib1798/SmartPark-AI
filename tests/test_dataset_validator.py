from pathlib import Path

from app.data.dataset_validator import (
    is_normalized_coordinate,
    validate_label_file,
)


# ============================================================
# NORMALIZED COORDINATES
# ============================================================

def test_normalized_coordinate_valid():

    assert is_normalized_coordinate(0.0)
    assert is_normalized_coordinate(0.5)
    assert is_normalized_coordinate(1.0)


def test_normalized_coordinate_invalid():

    assert not is_normalized_coordinate(-0.1)
    assert not is_normalized_coordinate(1.1)


# ============================================================
# VALID LABEL
# ============================================================

def test_valid_yolo_label(
    tmp_path: Path,
):

    label_file = (
        tmp_path / "sample.txt"
    )

    label_file.write_text(
        "0 0.5 0.5 0.2 0.3\n"
        "1 0.4 0.6 0.1 0.2\n",
        encoding="utf-8",
    )

    valid, errors = validate_label_file(
        label_path=label_file,
        num_classes=2,
    )

    assert valid is True
    assert errors == []


# ============================================================
# INVALID CLASS
# ============================================================

def test_invalid_class_id(
    tmp_path: Path,
):

    label_file = (
        tmp_path / "sample.txt"
    )

    label_file.write_text(
        "5 0.5 0.5 0.2 0.3\n",
        encoding="utf-8",
    )

    valid, errors = validate_label_file(
        label_path=label_file,
        num_classes=2,
    )

    assert valid is False
    assert len(errors) > 0


# ============================================================
# INVALID COORDINATE
# ============================================================

def test_invalid_coordinate(
    tmp_path: Path,
):

    label_file = (
        tmp_path / "sample.txt"
    )

    label_file.write_text(
        "0 1.5 0.5 0.2 0.3\n",
        encoding="utf-8",
    )

    valid, errors = validate_label_file(
        label_path=label_file,
        num_classes=2,
    )

    assert valid is False
    assert len(errors) > 0


# ============================================================
# WRONG YOLO FORMAT
# ============================================================

def test_invalid_yolo_format(
    tmp_path: Path,
):

    label_file = (
        tmp_path / "sample.txt"
    )

    label_file.write_text(
        "0 0.5 0.5\n",
        encoding="utf-8",
    )

    valid, errors = validate_label_file(
        label_path=label_file,
        num_classes=2,
    )

    assert valid is False
    assert len(errors) > 0