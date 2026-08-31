from collections import Counter
from pathlib import Path
from typing import Any

from app.data.dataset_validator import (
    get_label_files,
)


# ============================================================
# COUNT OBJECTS BY CLASS
# ============================================================

def count_objects_by_class(
    label_dir: Path,
) -> dict[int, int]:
    """
    Count bounding boxes belonging
    to each class.
    """

    class_counter: Counter[int] = Counter()

    label_files = get_label_files(
        label_dir
    )

    for label_path in label_files:

        lines = label_path.read_text(
            encoding="utf-8"
        ).splitlines()

        for line in lines:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                continue

            try:

                class_id = int(parts[0])

            except ValueError:
                continue

            class_counter[class_id] += 1

    return dict(
        sorted(class_counter.items())
    )


# ============================================================
# CREATE HUMAN READABLE STATISTICS
# ============================================================

def create_split_statistics(
    label_dir: Path,
    class_names: list[str],
) -> dict[str, Any]:
    """
    Generate class statistics
    for a dataset split.
    """

    counts = count_objects_by_class(
        label_dir
    )

    class_statistics = {}

    for class_id, class_name in enumerate(
        class_names
    ):

        class_statistics[class_name] = (
            counts.get(
                class_id,
                0,
            )
        )

    total_objects = sum(
        class_statistics.values()
    )

    return {
        "total_objects": total_objects,
        "class_counts": class_statistics,
    }