import pytest

from app.detection.analytics import (
    calculate_occupancy_statistics,
)


# ============================================================
# NORMAL CASE
# ============================================================

def test_occupancy_statistics():

    result = (
        calculate_occupancy_statistics(
            empty_count=6,
            occupied_count=14,
        )
    )

    assert (
        result["total_spaces"]
        == 20
    )

    assert (
        result["empty_spaces"]
        == 6
    )

    assert (
        result["occupied_spaces"]
        == 14
    )

    assert (
        result[
            "occupancy_percentage"
        ]
        == 70.0
    )

    assert (
        result[
            "availability_percentage"
        ]
        == 30.0
    )


# ============================================================
# EMPTY PARKING LOT
# ============================================================

def test_all_empty():

    result = (
        calculate_occupancy_statistics(
            empty_count=10,
            occupied_count=0,
        )
    )

    assert (
        result["total_spaces"]
        == 10
    )

    assert (
        result[
            "occupancy_percentage"
        ]
        == 0.0
    )

    assert (
        result[
            "availability_percentage"
        ]
        == 100.0
    )


# ============================================================
# FULL PARKING LOT
# ============================================================

def test_all_occupied():

    result = (
        calculate_occupancy_statistics(
            empty_count=0,
            occupied_count=10,
        )
    )

    assert (
        result[
            "occupancy_percentage"
        ]
        == 100.0
    )

    assert (
        result[
            "availability_percentage"
        ]
        == 0.0
    )


# ============================================================
# ZERO DETECTIONS
# ============================================================

def test_zero_spaces():

    result = (
        calculate_occupancy_statistics(
            empty_count=0,
            occupied_count=0,
        )
    )

    assert (
        result["total_spaces"]
        == 0
    )

    assert (
        result[
            "occupancy_percentage"
        ]
        == 0.0
    )


# ============================================================
# NEGATIVE VALUES
# ============================================================

def test_negative_empty_count():

    with pytest.raises(
        ValueError
    ):

        calculate_occupancy_statistics(
            empty_count=-1,
            occupied_count=5,
        )


def test_negative_occupied_count():

    with pytest.raises(
        ValueError
    ):

        calculate_occupancy_statistics(
            empty_count=5,
            occupied_count=-1,
        )