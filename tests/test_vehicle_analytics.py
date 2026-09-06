import pytest

from app.detection.analytics import (
    calculate_vehicle_statistics,
    create_parking_summary,
)


# ============================================================
# VEHICLE COUNT
# ============================================================

def test_vehicle_statistics():

    counts = {
        "car": 10,
        "motorcycle": 2,
        "bus": 1,
        "truck": 1,
    }

    result = (
        calculate_vehicle_statistics(
            counts
        )
    )

    assert (
        result["total_vehicles"]
        == 14
    )

    assert (
        result[
            "vehicle_counts"
        ]["car"]
        == 10
    )


# ============================================================
# MISSING CLASS
# ============================================================

def test_missing_vehicle_classes():

    counts = {
        "car": 5,
    }

    result = (
        calculate_vehicle_statistics(
            counts
        )
    )

    assert (
        result["total_vehicles"]
        == 5
    )

    assert (
        result[
            "vehicle_counts"
        ]["truck"]
        == 0
    )


# ============================================================
# NEGATIVE COUNT
# ============================================================

def test_negative_vehicle_count():

    counts = {
        "car": -1,
    }

    with pytest.raises(
        ValueError
    ):

        calculate_vehicle_statistics(
            counts
        )


# ============================================================
# COMBINED SUMMARY
# ============================================================

def test_create_parking_summary():

    parking = {
        "total_spaces": 20,
        "empty_spaces": 6,
        "occupied_spaces": 14,
        "occupancy_percentage": 70.0,
        "availability_percentage": 30.0,
    }

    vehicles = {
        "total_vehicles": 13,
        "vehicle_counts": {
            "car": 12,
            "motorcycle": 1,
            "bus": 0,
            "truck": 0,
        },
    }

    summary = (
        create_parking_summary(
            parking_statistics=parking,
            vehicle_statistics=vehicles,
        )
    )

    assert (
        summary["parking"]
        == parking
    )

    assert (
        summary["vehicles"]
        == vehicles
    )