from typing import Any


# ============================================================
# PARKING OCCUPANCY STATISTICS
# ============================================================

def calculate_occupancy_statistics(
    empty_count: int,
    occupied_count: int,
) -> dict[str, Any]:
    """
    Calculate parking occupancy statistics.
    """

    if empty_count < 0:

        raise ValueError(
            "empty_count cannot be negative."
        )

    if occupied_count < 0:

        raise ValueError(
            "occupied_count cannot be negative."
        )

    total_spaces = (
        empty_count
        + occupied_count
    )

    if total_spaces == 0:

        occupancy_percentage = 0.0
        availability_percentage = 0.0

    else:

        occupancy_percentage = (
            occupied_count
            / total_spaces
            * 100
        )

        availability_percentage = (
            empty_count
            / total_spaces
            * 100
        )

    return {
        "total_spaces": total_spaces,
        "empty_spaces": empty_count,
        "occupied_spaces": occupied_count,
        "occupancy_percentage": round(
            occupancy_percentage,
            2,
        ),
        "availability_percentage": round(
            availability_percentage,
            2,
        ),
    }


# ============================================================
# VEHICLE SUMMARY
# ============================================================

def calculate_vehicle_statistics(
    vehicle_counts: dict[str, int],
) -> dict[str, Any]:
    """
    Calculate total vehicle statistics.
    """

    expected_classes = [
        "car",
        "motorcycle",
        "bus",
        "truck",
    ]

    normalized_counts = {}

    for class_name in expected_classes:

        value = vehicle_counts.get(
            class_name,
            0,
        )

        if value < 0:

            raise ValueError(
                f"{class_name} count "
                "cannot be negative."
            )

        normalized_counts[
            class_name
        ] = value

    total_vehicles = sum(
        normalized_counts.values()
    )

    return {
        "total_vehicles": total_vehicles,
        "vehicle_counts": normalized_counts,
    }


# ============================================================
# COMBINED SMART PARKING SUMMARY
# ============================================================

def create_parking_summary(
    parking_statistics: dict[str, Any],
    vehicle_statistics: dict[str, Any],
) -> dict[str, Any]:
    """
    Combine occupancy and vehicle
    statistics into one response.
    """

    return {
        "parking": parking_statistics,
        "vehicles": vehicle_statistics,
    }