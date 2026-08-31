import json

from app.config import settings


# ============================================================
# LOAD PARKING SLOT CONFIGURATION
# ============================================================

def load_parking_slots() -> dict:
    """
    Load parking slot configuration
    from the JSON file.
    """

    file_path = settings.PARKING_SLOTS_FILE

    if not file_path.exists():

        raise FileNotFoundError(
            f"Parking slot configuration "
            f"not found: {file_path}"
        )

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    return data


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Main entry point for SmartPark AI.
    """

    print("=" * 60)

    print(settings.APP_NAME)

    print(
        f"Version: {settings.VERSION}"
    )

    print("=" * 60)

    parking_data = load_parking_slots()

    camera_name = parking_data.get(
        "camera_name",
        "Unknown Camera",
    )

    slots = parking_data.get(
        "slots",
        [],
    )

    print(
        f"Camera: {camera_name}"
    )

    print(
        f"Configured parking slots: "
        f"{len(slots)}"
    )

    for slot in slots:

        print(
            f"Slot ID: {slot['id']} "
            f"| Name: {slot['name']}"
        )

    print()

    print(
        "SmartPark AI started successfully."
    )


if __name__ == "__main__":
    main()